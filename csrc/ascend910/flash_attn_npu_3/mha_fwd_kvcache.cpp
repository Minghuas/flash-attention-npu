/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "online_softmax.hpp"
#include "rescale_o.hpp"
#include "init_outputs.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "pv_matmul.hpp"
#include "qk_matmul.hpp"
#include "combine_scale.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "fa_block.h"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "kernel_common.hpp"
#include "kernel_operator.h"
#include "tilingdata.h"
#include "fa_metadata_args.h"

using namespace Catlass;
using namespace KernelCommon;

namespace SplitFuse {
template <class BlockMmadQK, class BlockMmadPV, class EpilogueOnlineSoftmax, class EpilogueRescaleO,
          bool PAGED_CACHE_FLAG, bool FLASH_DECODE_FLAG,
          FaiKenel::MaskType MASK_TYPE = FaiKenel::MaskType::NO_MASK,
          FaiKenel::inputLayout INPUT_LAYOUT = FaiKenel::inputLayout::BSND, class CombineScale = void>
class FAInferKernel {
  public:
    bool hasSeqUsedQ = false;
    bool hasSeqUsedKv = false;
    using ArchTag = typename BlockMmadQK::ArchTag;
    using L1TileShape = typename BlockMmadQK::L1TileShape;
    using ElementQ = typename BlockMmadQK::ElementA;
    using LayoutQ = typename BlockMmadQK::LayoutA;
    using ElementK = typename BlockMmadQK::ElementB;
    using LayoutK = typename BlockMmadQK::LayoutB;
    using ElementS = typename BlockMmadQK::ElementC;
    using LayoutS = typename BlockMmadQK::LayoutC;

    using ElementP = typename BlockMmadPV::ElementA;
    using LayoutP = typename BlockMmadPV::LayoutA;
    using ElementV = typename BlockMmadPV::ElementB;
    using LayoutV = typename BlockMmadPV::LayoutB;

    using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;
    using LayoutMask = typename EpilogueOnlineSoftmax::LayoutMask;

    using ElementO = typename EpilogueRescaleO::ElementOutput;
    using LayoutO = typename EpilogueRescaleO::LayoutOutput;

    using ElementOTmp = typename EpilogueRescaleO::ElementInput;
    using LayoutOTmp = typename EpilogueRescaleO::LayoutInput;

    using ElementLse = typename EpilogueRescaleO::ElementLse;
    using LayoutLse = typename EpilogueRescaleO::LayoutLse;

    using ElementUpdate = typename EpilogueRescaleO::ElementUpdate;
    using LayoutUpdate = typename EpilogueRescaleO::LayoutUpdate;

    static constexpr Epilogue::LseModeT LSE_MODE = EpilogueRescaleO::LSE_MODE;

    // SWA empty-window tiles: init output (0) and lse (inf) without running attention.
    using InitOutDispatchPolicy = Epilogue::EpilogueAtlasA2InitOutWhenZero<LSE_MODE>;
    using OTypeInit = Gemm::GemmType<ElementO, LayoutO>;
    using LseTypeInit = Gemm::GemmType<ElementLse, LayoutLse>;
    using EpilogueInitOut = Epilogue::Block::BlockEpilogue<InitOutDispatchPolicy, OTypeInit, LseTypeInit>;

    static constexpr uint32_t STACK_SLOTS = PRE_LAUNCH + 1;
    static constexpr uint32_t MAX_STACK_BLOCKS = MAX_KV_STACK_LEN / 128;
    static constexpr uint32_t SCORE_SLOT_ELEMENTS = Q_TILE_CEIL * MAX_KV_STACK_LEN;

    struct StackDescriptor {
        uint64_t gmOffsetV;
        uint64_t gmOffsetVSrc;
        uint64_t pvCacheOffset;
        bool isAppendBlock;
        bool doCopyback;
        uint64_t gmOffsetO;
        uint64_t gmOffsetLse;
        // Delayed rescale must retain the producing task's split destination.
        uint64_t gmOffsetLseFD;
        uint64_t gmOffsetOFD;
        bool isSplitKV;
        uint64_t blockBOffset;
        uint32_t rowNum;
        uint32_t stackSeqTile;
        uint32_t qSeqlen;
        uint32_t qSBlockSize;
        uint32_t qNBlockSize;
        uint32_t kvSIdx;
        uint32_t kvSLoopNumTotal;
        uint32_t noSkipKvS;
        uint32_t slot;
        uint32_t taskStateSlot;
        bool firstStack;
        bool lastStack;
        int32_t invalidSuffixStartRow;
        int32_t invalidPrefixEndRow;
        int32_t qSBlockIdx;
        int32_t curQNBlockTile;
    };

    struct BatchRuntime {
        uint64_t qBOffset;
        uint64_t kBOffset;
        uint64_t vBOffset;
        uint64_t oBOffset;
        uint64_t lseBOffset;
        uint64_t blockBOffset;
        uint64_t kNewBOffset;
        uint64_t vNewBOffset;
        uint32_t lseHeadStride;
        uint32_t curQSBlockTile;
        uint32_t curQSBlockNum;
        uint32_t curKSBlockNum;
        uint32_t appendKVFlag;
        uint32_t kvSeqlenOld;
        uint32_t kvStartOld;
        uint32_t kvLoopNumTotalNew;
        uint32_t kvNewSeqlen;
    };

    // Task-local state shared by the prefill and steady-state stack paths.
    // It deliberately excludes the mutable core-global issue counter: callers
    // own descriptor ordering and decide whether a submitted stack is consumed.
    struct StackRunContext {
        uint32_t coreIdx;
        uint32_t qSeqlen;
        uint32_t kvSeqlen;
        uint32_t qNBlockSize;
        uint32_t qNBlockTile;
        uint32_t qSBlockIdx;
        uint32_t taskFlags;
        bool isSplitKV;
        bool appendKVFlag;
        uint32_t kvSeqlenOld;
        uint32_t kvStartOld;
        uint32_t kvLoopNumTotalNew;
        uint32_t kvNewSeqlen;
        uint64_t gmOffsetQ;
        uint64_t gmOffsetK;
        uint64_t gmOffsetV;
        uint64_t gmOffsetKNew;
        uint64_t gmOffsetVNew;
        uint64_t gmOffsetO;
        uint64_t gmOffsetLse;
        uint64_t gmOffsetLseFD;
        uint64_t gmOffsetOFD;
        uint64_t blockBOffset;
        uint32_t qSBlockSize;
        uint32_t noSkipKvS;
        uint32_t kvSLoopNumTotal;
        uint32_t kvSLoopNumTotalOld;
        uint32_t kvStart;
        uint32_t kvEnd;
        int32_t windowSizeLeftStartLen;
        int32_t windowSizeLeftEndLen;
        int32_t windowSizeRightStartLen;
        int32_t windowSizeRightEndLen;
        bool notPreMask;
        bool notNextMask;
        int32_t invalidSuffixStartRow;
        int32_t invalidPrefixEndRow;
        uint32_t currentTaskStateSlot;
    };

    // Per-batch Q length: seqused_q when provided, else cu prefix diff.
    __aicore__ inline uint32_t QSeqLenOf(uint32_t bIdx, AscendC::GlobalTensor<int32_t>& gCuSeq,
                                         AscendC::GlobalTensor<int32_t>& gSeqUsed) const
    {
        if (hasSeqUsedQ) {
            return static_cast<uint32_t>(gSeqUsed.GetValue(bIdx));
        }
        return static_cast<uint32_t>(gCuSeq.GetValue(bIdx + 1) - gCuSeq.GetValue(bIdx));
    }

    // Per-batch KV length: seqused_k when provided, else cu prefix diff.
    __aicore__ inline uint32_t KvSeqLenOf(uint32_t bIdx, AscendC::GlobalTensor<int32_t>& gCuSeq,
                                          AscendC::GlobalTensor<int32_t>& gSeqUsed) const
    {
        if (hasSeqUsedKv) {
            return static_cast<uint32_t>(gSeqUsed.GetValue(bIdx));
        }
        return static_cast<uint32_t>(gCuSeq.GetValue(bIdx + 1) - gCuSeq.GetValue(bIdx));
    }

    struct GlobalTensorBundle {
        AscendC::GlobalTensor<ElementQ>& gQ;
        AscendC::GlobalTensor<ElementK>& gK;
        AscendC::GlobalTensor<ElementK>& gV;
        AscendC::GlobalTensor<ElementMask>& gMask;
        AscendC::GlobalTensor<int32_t>& gBlockTable;
        AscendC::GlobalTensor<int32_t>& gActualQseqlen;
        AscendC::GlobalTensor<int32_t>& gActualKvseqlen;
        AscendC::GlobalTensor<int32_t>& gSeqUsedQ;
        AscendC::GlobalTensor<int32_t>& gSeqUsedKv;
        AscendC::GlobalTensor<ElementO>& gO;
        AscendC::GlobalTensor<ElementLse>& gLse;
        AscendC::GlobalTensor<ElementLse>& gLseFD;
        AscendC::GlobalTensor<ElementLse>& gOFD;
        AscendC::GlobalTensor<ElementS>& gS;
        AscendC::GlobalTensor<ElementP>& gP;
        AscendC::GlobalTensor<ElementOTmp>& gOTmp;
        AscendC::GlobalTensor<ElementOTmp>& gOUpdate;
        AscendC::GlobalTensor<ElementK>& gKNew;
        AscendC::GlobalTensor<ElementK>& gVNew;
    };

    __aicore__ inline FAInferKernel() {}

    __aicore__ inline void operator()(FAIKernelParams const& params)
    {
        hasSeqUsedQ = params.seqUsedQ != nullptr;
        hasSeqUsedKv = params.seqUsedKv != nullptr;
        __gm__ FAInferTilingData* fATilingData =
            reinterpret_cast<__gm__ FAInferTilingData*>(params.tiling.tilingDataAddr);
        mm1OutSize = fATilingData->mm1OutSize;
        smOnlineOutSize = fATilingData->smOnlineOutSize;
        mm2OutSize = fATilingData->mm2OutSize;
        batch = fATilingData->batch;
        qHeads = fATilingData->numHeads;
        kvHeads = fATilingData->kvHeads;
        embed = fATilingData->embeddingSize;
        embedV = fATilingData->embeddingSizeV;
        pagedBlockSize = fATilingData->blockSize;
        maxNumBlocksPerBatch = fATilingData->maxNumBlocksPerBatch;
        blockSize = fATilingData->blockSize;
        maskType = fATilingData->maskType;
        windowSizeLeft = fATilingData->windowSizeLeft;
        windowSizeRight = fATilingData->windowSizeRight;
        scaleValue = params.tiling.scaleValue;
        softcapValue = params.tiling.softcapValue;
        maxQSeqlen = fATilingData->maxQSeqlen;
        maxKvSeqlen = fATilingData->maxKvSeqlen;
        kvNewSeqlen = fATilingData->kvNewSeqlen;
        kvCacheSeqlen = fATilingData->kvCacheSeqlen;
        kvSeqlenAllocated = params.kvSeqlenAllocated;

        // FD workspace sizing: reserve head of workspace for gLseFD/gOFD.
        uint64_t Lsesize = 0;
        uint64_t Losize = 0;
        if constexpr (FLASH_DECODE_FLAG) {
            Lsesize = fATilingData->splitLseTotalSize;
            Losize = fATilingData->splitOTotalSize;
        }

        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ*)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK*)params.k);
        AscendC::GlobalTensor<ElementK> gV;
        gV.SetGlobalBuffer((__gm__ ElementK*)params.v);
        AscendC::GlobalTensor<ElementMask> gMask;
        gMask.SetGlobalBuffer((__gm__ ElementMask*)params.mask);
        AscendC::GlobalTensor<int32_t> gBlockTable;
        gBlockTable.SetGlobalBuffer((__gm__ int32_t*)(params.blockTables));
        AscendC::GlobalTensor<int32_t> gActualQseqlen;
        gActualQseqlen.SetGlobalBuffer((__gm__ int32_t*)params.actualQseqlen);
        AscendC::GlobalTensor<int32_t> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ int32_t*)params.actualKvseqlen);
        AscendC::GlobalTensor<int32_t> gSeqUsedQ;
        gSeqUsedQ.SetGlobalBuffer((__gm__ int32_t*)params.seqUsedQ);
        AscendC::GlobalTensor<int32_t> gSeqUsedKv;
        gSeqUsedKv.SetGlobalBuffer((__gm__ int32_t*)params.seqUsedKv);
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO*)params.o);
        AscendC::GlobalTensor<ElementLse> gLse;
        gLse.SetGlobalBuffer((__gm__ ElementLse*)params.lse);

        AscendC::GlobalTensor<ElementLse> gLseFD;
        AscendC::GlobalTensor<ElementLse> gOFD;
        if constexpr (FLASH_DECODE_FLAG) {
            gLseFD.SetGlobalBuffer((__gm__ ElementLse*)(params.workSpace));
            gOFD.SetGlobalBuffer((__gm__ ElementLse*)(params.workSpace + Lsesize));
        }

        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS*)(params.workSpace + Lsesize + Losize));
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP*)(params.workSpace + Lsesize + Losize + mm1OutSize));
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer(
            (__gm__ ElementOTmp*)(params.workSpace + Lsesize + Losize + mm1OutSize + smOnlineOutSize));
        AscendC::GlobalTensor<ElementOTmp> gOUpdate;
        gOUpdate.SetGlobalBuffer(
            (__gm__ ElementOTmp*)(params.workSpace + Lsesize + Losize + mm1OutSize + smOnlineOutSize + mm2OutSize));

        // Append-KV newkv tensors (nullptr when kvNewSeqlen == 0).
        AscendC::GlobalTensor<ElementK> gKNew;
        gKNew.SetGlobalBuffer((__gm__ ElementK*)params.kNew);
        AscendC::GlobalTensor<ElementK> gVNew;
        gVNew.SetGlobalBuffer((__gm__ ElementK*)params.vNew);
        // Append-KV per-batch cache-batch indices / left padding (GM params).

        GlobalTensorBundle globalTensors{
            gQ,   gK,     gV,   gMask, gBlockTable, gActualQseqlen, gActualKvseqlen, gSeqUsedQ, gSeqUsedKv, gO,
            gLse, gLseFD, gOFD, gS,    gP,          gOTmp,          gOUpdate,        gKNew,     gVNew};

        if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::BNSD) {
            // BNSD: (batch, head, seq, dim). Adjacent tokens within a head are
            // `embed`/`embedV` apart; adjacent heads are maxQSeqlen*embed(V) apart
            // for Q/O and kvSeqlenAllocated*embed(V) apart for the KV cache.
            strideQ = static_cast<uint64_t>(embed);
            strideO = static_cast<uint64_t>(embedV);
            strideK = static_cast<uint64_t>(embed);
            strideV = static_cast<uint64_t>(embedV);
            qHeadStride = static_cast<uint64_t>(maxQSeqlen) * embed;
            oHeadStride = static_cast<uint64_t>(maxQSeqlen) * embedV;
            kvHeadStrideK = static_cast<uint64_t>(kvSeqlenAllocated) * embed;
            kvHeadStrideV = static_cast<uint64_t>(kvSeqlenAllocated) * embedV;
        } else {
            // BSND / TND: (batch|token, seq, head, dim). Adjacent heads are
            // `embed`/`embedV` apart; adjacent tokens are qHeads*embed(V) (Q/O) or
            // kvHeads*embed(V) (KV).
            strideQ = static_cast<uint64_t>(qHeads * embed);
            strideO = static_cast<uint64_t>(qHeads * embedV);
            strideK = static_cast<uint64_t>(kvHeads * embed);
            strideV = static_cast<uint64_t>(kvHeads * embedV);
            qHeadStride = static_cast<uint64_t>(embed);
            oHeadStride = static_cast<uint64_t>(embedV);
            kvHeadStrideK = static_cast<uint64_t>(embed);
            kvHeadStrideV = static_cast<uint64_t>(embedV);
        }

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
#ifdef __DAV_C220_CUBE__
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);

        const bool headDimSplit = embed > NUM_256;
        groupSize = qHeads / kvHeads;
        const bool compactDecodeL1 = !FLASH_DECODE_FLAG && embed == 512U && embedV == 512U && kvNewSeqlen == 0U &&
                                     maxQSeqlen * groupSize <= 64U;
        kvStackLen = MAX_KV_STACK_LEN;
        uint32_t kDynNum = NUM_256;
        uint32_t nDynNum = L1_D512_KV_TILE;
        uint32_t qL1DynNum = RoundUp(embed, FaiKenel::BLOCK_SIZE);
        uint32_t kPVDynNum = NUM_256;
        uint32_t vL1Stages = DOUBLE_BUFFER;
        uint32_t vSegLen = NUM_256;
        uint32_t vSliceWidth = L1_HEAD_DIM_SLICE;
        uint32_t qL1Rows = Q_TILE_CEIL;
        uint32_t pL1Rows = BlockMmadPV::L1TileShape::M;
        uint32_t pL1Stages = DOUBLE_BUFFER;
        if (!headDimSplit) {
            // Preserve the production <=256 layout.
            kDynNum = RoundUp(embed, NUM_128);
            kDynNum = kDynNum < NUM_256 ? NUM_256 : kDynNum;
            uint32_t maxQKPL1Size = L1_MAX_SIZE - embedV * kvStackLen * sizeof(ElementV);
            uint32_t maxQL1Size = Q_TILE_CEIL * kDynNum * sizeof(ElementQ);
            uint32_t maxNDynNum =
                ((maxQKPL1Size - maxQL1Size) / kDynNum / sizeof(ElementV) / DOUBLE_BUFFER) / NUM_32 * NUM_32;
            nDynNum = maxNDynNum < L1_MAX_N_NUM ? maxNDynNum : L1_MAX_N_NUM;
            nDynNum = L1_MAX_N_NUM % nDynNum != 0 ? RoundDown((nDynNum - 1), NUM_32) : nDynNum;
            qL1DynNum = kDynNum;
            kPVDynNum = nDynNum * kDynNum / BlockMmadPV::L1TileShape::M;
            vL1Stages = 1U;
            vSegLen = kvStackLen;
            vSliceWidth = embedV;
        } else if (compactDecodeL1) {
            qL1Rows = 64U;
            pL1Rows = 64U;
            pL1Stages = 1U;
            kPVDynNum = kvStackLen;
        }

        blockMmadQK.SetPingPongState(&pingPongState);
        blockMmadPV.SetPingPongState(&pingPongState);

        uint32_t L1_QK_SIZE = qL1Rows * qL1DynNum * sizeof(ElementQ);
        uint32_t qkKPoolSize = nDynNum * kDynNum * sizeof(ElementK) * DOUBLE_BUFFER;
        uint32_t kPVPoolSize = pL1Rows * kPVDynNum * sizeof(ElementP) * pL1Stages;
        uint32_t vPoolSize = vL1Stages * vSegLen * vSliceWidth * sizeof(ElementV);
        // Append-KV QK writeback staging (ND copy of the new-K sub-tile), placed after
        // the QK/PV shared K/P region and the PV V region (the K and P tiles share one
        // L1 region across the QK/PV phases, so it must not be counted twice).
        uint32_t pBufOffset = compactDecodeL1 ? (L1_QK_SIZE + qkKPoolSize) : L1_QK_SIZE;
        uint32_t vBufOffset = pBufOffset + kPVPoolSize;
        uint32_t ndCopyL1Offset = compactDecodeL1 ? (vBufOffset + vPoolSize) : (L1_QK_SIZE + kPVPoolSize + vPoolSize);
        blockMmadQK.init(resource, nDynNum, kDynNum, qL1Rows, qL1DynNum, kvStackLen, 0, ndCopyL1Offset);
        if (compactDecodeL1) {
            blockMmadPV.initCompactDecode(resource, pL1Rows, kvStackLen, pBufOffset, vBufOffset);
        } else {
            blockMmadPV.init(resource, nDynNum, kPVDynNum, kvStackLen, L1_QK_SIZE, vL1Stages, vSegLen, vSliceWidth,
                             ndCopyL1Offset);
        }
#endif
#ifdef __DAV_C220_VEC__
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);

        epilogueOnlineSoftmax.init(resource, scaleValue, softcapValue);
        epilogueRescaleO.init(resource);

        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif
        uint32_t softmaxPingPongFlag = 0;
        embedRound = RoundUp(embed, FaiKenel::BLOCK_SIZE);
        embedRoundV = RoundUp(embedV, FaiKenel::BLOCK_SIZE);
        groupSize = qHeads / kvHeads;

        totalQTokens = 0;
        if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
            totalQTokens = static_cast<uint32_t>(gActualQseqlen.GetValue(batch));
        }

        StackDescriptor descriptors[STACK_SLOTS];
        uint32_t issuedStackCount = 0;
        uint32_t prefillRemaining = PRE_LAUNCH;
        uint32_t taskStateSequence = 0;

        const auto* schedulerHeader = reinterpret_cast<const __gm__ SchedulerHeader*>(
            reinterpret_cast<const __gm__ uint8_t*>(fATilingData) + fa_metadata::SCHEDULER_HEADER_OFFSET);
        const auto* taskRanges = reinterpret_cast<const __gm__ CoreTaskRange*>(
            reinterpret_cast<const __gm__ uint8_t*>(fATilingData) + fa_metadata::CORE_RANGE_OFFSET);
        const auto* taskList = reinterpret_cast<const __gm__ TaskDesc*>(
            reinterpret_cast<const __gm__ uint8_t*>(fATilingData) + schedulerHeader->taskDataOffset);
        const auto* batchPlans = reinterpret_cast<const __gm__ BatchPlan*>(
            reinterpret_cast<const __gm__ uint8_t*>(fATilingData) + schedulerHeader->batchPlanOffset);
        const auto* headPlans = reinterpret_cast<const __gm__ HeadPlan*>(
            reinterpret_cast<const __gm__ uint8_t*>(fATilingData) + schedulerHeader->headPlanOffset);
        const auto* fdPlans = reinterpret_cast<const __gm__ FDPlan*>(
            reinterpret_cast<const __gm__ uint8_t*>(fATilingData) + schedulerHeader->fdPlanOffset);
        const auto* maskPlans = reinterpret_cast<const __gm__ MaskPlan*>(
            reinterpret_cast<const __gm__ uint8_t*>(fATilingData) + schedulerHeader->maskPlanOffset);
        const auto* appendPlans = reinterpret_cast<const __gm__ AppendPlan*>(
            reinterpret_cast<const __gm__ uint8_t*>(fATilingData) + schedulerHeader->appendPlanOffset);
        const auto* addressPlans = reinterpret_cast<const __gm__ AddressPlan*>(
            reinterpret_cast<const __gm__ uint8_t*>(fATilingData) + schedulerHeader->addressPlanOffset);

        const bool idleCoreFD =
            FLASH_DECODE_FLAG && (fATilingData->needCoreNum != 0U) && (coreIdx >= fATilingData->needCoreNum);
        if (schedulerHeader->status == fa_metadata::TASK_METADATA_OK &&
            coreIdx < schedulerHeader->coreNum && !idleCoreFD) {
            const __gm__ CoreTaskRange* taskRange = &taskRanges[coreIdx];
            uint32_t cachedBatchIdx = UINT32_MAX;
            const __gm__ BatchPlan* cachedBatchPlan = nullptr;
            const __gm__ AppendPlan* cachedAppendPlan = nullptr;
            const __gm__ AddressPlan* cachedAddressPlan = nullptr;
            BatchRuntime batchRuntime{};
            uint32_t cachedMaskPlanIndex = UINT32_MAX;
            const __gm__ MaskPlan* cachedMaskPlan = nullptr;
            const __gm__ TaskDesc* drainTask = nullptr;
            const uint32_t blockStackNum = (MAX_KV_STACK_LEN - 1 + pagedBlockSize) / pagedBlockSize;
            for (uint32_t localTaskIdx = 0; localTaskIdx < taskRange->taskCount; ++localTaskIdx) {
                const __gm__ TaskDesc* task = &taskList[taskRange->taskOffset + localTaskIdx];
                if (task->batchIdx != cachedBatchIdx) {
                    cachedBatchIdx = task->batchIdx;
                    cachedBatchPlan = task->batchIdx < schedulerHeader->batchPlanCount
                        ? &batchPlans[task->batchIdx] : nullptr;
                    cachedAppendPlan = task->batchIdx < schedulerHeader->appendPlanCount
                        ? &appendPlans[task->batchIdx] : nullptr;
                    cachedAddressPlan = task->batchIdx < schedulerHeader->addressPlanCount
                        ? &addressPlans[task->batchIdx] : nullptr;
                    if (cachedBatchPlan != nullptr && cachedAppendPlan != nullptr && cachedAddressPlan != nullptr) {
                        PrepareBatchRuntime(batchRuntime, cachedBatchPlan, cachedAppendPlan, cachedAddressPlan);
                    }
                }
                if (cachedBatchPlan == nullptr || cachedAppendPlan == nullptr || cachedAddressPlan == nullptr) {
                    continue;
                }
                const __gm__ BatchPlan* batchPlan = cachedBatchPlan;
                const uint32_t maskPlanBase = batchPlan->maskPlanBase;
                const bool maskPlanOwned =
                    task->maskPlanIndex >= maskPlanBase &&
                    (task->maskPlanIndex - maskPlanBase) < batchPlan->maskPlanCount &&
                    task->maskPlanIndex < schedulerHeader->maskPlanCount;
                if (!maskPlanOwned) {
                    cachedMaskPlanIndex = UINT32_MAX;
                    cachedMaskPlan = nullptr;
                    continue;
                }
                if (task->maskPlanIndex != cachedMaskPlanIndex) {
                    cachedMaskPlanIndex = task->maskPlanIndex;
                    cachedMaskPlan = &maskPlans[task->maskPlanIndex];
                }
                if (cachedMaskPlan == nullptr) {
                    continue;
                }
                if (task->headPlanIndex >= schedulerHeader->headPlanCount) {
                    continue;
                }
                const __gm__ HeadPlan* headPlan = &headPlans[task->headPlanIndex];
                uint64_t lseOffset = 0;
                uint64_t oOffset = 0;
                if (task->fdPlanIndex != fa_metadata::INVALID_FD_PLAN) {
                    if (task->fdPlanIndex >= schedulerHeader->fdPlanCount) {
                        continue;
                    }
                    const __gm__ FDPlan* fdPlan = &fdPlans[task->fdPlanIndex];
                    lseOffset = fdPlan->lseOffset;
                    oOffset = fdPlan->oOffset;
                }
                const uint32_t currentTaskStateSlot = taskStateSequence % STACK_SLOTS;
                // Save the last valid task context.  The tail drain is emitted
                // after the task loop, but it still consumes descriptors
                // produced by the whole core-local task sequence.
                drainTask = task;
                const bool hasTaskWork = runMainLoop(
                    coreIdx, batchPlan->qSeqlen, batchPlan->kvSeqlen,
                    headPlan->qNStart, headPlan->qNSize, batchPlan->qNBlockTile, headPlan->kvHeadOffset,
                    task->qSBlockIdx, task->flags,
                    task->isSplitKV != 0, task->startS2Idx, task->endS2Idx, lseOffset, oOffset,
                    cachedMaskPlan, batchRuntime, globalTensors, descriptors, currentTaskStateSlot,
                    issuedStackCount, prefillRemaining, softmaxPingPongFlag, blockStackNum);
                taskStateSequence += static_cast<uint32_t>(hasTaskWork);
            }
            // Drain only consumes descriptors already submitted by the task
            // loop.  Keep it outside runMainLoop so the normal KV loop has no
            // drain branch or tail iteration in its hot footprint.
            if (drainTask != nullptr && drainTask->pipelineDrain != 0U && issuedStackCount != 0U) {
                const uint32_t blockStackNum =
                    (MAX_KV_STACK_LEN - 1 + pagedBlockSize) / pagedBlockSize;
                const uint32_t stackSeqTilePad = MAX_KV_STACK_LEN;
                const uint32_t readyDrainCount = AscendC::Std::min(issuedStackCount, PRE_LAUNCH);
                const uint32_t drainCount = AscendC::Std::min(drainTask->pipelineDrain, readyDrainCount);
                uint32_t drainStackCount = issuedStackCount - readyDrainCount;
                for (uint32_t drainIdx = 0; drainIdx < drainCount; ++drainIdx) {
                    StackDescriptor& pvDesc = descriptors[drainStackCount % STACK_SLOTS];
                    ConsumeDelayedPV(coreIdx, pvDesc, blockStackNum, stackSeqTilePad, globalTensors);
                    ++drainStackCount;
                    ++issuedStackCount;
                }
            }
        }

#ifdef __DAV_C220_CUBE__
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
#endif
#ifdef __DAV_C220_VEC__
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
#endif
        AscendC::PipeBarrier<PIPE_ALL>();

        if constexpr (FLASH_DECODE_FLAG) {
            AscendC::SyncAll();
#ifdef __DAV_C220_VEC__
            CombineScale combineScale;
            combineScale.init(resource);
            combineScale(qHeads, fATilingData->totalSplitNodeNum, embedV, fATilingData->splitInfo, gLseFD, gOFD, gO,
                         gActualQseqlen, INPUT_LAYOUT == FaiKenel::inputLayout::TND, maxQSeqlen,
                         (LSE_MODE == Epilogue::LseModeT::OUT_ONLY), gLse, totalQTokens);
#endif
        }
    }

    __aicore__ __attribute__((noinline)) void PrepareBatchRuntime(
        BatchRuntime& runtime, const __gm__ BatchPlan* batchPlan,
        const __gm__ AppendPlan* appendPlan, const __gm__ AddressPlan* addressPlan)
    {
        runtime.qBOffset = addressPlan->qBOffset;
        runtime.kBOffset = addressPlan->kBOffset;
        runtime.vBOffset = addressPlan->vBOffset;
        runtime.oBOffset = addressPlan->oBOffset;
        runtime.lseBOffset = addressPlan->lseBOffset;
        runtime.blockBOffset = addressPlan->blockBOffset;
        runtime.kNewBOffset = addressPlan->kNewBOffset;
        runtime.vNewBOffset = addressPlan->vNewBOffset;
        runtime.lseHeadStride = addressPlan->lseHeadStride;
        runtime.curQSBlockTile = batchPlan->qSBlockTile;
        runtime.curQSBlockNum = batchPlan->qSBlockNum;
        runtime.curKSBlockNum = batchPlan->kSBlockNum;
        runtime.appendKVFlag = (appendPlan->flags & fa_metadata::APPEND_PLAN_ENABLED) != 0U;
        runtime.kvSeqlenOld = appendPlan->kvSeqlenOld;
        runtime.kvStartOld = appendPlan->kvStartOld;
        runtime.kvLoopNumTotalNew = appendPlan->kvLoopNumTotalNew;
        runtime.kvNewSeqlen = appendPlan->kvNewSeqlen;
    }

    // Consume one delayed PV descriptor.  Keeping the consumer in one outlined
    // body makes the steady-state path and the tail-drain path share the same
    // PV/rescale code instead of growing two copies in runMainLoop.
    __aicore__ inline void ConsumeDelayedPV(
        uint32_t coreIdx, StackDescriptor& pvDesc, uint32_t blockStackNum,
        uint32_t stackSeqTilePad, GlobalTensorBundle& globalTensors)
    {
        auto& gBlockTable = globalTensors.gBlockTable;
        auto& gO = globalTensors.gO;
        auto& gLse = globalTensors.gLse;
        auto& gLseFD = globalTensors.gLseFD;
        auto& gOFD = globalTensors.gOFD;
        auto& gP = globalTensors.gP;
        auto& gOTmp = globalTensors.gOTmp;
        auto& gOUpdate = globalTensors.gOUpdate;
        auto& gV = globalTensors.gV;
        auto& gVNew = globalTensors.gVNew;

        uint64_t gmOffsetOTmp = static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * STACK_SLOTS +
                                                       pvDesc.slot * WORKSPACE_BLOCK_SIZE_DB);
        GemmCoord actualBlockShapePV{pvDesc.rowNum, embedV, pvDesc.stackSeqTile};
        LayoutOTmp layoutOTmp(pvDesc.rowNum, embedV, embedRoundV);
#ifdef __DAV_C220_CUBE__
        auto gPVSrc = pvDesc.isAppendBlock ? gVNew : gV;
        LayoutV layoutVTemp(MAX_KV_STACK_LEN, strideV);
        LayoutP layoutPTemp(pvDesc.rowNum, pvDesc.stackSeqTile, stackSeqTilePad);
        uint64_t gmOffsetP =
            coreIdx * WORKSPACE_BLOCK_SIZE_DB * STACK_SLOTS + pvDesc.slot * WORKSPACE_BLOCK_SIZE_DB;
        if constexpr (PAGED_CACHE_FLAG) {
            // Delayed PV derives its page cursor from its own stack, not the current QK task.
            blockMmadPV.resetBlockStart(pvDesc.kvSIdx, pagedBlockSize);
            blockMmadPV(gP[gmOffsetP], gPVSrc[pvDesc.gmOffsetVSrc], gOTmp[gmOffsetOTmp],
                        pvDesc.isAppendBlock ? gBlockTable : gBlockTable[pvDesc.blockBOffset], layoutPTemp,
                        layoutVTemp, layoutOTmp, actualBlockShapePV, pvDesc.kvSIdx, pvDesc.kvSLoopNumTotal,
                        pagedBlockSize, pvDesc.noSkipKvS, strideV, blockStackNum, softmaxReady,
                        pvDesc.doCopyback, gV[pvDesc.gmOffsetV], pvDesc.pvCacheOffset, gBlockTable,
                        pvDesc.isAppendBlock ? pagedBlockSize : 0U, static_cast<uint32_t>(pvDesc.blockBOffset));
        } else {
            blockMmadPV(gP[gmOffsetP], gPVSrc[pvDesc.gmOffsetVSrc], gOTmp[gmOffsetOTmp], gBlockTable,
                        layoutPTemp, layoutVTemp, layoutOTmp, actualBlockShapePV, pvDesc.kvSIdx,
                        pvDesc.kvSLoopNumTotal, pagedBlockSize, pvDesc.noSkipKvS, strideV, blockStackNum,
                        softmaxReady, pvDesc.doCopyback, gV[pvDesc.gmOffsetV], pvDesc.pvCacheOffset);
        }
        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
#endif
#ifdef __DAV_C220_VEC__
        LayoutO layoutO(pvDesc.qSeqlen, embed * qHeads);
        LayoutUpdate layoutUpdate(pvDesc.rowNum, embed, embedRound);
        // Head-major LSE: stride(0) is maxQSeqlen for BSND and totalQTokens for TND.
        LayoutLse layoutLse(qHeads, (INPUT_LAYOUT == FaiKenel::inputLayout::TND) ? totalQTokens : maxQSeqlen);
        uint64_t gmOffsetUpdate = static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB);

        if constexpr (FLASH_DECODE_FLAG) {
            LayoutLse layoutgmLse(pvDesc.qSBlockSize, pvDesc.qNBlockSize);
            LayoutLse layoutgmLo(pvDesc.qSBlockSize, embed * pvDesc.qNBlockSize);
            typename EpilogueRescaleO::SplitKVParams splitParams;
            splitParams.isSplitkv = pvDesc.isSplitKV;
            splitParams.gCombineLse = gLseFD[pvDesc.gmOffsetLseFD];
            splitParams.gCombineo = gOFD[pvDesc.gmOffsetOFD];
            splitParams.layoutgmLse = &layoutgmLse;
            splitParams.layoutgmLo = &layoutgmLo;

            epilogueRescaleO(gO[pvDesc.gmOffsetO], gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate],
                             gLse[pvDesc.gmOffsetLse], layoutO, layoutOTmp, layoutUpdate, layoutLse,
                             actualBlockShapePV, pvReady, pvDesc.qSBlockSize, pvDesc.qNBlockSize,
                             pvDesc.firstStack, pvDesc.lastStack, pvDesc.slot, pvDesc.taskStateSlot,
                             splitParams);
        } else {
            epilogueRescaleO(
                gO[pvDesc.gmOffsetO], gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate], gLse[pvDesc.gmOffsetLse],
                layoutO, layoutOTmp, layoutUpdate, layoutLse, actualBlockShapePV, pvReady, pvDesc.qSBlockSize,
                pvDesc.qNBlockSize, pvDesc.firstStack, pvDesc.lastStack, pvDesc.slot, pvDesc.taskStateSlot,
                typename EpilogueRescaleO::SplitKVParams(), pvDesc.invalidSuffixStartRow,
                pvDesc.invalidPrefixEndRow, pvDesc.qSeqlen, pvDesc.qSBlockIdx, pvDesc.curQNBlockTile);
        }
#endif
    }

    // Submit exactly one QK/softmax stack. The descriptor is owned by the
    // caller so all fields written here are preserved in the pipeline ring.
    // SWA state is passed back explicitly.
    __aicore__ inline void RunQKSoftmaxStackImpl(
        uint32_t coreIdx, uint32_t qSeqlen, uint32_t kvSeqlen, uint32_t qNBlockSize,
        uint32_t qNBlockTile, uint32_t qSBlockIdx, uint32_t taskFlags, bool isSplitKV,
        bool appendKVFlag, uint32_t kvSeqlenOld, uint32_t kvStartOld,
        uint32_t kvLoopNumTotalNew, uint32_t kvNewSeqlen, uint64_t gmOffsetQ,
        uint64_t gmOffsetK, uint64_t gmOffsetV, uint64_t gmOffsetKNew,
        uint64_t gmOffsetVNew, uint64_t gmOffsetO, uint64_t gmOffsetLse,
        uint64_t gmOffsetLseFD, uint64_t gmOffsetOFD, uint64_t blockBOffset,
        uint32_t qSBlockSize, uint32_t rowNum, uint32_t noSkipKvS,
        uint32_t kvSLoopNumTotal, uint32_t kvSLoopNumTotalOld, uint32_t kvStart,
        uint32_t kvEnd, int32_t windowSizeLeftStartLen, int32_t windowSizeLeftEndLen,
        int32_t windowSizeRightStartLen, int32_t windowSizeRightEndLen, bool notPreMask,
        bool notNextMask, int32_t invalidSuffixStartRow, int32_t invalidPrefixEndRow,
        uint32_t currentTaskStateSlot, uint32_t kvSIdx, int32_t stackSeqCount,
        uint32_t taskStackCount, uint32_t issuedStackCount, uint32_t& softmaxPingPongFlag,
        AscendC::GlobalTensor<ElementQ> gQ, AscendC::GlobalTensor<ElementK> gK,
        AscendC::GlobalTensor<ElementMask> gMask, AscendC::GlobalTensor<int32_t> gBlockTable,
        AscendC::GlobalTensor<ElementS> gS, AscendC::GlobalTensor<ElementP> gP,
        AscendC::GlobalTensor<ElementK> gKNew, AscendC::GlobalTensor<ElementK> gVNew,
        StackDescriptor& desc, bool& startsWithMaskTile, bool& startsWithMaskThenNomaskFlag)
    {
        StackRunContext ctx{
            coreIdx, qSeqlen, kvSeqlen, qNBlockSize, qNBlockTile, qSBlockIdx, taskFlags,
            isSplitKV, appendKVFlag, kvSeqlenOld, kvStartOld, kvLoopNumTotalNew, kvNewSeqlen,
            gmOffsetQ, gmOffsetK, gmOffsetV, gmOffsetKNew, gmOffsetVNew, gmOffsetO, gmOffsetLse,
            gmOffsetLseFD, gmOffsetOFD, blockBOffset, qSBlockSize, noSkipKvS,
            kvSLoopNumTotal, kvSLoopNumTotalOld, kvStart, kvEnd, windowSizeLeftStartLen,
            windowSizeLeftEndLen, windowSizeRightStartLen, windowSizeRightEndLen, notPreMask,
            notNextMask, invalidSuffixStartRow, invalidPrefixEndRow, currentTaskStateSlot};

        const bool isAppendBlock = ctx.appendKVFlag && (kvSIdx >= ctx.kvStartOld);
        const bool doCopyback = isAppendBlock &&
            ((ctx.taskFlags & fa_metadata::TASK_FLAG_NEED_COPYBACK) != 0U);
        const uint32_t kvSIdxLocal = isAppendBlock ? kvSIdx - ctx.kvStartOld : kvSIdx;
        auto gQKSrc = isAppendBlock ? gKNew : gK;
        const uint64_t qkSrcOffset = isAppendBlock ? ctx.gmOffsetKNew : ctx.gmOffsetK;
        const uint64_t qkCacheOffset =
            doCopyback ? static_cast<uint64_t>(ctx.kvSeqlenOld) + kvSIdxLocal * MAX_KV_STACK_LEN : 0;
        const uint32_t curStackTileMod = issuedStackCount % STACK_SLOTS;
        desc.stackSeqTile = AscendC::Std::min(
            MAX_KV_STACK_LEN, GetStackSeqTile(isAppendBlock, kvSIdx, kvSIdxLocal, ctx.kvLoopNumTotalNew,
                                              ctx.kvSLoopNumTotalOld, ctx.noSkipKvS, ctx.kvSeqlenOld,
                                              ctx.kvNewSeqlen));
        desc.gmOffsetV = ctx.gmOffsetV;
        desc.gmOffsetVSrc = isAppendBlock ? ctx.gmOffsetVNew : ctx.gmOffsetV;
        desc.pvCacheOffset = qkCacheOffset;
        desc.isAppendBlock = isAppendBlock;
        desc.doCopyback = doCopyback;
        desc.gmOffsetO = ctx.gmOffsetO;
        desc.gmOffsetLse = ctx.gmOffsetLse;
        desc.gmOffsetLseFD = ctx.gmOffsetLseFD;
        desc.gmOffsetOFD = ctx.gmOffsetOFD;
        desc.isSplitKV = ctx.isSplitKV;
        desc.blockBOffset = ctx.blockBOffset;
        desc.rowNum = rowNum;
        desc.qSeqlen = ctx.qSeqlen;
        desc.qSBlockSize = ctx.qSBlockSize;
        desc.qNBlockSize = ctx.qNBlockSize;
        desc.kvSIdx = kvSIdxLocal;
        desc.kvSLoopNumTotal = ctx.kvSLoopNumTotal;
        desc.noSkipKvS = ctx.noSkipKvS;
        desc.slot = curStackTileMod;
        desc.taskStateSlot = ctx.currentTaskStateSlot;
        desc.firstStack = taskStackCount == 0;
        desc.lastStack = kvSIdx + 1 >= ctx.kvEnd;
        desc.invalidSuffixStartRow = ctx.invalidSuffixStartRow;
        desc.invalidPrefixEndRow = ctx.invalidPrefixEndRow;
        desc.qSBlockIdx = ctx.qSBlockIdx;
        desc.curQNBlockTile = ctx.qNBlockTile;

        const uint32_t stackSeqTilePad = MAX_KV_STACK_LEN;
        const uint64_t gmOffsetS = static_cast<uint64_t>(ctx.coreIdx * WORKSPACE_BLOCK_SIZE_DB * STACK_SLOTS +
                                                         curStackTileMod * WORKSPACE_BLOCK_SIZE_DB);
        GemmCoord actualBlockShapeQK{rowNum, desc.stackSeqTile, embed};
        LayoutS layOutS(rowNum, desc.stackSeqTile, stackSeqTilePad);
#ifdef __DAV_C220_CUBE__
        LayoutQ layoutQTemp(rowNum, embed);
        LayoutK layoutKTemp(strideK, MAX_KV_STACK_LEN);
        if constexpr (PAGED_CACHE_FLAG) {
            blockMmadQK(gQ[ctx.gmOffsetQ], gQKSrc[qkSrcOffset], gS[gmOffsetS],
                        isAppendBlock ? gBlockTable : gBlockTable[ctx.blockBOffset], layoutQTemp, layoutKTemp,
                        layOutS, actualBlockShapeQK, kvSIdxLocal, ctx.kvSLoopNumTotal, pagedBlockSize, strideK,
                        doCopyback, gK[ctx.gmOffsetK], qkCacheOffset, gBlockTable,
                        isAppendBlock ? pagedBlockSize : 0U, static_cast<uint32_t>(ctx.blockBOffset));
        } else {
            blockMmadQK(gQ[ctx.gmOffsetQ], gQKSrc[qkSrcOffset], gS[gmOffsetS], gBlockTable, layoutQTemp,
                        layoutKTemp, layOutS, actualBlockShapeQK, kvSIdxLocal, ctx.kvSLoopNumTotal, pagedBlockSize,
                        strideK, doCopyback, gK[ctx.gmOffsetK], qkCacheOffset);
        }
        if (kvSIdx == ctx.kvEnd - 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        }
        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#endif
#ifdef __DAV_C220_VEC__
        LayoutP layOutP(rowNum, desc.stackSeqTile, stackSeqTilePad);
        LayoutMask layOutMask(COMP_TRIU_MASK_DIM_LEN, COMP_TRIU_MASK_DIM_LEN);
        const uint32_t kvSStartIdx =
            isAppendBlock ? ctx.kvSeqlenOld + kvSIdxLocal * MAX_KV_STACK_LEN : kvSIdx * MAX_KV_STACK_LEN;
        const uint32_t kvSEndIdx = kvSStartIdx + desc.stackSeqTile;
        if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_CAUSAL) {
            const int64_t triUp = static_cast<int64_t>(ctx.noSkipKvS) - static_cast<int64_t>(ctx.qSBlockSize);
            const uint32_t triDown = ctx.noSkipKvS;
            if (triUp < static_cast<int64_t>(kvSEndIdx - 1U)) {
                epilogueOnlineSoftmax(gP[gmOffsetS], gS[gmOffsetS], gMask, layOutP, layOutS, layOutMask,
                                      actualBlockShapeQK, desc.firstStack, ctx.qSBlockSize, ctx.qNBlockSize,
                                      curStackTileMod, desc.taskStateSlot, qkReady, softmaxPingPongFlag,
                                      triUp, triDown, kvSStartIdx, kvSEndIdx, ctx.isSplitKV);
            } else {
                uint32_t lastNoMaskTile;
                if (triUp + 1U == ctx.kvSeqlen) {
                    lastNoMaskTile = ctx.kvSLoopNumTotal - 1U;
                } else if (triUp + 1U == ctx.kvSeqlenOld) {
                    lastNoMaskTile = ctx.kvStartOld - 1U;
                } else if (triUp + 1U < ctx.kvSeqlenOld) {
                    lastNoMaskTile = (triUp + 1U) / MAX_KV_STACK_LEN - 1U;
                } else {
                    lastNoMaskTile = ctx.kvStartOld + (triUp + 1U - ctx.kvSeqlenOld) / MAX_KV_STACK_LEN - 1U;
                }
                const int32_t lastNoMaskStackId =
                    lastNoMaskTile - (FLASH_DECODE_FLAG ? static_cast<int32_t>(ctx.kvStart) : 0);
                epilogueOnlineSoftmax(gP[gmOffsetS], gS[gmOffsetS], layOutP, layOutS, actualBlockShapeQK,
                                      desc.firstStack, (stackSeqCount == lastNoMaskStackId), ctx.qSBlockSize,
                                      ctx.qNBlockSize, curStackTileMod, desc.taskStateSlot, qkReady,
                                      softmaxPingPongFlag, ctx.isSplitKV, false, false);
            }
        } else if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_SWA) {
            const int32_t kvStartI = static_cast<int32_t>(kvSStartIdx);
            const int32_t kvEndI = static_cast<int32_t>(kvSEndIdx);
            const bool doTriUPreMask =
                (maskType != 2 || ctx.notPreMask) ? false
                : (ctx.windowSizeLeftStartLen >= kvStartI && ctx.windowSizeLeftStartLen < kvEndI) ||
                      (ctx.windowSizeLeftEndLen > kvStartI && ctx.windowSizeLeftEndLen <= kvEndI) ||
                      (ctx.windowSizeLeftStartLen <= kvStartI && ctx.windowSizeLeftEndLen >= kvEndI);
            const bool doTriUNextMask =
                (maskType != 2 || ctx.notNextMask) ? false
                : (ctx.windowSizeRightStartLen >= kvStartI && ctx.windowSizeRightStartLen < kvEndI) ||
                      (ctx.windowSizeRightEndLen > kvStartI && ctx.windowSizeRightEndLen <= kvEndI) ||
                      (ctx.windowSizeRightStartLen <= kvStartI && ctx.windowSizeRightEndLen >= kvEndI);
            if (doTriUPreMask || doTriUNextMask) {
                startsWithMaskTile = true;
                startsWithMaskThenNomaskFlag = true;
                epilogueOnlineSoftmax(gP[gmOffsetS], gS[gmOffsetS], gMask, layOutP, layOutS, layOutMask,
                                      actualBlockShapeQK, desc.firstStack, ctx.qSBlockSize, ctx.qNBlockSize,
                                      curStackTileMod, desc.taskStateSlot, qkReady, softmaxPingPongFlag,
                                      kvSStartIdx, doTriUPreMask, doTriUNextMask, ctx.windowSizeLeftStartLen,
                                      ctx.windowSizeLeftEndLen, ctx.windowSizeRightStartLen,
                                      ctx.windowSizeRightEndLen, ctx.isSplitKV);
            } else {
                const bool isLastNoMaskStackTile =
                    (ctx.windowSizeRightStartLen >= ctx.kvSeqlen) || (ctx.windowSizeRightStartLen < 0);
                const uint32_t kvSeqlenLimit =
                    isLastNoMaskStackTile ? ctx.kvSeqlen : ctx.windowSizeRightStartLen;
                const uint32_t alignedKvSeqlenLimit = isLastNoMaskStackTile
                    ? RoundUp(kvSeqlenLimit, MAX_KV_STACK_LEN)
                    : RoundDown(kvSeqlenLimit, MAX_KV_STACK_LEN);
                const uint32_t noMaskStackSeqNum =
                    (alignedKvSeqlenLimit - ctx.kvStart * MAX_KV_STACK_LEN) / MAX_KV_STACK_LEN;
                epilogueOnlineSoftmax(gP[gmOffsetS], gS[gmOffsetS], layOutP, layOutS, actualBlockShapeQK,
                                      desc.firstStack, (stackSeqCount == noMaskStackSeqNum - 1), ctx.qSBlockSize,
                                      ctx.qNBlockSize, curStackTileMod, desc.taskStateSlot, qkReady,
                                      softmaxPingPongFlag, ctx.isSplitKV, startsWithMaskTile,
                                      startsWithMaskThenNomaskFlag);
                startsWithMaskTile = false;
            }
        } else {
            epilogueOnlineSoftmax(gP[gmOffsetS], gS[gmOffsetS], layOutP, layOutS, actualBlockShapeQK,
                                  desc.firstStack, 0, ctx.qSBlockSize, ctx.qNBlockSize, curStackTileMod,
                                  desc.taskStateSlot, qkReady, softmaxPingPongFlag, ctx.isSplitKV, false, false);
        }
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
    }

    __aicore__ inline void RunQKSoftmaxStack(
        StackRunContext& ctx, uint32_t kvSIdx, int32_t stackSeqCount,
        uint32_t taskStackCount, uint32_t issuedStackCount,
        uint32_t& softmaxPingPongFlag, GlobalTensorBundle& globalTensors,
        StackDescriptor& desc, bool& startsWithMaskTile, bool& startsWithMaskThenNomaskFlag)
    {
        RunQKSoftmaxStackImpl(
            ctx.coreIdx, ctx.qSeqlen, ctx.kvSeqlen, ctx.qNBlockSize, ctx.qNBlockTile, ctx.qSBlockIdx,
            ctx.taskFlags, ctx.isSplitKV, ctx.appendKVFlag, ctx.kvSeqlenOld, ctx.kvStartOld,
            ctx.kvLoopNumTotalNew, ctx.kvNewSeqlen, ctx.gmOffsetQ, ctx.gmOffsetK, ctx.gmOffsetV,
            ctx.gmOffsetKNew, ctx.gmOffsetVNew, ctx.gmOffsetO, ctx.gmOffsetLse, ctx.gmOffsetLseFD,
            ctx.gmOffsetOFD, ctx.blockBOffset, ctx.qSBlockSize, ctx.qSBlockSize * ctx.qNBlockSize, ctx.noSkipKvS,
            ctx.kvSLoopNumTotal, ctx.kvSLoopNumTotalOld, ctx.kvStart, ctx.kvEnd,
            ctx.windowSizeLeftStartLen, ctx.windowSizeLeftEndLen, ctx.windowSizeRightStartLen,
            ctx.windowSizeRightEndLen, ctx.notPreMask, ctx.notNextMask, ctx.invalidSuffixStartRow,
            ctx.invalidPrefixEndRow, ctx.currentTaskStateSlot, kvSIdx, stackSeqCount, taskStackCount,
            issuedStackCount, softmaxPingPongFlag, globalTensors.gQ, globalTensors.gK,
            globalTensors.gMask, globalTensors.gBlockTable, globalTensors.gS, globalTensors.gP,
            globalTensors.gKNew, globalTensors.gVNew, desc, startsWithMaskTile,
            startsWithMaskThenNomaskFlag);
    }

    __aicore__ inline uint32_t RunPrefillStacks(
        StackRunContext& ctx, uint32_t& issuedStackCount, uint32_t& prefillRemaining,
        uint32_t& softmaxPingPongFlag, GlobalTensorBundle& globalTensors,
        StackDescriptor (&descriptors)[STACK_SLOTS], bool& startsWithMaskTile,
        bool& startsWithMaskThenNomaskFlag)
    {
        uint32_t taskPrefillCount = 0;
        while (prefillRemaining != 0U && ctx.kvStart + taskPrefillCount < ctx.kvEnd) {
            StackDescriptor& desc = descriptors[issuedStackCount % STACK_SLOTS];
            RunQKSoftmaxStack(ctx, ctx.kvStart + taskPrefillCount, taskPrefillCount,
                              taskPrefillCount, issuedStackCount, softmaxPingPongFlag,
                              globalTensors, desc, startsWithMaskTile, startsWithMaskThenNomaskFlag);
            ++taskPrefillCount;
            ++issuedStackCount;
            --prefillRemaining;
        }
        return taskPrefillCount;
    }

    __aicore__ inline bool runMainLoop(uint32_t coreIdx, uint32_t qSeqlen, uint32_t kvSeqlenCur,
                                       uint32_t qNStartIdx, uint32_t qNBlockSize, uint32_t qNBlockTile,
                                       uint32_t kvNIdx, uint32_t qSBlockIdx, uint32_t taskFlags,
                                       bool isSplitKV, int32_t stS2IdxNow,
                                       int32_t enS2IdxNow, uint64_t gmOffsetLseFD, uint64_t gmOffsetOFD,
                                       const __gm__ MaskPlan* maskPlan, const BatchRuntime& batchRuntime,
                                       GlobalTensorBundle& globalTensors,
                                       StackDescriptor (&descriptors)[STACK_SLOTS],
                                       const uint32_t& currentTaskStateSlot, uint32_t& issuedStackCount,
                                       uint32_t& prefillRemaining,
                                       uint32_t& softmaxPingPongFlag, uint32_t blockStackNum)
    {
        // Per batch KV length from the caller, already includes the appended new KV
        uint32_t kvSeqlen = kvSeqlenCur;
        const bool appendKVFlag = batchRuntime.appendKVFlag != 0U;
        uint32_t kvSeqlenOld = batchRuntime.kvSeqlenOld;
        const uint64_t qBOffset = batchRuntime.qBOffset;
        const uint64_t kBOffset = batchRuntime.kBOffset;
        const uint64_t vBOffset = batchRuntime.vBOffset;
        const uint64_t oBOffset = batchRuntime.oBOffset;
        const uint64_t lseBOffset = batchRuntime.lseBOffset;
        const uint64_t blockBOffset = batchRuntime.blockBOffset;
        const uint64_t kNewBOffset = batchRuntime.kNewBOffset;
        const uint64_t vNewBOffset = batchRuntime.vNewBOffset;
        const uint32_t lseHeadStride = batchRuntime.lseHeadStride;
        const uint32_t curQSBlockTile = batchRuntime.curQSBlockTile;
        const uint32_t curQSBlockNum = batchRuntime.curQSBlockNum;
        const uint32_t curKSBlockNum = batchRuntime.curKSBlockNum;
        const uint32_t kvStartOld = batchRuntime.kvStartOld;
        const uint32_t kvLoopNumTotalNew = batchRuntime.kvLoopNumTotalNew;
        const uint32_t kvNewSeqlen = batchRuntime.kvNewSeqlen;

        uint32_t lseTokenOffset = qSBlockIdx * curQSBlockTile;

        uint64_t gmOffsetQ = qBOffset + static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideQ +
                             static_cast<uint64_t>(qNStartIdx) * qHeadStride;
        uint64_t gmOffsetK = kBOffset + static_cast<uint64_t>(kvNIdx) * kvHeadStrideK;
        uint64_t gmOffsetV = vBOffset + static_cast<uint64_t>(kvNIdx) * kvHeadStrideV;
        // newkv are (b, s_new, h_k, d); batch b starts at b * kvNewSeqlen tokens.
        uint64_t gmOffsetKNew = appendKVFlag ? kNewBOffset + static_cast<uint64_t>(kvNIdx * embed) : 0;
        uint64_t gmOffsetVNew = appendKVFlag ? vNewBOffset + static_cast<uint64_t>(kvNIdx * embedV) : 0;
        uint64_t gmOffsetO = oBOffset + static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideO +
                             static_cast<uint64_t>(qNStartIdx) * oHeadStride;
        // head-major: base + head*headStride + qToken
        uint64_t gmOffsetLse =
            lseBOffset + static_cast<uint64_t>(qNStartIdx) * lseHeadStride + static_cast<uint64_t>(lseTokenOffset);

        uint32_t qSBlockSize =
            (qSBlockIdx == (curQSBlockNum - 1U)) ? (qSeqlen - qSBlockIdx * curQSBlockTile) : curQSBlockTile;
        uint32_t rowNum = qSBlockSize * qNBlockSize;

        const uint32_t noSkipKvS = maskPlan->noSkipKvS;
        uint32_t kvSLoopNumTotal = maskPlan->kvSLoopNumTotal;
        uint32_t kvStart = maskPlan->kvStart;
        const int32_t windowSizeLeftStartLen = maskPlan->windowSizeLeftStartLen;
        const int32_t windowSizeLeftEndLen = maskPlan->windowSizeLeftEndLen;
        const int32_t windowSizeRightStartLen = maskPlan->windowSizeRightStartLen;
        const int32_t windowSizeRightEndLen = maskPlan->windowSizeRightEndLen;
        const bool notPreMask = (maskPlan->flags & fa_metadata::MASK_PLAN_NOT_PRE_MASK) != 0;
        const bool notNextMask = (maskPlan->flags & fa_metadata::MASK_PLAN_NOT_NEXT_MASK) != 0;
        const int32_t invalidSuffixStartRow = maskPlan->invalidSuffixStartRow;
        const int32_t invalidPrefixEndRow = maskPlan->invalidPrefixEndRow;

        // Match FA GPU split-KV: n_block_min/max = split ∩ window (do not overwrite SWA).
        uint32_t winKvStart = kvStart;
        uint32_t winKvEnd = kvSLoopNumTotal;
        uint32_t kvEnd = winKvEnd;
        // Append-KV phase split: [0, kvStartOld) reads the cache,
        // [kvStartOld, kvEnd) reads newkv and writes it back to the cache
        // (writeback must run even for fully-masked tiles).
        uint32_t kvSLoopNumTotalOld = kvSLoopNumTotal;
        if (appendKVFlag) {
            const uint32_t oldKvForMask = AscendC::Std::min(noSkipKvS, kvSeqlenOld);
            kvSLoopNumTotalOld = CeilDiv(oldKvForMask, MAX_KV_STACK_LEN);
            kvSLoopNumTotal = kvSLoopNumTotalOld + kvLoopNumTotalNew;
            kvEnd = kvSLoopNumTotal;
            // winKvEnd must be the per-phase total (v3 captures it after
            // the append update) so the FD fdEnd fallback covers old+new.
            winKvEnd = kvSLoopNumTotal;
        }

        // FD cap runs AFTER the append update (v3 order): the coreInfo S2
        // slice is final and must not be overridden by append's kvEnd.
        if constexpr (FLASH_DECODE_FLAG) {
            uint32_t fdStart = static_cast<uint32_t>(stS2IdxNow);
            uint32_t fdEnd =
                (enS2IdxNow == static_cast<int32_t>(curKSBlockNum)) ? winKvEnd : static_cast<uint32_t>(enS2IdxNow);
            kvStart = AscendC::Std::max(fdStart, winKvStart);
            kvEnd = AscendC::Std::min(fdEnd, winKvEnd);
        }

        const bool isEmptyTask = kvStart >= kvEnd;
        if (__builtin_expect(isEmptyTask, false)) {
#ifdef __DAV_C220_VEC__
            if (!isSplitKV) {
                auto& gO = globalTensors.gO;
                auto& gLse = globalTensors.gLse;
                LayoutO layoutOInit(qSeqlen, embed * qHeads);
                LayoutLse layoutLseInit(qHeads, lseHeadStride);
                EpilogueInitOut epilogueInitOut(resource);
                epilogueInitOut(gO[gmOffsetO], gLse[gmOffsetLse], layoutOInit, layoutLseInit, qSBlockSize, qNBlockSize,
                                static_cast<uint32_t>(strideO), static_cast<uint32_t>(oHeadStride));
            }
#endif
            return false;
        }

#ifdef __DAV_C220_CUBE__
        LayoutQ layoutQTemp(rowNum, embed);
        blockMmadQK.resetBlockStart(kvStart, pagedBlockSize);
        blockMmadQK.loadQGM(globalTensors.gQ[gmOffsetQ], layoutQTemp, rowNum, qNBlockSize,
                            static_cast<uint32_t>(qHeadStride), static_cast<uint32_t>(strideQ));
#endif
        bool startsWithMaskTile = false;
        bool startsWithMaskThenNomaskFlag = false;
        StackRunContext stackCtx{
            coreIdx, qSeqlen, kvSeqlen, qNBlockSize, qNBlockTile, qSBlockIdx, taskFlags,
            isSplitKV, appendKVFlag, kvSeqlenOld, kvStartOld, kvLoopNumTotalNew, kvNewSeqlen,
            gmOffsetQ, gmOffsetK, gmOffsetV, gmOffsetKNew, gmOffsetVNew, gmOffsetO, gmOffsetLse,
            gmOffsetLseFD, gmOffsetOFD, blockBOffset, qSBlockSize, noSkipKvS,
            kvSLoopNumTotal, kvSLoopNumTotalOld, kvStart, kvEnd, windowSizeLeftStartLen,
            windowSizeLeftEndLen, windowSizeRightStartLen, windowSizeRightEndLen, notPreMask, notNextMask,
            invalidSuffixStartRow, invalidPrefixEndRow, currentTaskStateSlot,
            };

        const uint32_t taskPrefillCount = RunPrefillStacks(
            stackCtx, issuedStackCount, prefillRemaining, softmaxPingPongFlag, globalTensors, descriptors,
            startsWithMaskTile, startsWithMaskThenNomaskFlag);

        int32_t steadyStackSeqCount = static_cast<int32_t>(taskPrefillCount);
        uint32_t steadyTaskStackCount = taskPrefillCount;
        for (uint32_t kvSIdx = kvStart + taskPrefillCount; kvSIdx < kvEnd; ++kvSIdx) {
            StackDescriptor& desc = descriptors[issuedStackCount % STACK_SLOTS];
            RunQKSoftmaxStack(stackCtx, kvSIdx, steadyStackSeqCount, steadyTaskStackCount,
                              issuedStackCount, softmaxPingPongFlag, globalTensors, desc,
                              startsWithMaskTile, startsWithMaskThenNomaskFlag);
            StackDescriptor& pvDesc = descriptors[(issuedStackCount - PRE_LAUNCH) % STACK_SLOTS];
            ConsumeDelayedPV(coreIdx, pvDesc, blockStackNum, MAX_KV_STACK_LEN, globalTensors);
            ++steadyStackSeqCount;
            ++steadyTaskStackCount;
            ++issuedStackCount;
        }
        return !isEmptyTask;
    }

  private:
    uint64_t mm1OutSize;
    uint64_t smOnlineOutSize;
    uint64_t mm2OutSize;
    uint32_t batch;
    uint32_t qHeads;
    uint32_t kvHeads;
    uint32_t embed;
    uint32_t embedV;
    uint32_t pagedBlockSize;
    uint32_t maxNumBlocksPerBatch;
    uint32_t blockSize;
    uint32_t maskType;
    int64_t windowSizeLeft;
    int64_t windowSizeRight;
    float scaleValue;
    float softcapValue;
    uint32_t totalQTokens;
    uint32_t maxQSeqlen;
    uint32_t maxKvSeqlen;
    uint32_t kvStackLen = MAX_KV_STACK_LEN;
    // Append-KV: per-batch new length (0 = disabled) and cache capacity.
    uint32_t kvNewSeqlen;
    uint32_t kvCacheSeqlen;
    // BNSD only: allocated KV sequence length (KV head-plane pitch in tokens).
    int64_t kvSeqlenAllocated;

    // Row (token) strides and per-head plane pitches. BSND/TND use the packed
    // forms (head stride = embed(V), token stride = H*embed(V)); BNSD swaps them
    // (token stride = embed(V), head stride = S_alloc*embed(V)).
    uint64_t strideQ;
    uint64_t strideO;
    uint64_t strideK;
    uint64_t strideV;
    uint64_t qHeadStride;
    uint64_t oHeadStride;
    uint64_t kvHeadStrideK;
    uint64_t kvHeadStrideV;
    uint32_t embedRound;
    uint32_t embedRoundV;
    uint32_t groupSize;

    Arch::Resource<ArchTag> resource;
    Arch::CrossCoreFlag qkReady{QK_READY_ID};
    Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
    Arch::CrossCoreFlag pvReady{PV_READY_ID};
    Gemm::Block::BlockPingPongState pingPongState;

    BlockMmadQK blockMmadQK;
    BlockMmadPV blockMmadPV;
    EpilogueOnlineSoftmax epilogueOnlineSoftmax;
    EpilogueRescaleO epilogueRescaleO;
};
} // namespace SplitFuse

namespace SplitFuse {
template <typename InputDtypeQ = half, typename InputDtypeKv = half, typename IntermCalcPrec = float,
          bool PagedCacheFlag = false, bool FlashDecodeFlag = false,
          FaiKenel::MaskType maskCategory = FaiKenel::MaskType::NO_MASK,
          FaiKenel::inputLayout inLayout = FaiKenel::inputLayout::TND,
          Epilogue::LseModeT lseMode = Epilogue::LseModeT::NONE, bool HAS_SOFTCAP = false>
__global__ __aicore__ void
FAInfer(uint64_t fftsAddr, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables, GM_ADDR o, GM_ADDR lse,
        GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR workspace, FAInferTilingParams tiling, GM_ADDR kNew,
        GM_ADDR vNew, GM_ADDR seqUsedQ = nullptr, GM_ADDR seqUsedKv = nullptr, int64_t kvSeqlenAllocated = 0)
{
    AscendC::SetSyncBaseAddr(fftsAddr);

    using ArchTag = Arch::AtlasA2;
    using ElementQ = InputDtypeQ;
    using LayoutQ = layout::RowMajor;
    using ElementK = InputDtypeKv;
    using LayoutK = layout::ColumnMajor;
    using ElementV = InputDtypeKv;
    using LayoutV = layout::RowMajor;
    using ElementS = IntermCalcPrec;
    using LayoutS = layout::RowMajor;
    using ElementP = InputDtypeQ;
    using LayoutP = layout::RowMajor;
    using ElementO = InputDtypeQ;
    using LayoutO = layout::RowMajor;
    using ElementLse = float;
    using LayoutLse = layout::RowMajor;
    using ElementMask = uint8_t;
    using LayoutMask = layout::RowMajor;
    using ElementOTmp = IntermCalcPrec;
    using LayoutOTmp = layout::RowMajor;
    using ElementUpdate = IntermCalcPrec;
    using LayoutUpdate = layout::RowMajor;

    using L1TileShapeQK = GemmShape<Q_TILE_CEIL, 128, 128>;
    using L0TileShapeQK = GemmShape<128, 128, 128>;
    using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQKT<PagedCacheFlag, false>;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, QType, KType, SType>;

    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmaxT<lseMode, IntermCalcPrec, HAS_SOFTCAP>;
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using maskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, maskType>;

    using L1TileShapePV = GemmShape<128, 128, 256>;
    using L0TileShapePV = GemmShape<128, 128, 128>;
    using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPVT<PagedCacheFlag, false>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, PType, VType, OTmpType>;

    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleOT<lseMode, IntermCalcPrec>;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using LseType = Gemm::GemmType<ElementLse, LayoutLse>;
    using EpilogueRescaleO =
        Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType, LseType>;

    using SplitInfoType = splitNode[25];
    using CombineScale = Epilogue::Block::CombineScale<OType, LseType, SplitInfoType>;
    using FAInferKernelType = FAInferKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO,
                                            PagedCacheFlag, FlashDecodeFlag, maskCategory, inLayout, CombineScale>;

    FAIKernelParams params{q,   k,         v,      mask, blockTables, actualQseqlen, actualKvseqlen, o,
                           lse, workspace, tiling, kNew, vNew,        seqUsedQ,      seqUsedKv,      kvSeqlenAllocated};
    FAInferKernelType flashAttnInfer;
    flashAttnInfer(params);
}
} // namespace SplitFuse

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

    // A task's PV/RS inputs remain live across subsequent tasks. Save them
    // once per nonempty task, using the same ring as its softmax state.
    struct DelayedPVTaskContext {
        uint64_t gmOffsetV;
        uint64_t gmOffsetVNew;
        uint64_t gmOffsetO;
        uint64_t gmOffsetLse;
        // Delayed rescale must retain the producing task's split destination.
        uint64_t gmOffsetLseFD;
        uint64_t gmOffsetOFD;
        uint64_t blockBOffset;
        uint32_t rowNum;
        uint32_t qSeqlen;
        uint32_t qSBlockSize;
        uint32_t qNBlockSize;
        uint32_t kvSLoopNumTotal;
        uint32_t noSkipKvS;
        int32_t invalidSuffixStartRow;
        int32_t invalidPrefixEndRow;
        int32_t qSBlockIdx;
        int32_t curQNBlockTile;
        bool isSplitKV;
        typename EpilogueRescaleO::TaskPlan rescaleTaskPlan;
    };

    struct StackDescriptor {
        uint64_t pvCacheOffset;
        uint32_t stackSeqTile;
        uint32_t kvSIdx;
        uint32_t slot;
        uint32_t taskStateSlot;
        bool isAppendBlock;
        bool doCopyback;
        bool firstStack;
        bool lastStack;
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
#ifdef __DAV_C220_VEC__
        uint32_t taskEnd;
        uint32_t taskInnerCount;
        uint32_t taskStepInner;
        uint32_t taskStepOuter;
        uint32_t copybackQSBlock;
#endif
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
        const auto* schedulerHeader = reinterpret_cast<const __gm__ SchedulerHeader*>(
            reinterpret_cast<const __gm__ uint8_t*>(fATilingData) + fa_metadata::SCHEDULER_HEADER_OFFSET);
        // Both AIC and AIV reject stale metadata before arming pipeline events.
        if (schedulerHeader->version != fa_metadata::TASK_METADATA_VERSION ||
            schedulerHeader->status != fa_metadata::TASK_METADATA_OK) {
            return;
        }
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
        if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_SWA) {
            gMask.SetGlobalBuffer((__gm__ ElementMask*)params.mask);
        }
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
        if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_CAUSAL ||
                      MASK_TYPE == FaiKenel::MaskType::MASK_SWA) {
            epilogueOnlineSoftmax.InitArithMask(resource);
        }
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
        DelayedPVTaskContext taskContexts[STACK_SLOTS];
        uint32_t issuedStackCount = 0;
        uint32_t taskStateSequence = 0;

#ifdef __DAV_C220_VEC__
        const auto* taskRanges = reinterpret_cast<const __gm__ CoreTaskRange*>(
            reinterpret_cast<const __gm__ uint8_t*>(fATilingData) + fa_metadata::CORE_RANGE_OFFSET);
        const auto* batchPlans = reinterpret_cast<const __gm__ BatchPlan*>(
            reinterpret_cast<const __gm__ uint8_t*>(fATilingData) + schedulerHeader->batchPlanOffset);
        const auto* headPlans = reinterpret_cast<const __gm__ HeadPlan*>(
            reinterpret_cast<const __gm__ uint8_t*>(fATilingData) + schedulerHeader->headPlanOffset);
        const __gm__ CausalMaskPlan* causalMaskPlans = nullptr;
        const __gm__ SwaMaskPlan* swaMaskPlans = nullptr;
        if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_CAUSAL) {
            causalMaskPlans = reinterpret_cast<const __gm__ CausalMaskPlan*>(
                reinterpret_cast<const __gm__ uint8_t*>(fATilingData) + schedulerHeader->maskPlanOffset);
        } else if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_SWA) {
            swaMaskPlans = reinterpret_cast<const __gm__ SwaMaskPlan*>(
                reinterpret_cast<const __gm__ uint8_t*>(fATilingData) + schedulerHeader->maskPlanOffset);
        }
        const auto* appendPlans = reinterpret_cast<const __gm__ AppendPlan*>(
            reinterpret_cast<const __gm__ uint8_t*>(fATilingData) + schedulerHeader->appendPlanOffset);
        const auto* addressPlans = reinterpret_cast<const __gm__ AddressPlan*>(
            reinterpret_cast<const __gm__ uint8_t*>(fATilingData) + schedulerHeader->addressPlanOffset);
        const bool idleCoreFD =
            FLASH_DECODE_FLAG && (fATilingData->needCoreNum != 0U) && (coreIdx >= fATilingData->needCoreNum);
        if (coreIdx < schedulerHeader->coreNum && !idleCoreFD) {
            const __gm__ CoreTaskRange* range = &taskRanges[coreIdx];
            const uint32_t taskCount = range->taskCount;
            const uint32_t blockStackNum = (MAX_KV_STACK_LEN - 1 + pagedBlockSize) / pagedBlockSize;
            if (taskCount != 0U) {
                uint32_t batchIdx = range->firstBatch;
                uint32_t inner = range->firstInner;
                uint32_t outer = range->firstOuter;
                uint32_t ordinal = range->taskStart;
                const uint32_t taskStride = FLASH_DECODE_FLAG ? 1U : schedulerHeader->coreNum;
                const __gm__ BatchPlan* batchPlan = &batchPlans[batchIdx];
                BatchRuntime runtime{};
                const __gm__ AppendPlan* appendPlan = &appendPlans[batchIdx];
                const __gm__ AddressPlan* addressPlan = &addressPlans[batchIdx];
                runtime.qBOffset = addressPlan->qBOffset;
                runtime.kBOffset = addressPlan->kBOffset;
                runtime.vBOffset = addressPlan->vBOffset;
                if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::BNSD) {
                    runtime.kBOffset = static_cast<uint64_t>(batchIdx) * kvHeads * kvHeadStrideK;
                    runtime.vBOffset = static_cast<uint64_t>(batchIdx) * kvHeads * kvHeadStrideV;
                }
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
                runtime.taskEnd = batchPlan->taskBase + batchPlan->qNBlockNum * batchPlan->qSBlockNum;
                runtime.taskInnerCount = FLASH_DECODE_FLAG ? batchPlan->qSBlockNum : batchPlan->qNBlockNum;
                runtime.taskStepInner = batchPlan->taskStepInner;
                runtime.taskStepOuter = batchPlan->taskStepOuter;
                runtime.copybackQSBlock = batchPlan->copybackQSBlock;
                uint32_t cachedMaskPlanIndex = UINT32_MAX;
                MaskPlan maskValue{};
                for (uint32_t local = 0; local < taskCount; ++local, ordinal += taskStride) {
                    if (ordinal >= runtime.taskEnd) {
                        do {
                            batchPlan = &batchPlans[++batchIdx];
                        } while (ordinal >= batchPlan->taskBase + batchPlan->qNBlockNum * batchPlan->qSBlockNum);
                        appendPlan = &appendPlans[batchIdx];
                        addressPlan = &addressPlans[batchIdx];
                        runtime.qBOffset = addressPlan->qBOffset;
                        runtime.kBOffset = addressPlan->kBOffset;
                        runtime.vBOffset = addressPlan->vBOffset;
                        if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::BNSD) {
                            runtime.kBOffset = static_cast<uint64_t>(batchIdx) * kvHeads * kvHeadStrideK;
                            runtime.vBOffset = static_cast<uint64_t>(batchIdx) * kvHeads * kvHeadStrideV;
                        }
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
                        runtime.taskEnd = batchPlan->taskBase + batchPlan->qNBlockNum * batchPlan->qSBlockNum;
                        runtime.taskInnerCount = FLASH_DECODE_FLAG ? batchPlan->qSBlockNum : batchPlan->qNBlockNum;
                        runtime.taskStepInner = batchPlan->taskStepInner;
                        runtime.taskStepOuter = batchPlan->taskStepOuter;
                        runtime.copybackQSBlock = batchPlan->copybackQSBlock;
                        // Decode only at batch transitions; in-batch tasks use a carry.
                        const uint32_t inBatch = ordinal - batchPlan->taskBase;
                        outer = inBatch / runtime.taskInnerCount;
                        inner = inBatch - outer * runtime.taskInnerCount;
                    }
                    const uint32_t qS = FLASH_DECODE_FLAG ? inner : outer;
                    const uint32_t qN = FLASH_DECODE_FLAG ? outer : inner;
                    const __gm__ HeadPlan* head = &headPlans[batchPlan->headPlanBase + qN];
                    if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_CAUSAL ||
                                  MASK_TYPE == FaiKenel::MaskType::MASK_SWA) {
                        const uint32_t maskIndex = batchPlan->maskPlanBase + qS;
                        if (maskIndex != cachedMaskPlanIndex) {
                            cachedMaskPlanIndex = maskIndex;
                            if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_CAUSAL) {
                                const __gm__ CausalMaskPlan* plan = &causalMaskPlans[maskIndex];
                                maskValue.causal.invalidPrefixEndRow = plan->invalidPrefixEndRow;
                                maskValue.causal.noSkipKvS = plan->noSkipKvS;
                                maskValue.causal.kvSLoopNumTotal = plan->kvSLoopNumTotal;
                            } else {
                                const __gm__ SwaMaskPlan* plan = &swaMaskPlans[maskIndex];
                                maskValue.swa.windowSizeLeftStartLen = plan->windowSizeLeftStartLen;
                                maskValue.swa.windowSizeRightStartLen = plan->windowSizeRightStartLen;
                                maskValue.swa.invalidSuffixStartRow = plan->invalidSuffixStartRow;
                                maskValue.swa.invalidPrefixEndRow = plan->invalidPrefixEndRow;
                                maskValue.swa.noSkipKvS = plan->noSkipKvS;
                                maskValue.swa.kvStart = plan->kvStart;
                                maskValue.swa.kvSLoopNumTotal = plan->kvSLoopNumTotal;
                                maskValue.swa.flags = plan->flags;
                            }
                        }
                    }
                    int32_t startS2 = 0;
                    int32_t endS2 = static_cast<int32_t>(runtime.curKSBlockNum);
                    bool split = false;
                    uint64_t lseOffset = 0;
                    uint64_t oOffset = 0;
                    if constexpr (FLASH_DECODE_FLAG) {
                        if (local == 0U || local + 1U == taskCount) {
                            const bool first = local == 0U;
                            startS2 = first ? range->firstS2 : 0;
                            endS2 = local + 1U == taskCount ? range->lastS2 : endS2;
                            split = (range->flags & (first ? fa_metadata::CORE_FIRST_TASK_SPLIT
                                                         : fa_metadata::CORE_LAST_TASK_SPLIT)) != 0U;
                            lseOffset = first ? range->firstLseOffset : range->lastLseOffset;
                            oOffset = first ? range->firstOOffset : range->lastOOffset;
                        }
                    }
                    const uint32_t taskFlags = qS == runtime.copybackQSBlock
                        ? fa_metadata::TASK_FLAG_NEED_COPYBACK : 0U;
                    const uint32_t stateSlot = taskStateSequence % STACK_SLOTS;
                    const bool hasTaskWork = runMainLoop(
                        coreIdx, batchPlan->qSeqlen, batchPlan->kvSeqlen,
                        head->qNStart, head->qNSize, batchPlan->qNBlockTile, head->kvHeadOffset,
                        qS, taskFlags, split, startS2, endS2, lseOffset, oOffset,
                        maskValue, runtime, globalTensors, descriptors, taskContexts,
                        stateSlot, issuedStackCount, softmaxPingPongFlag, blockStackNum,
                        local + 1U == taskCount ? PRE_LAUNCH : 0U);
                    taskStateSequence += static_cast<uint32_t>(hasTaskWork);
                    inner += runtime.taskStepInner;
                    outer += runtime.taskStepOuter;
                    if (inner >= runtime.taskInnerCount) {
                        inner -= runtime.taskInnerCount;
                        ++outer;
                    }
                }
            }
        }
#else
        if constexpr (FLASH_DECODE_FLAG) {
            if (fATilingData->needCoreNum == 0U || coreIdx < fATilingData->needCoreNum) {
                RunCubeTaskLoop(fATilingData, coreIdx, coreNum, globalTensors, descriptors, taskContexts, issuedStackCount,
                                taskStateSequence, softmaxPingPongFlag);
            }
        } else {
            RunCubeTaskLoop(fATilingData, coreIdx, coreNum, globalTensors, descriptors, taskContexts, issuedStackCount,
                            taskStateSequence, softmaxPingPongFlag);
        }
#endif

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
        if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_CAUSAL ||
                      MASK_TYPE == FaiKenel::MaskType::MASK_SWA) {
            epilogueOnlineSoftmax.ReleaseArithMaskEvents();
        }
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

#ifdef __DAV_C220_CUBE__
    __aicore__ inline void BuildCubeBatchRuntime(BatchRuntime& runtime, uint32_t batchIdx,
                                                 uint32_t qSeqlen, uint32_t kvSeqlen,
                                                 uint32_t qPrefix, uint32_t kvPrefix) const
    {
        if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::BNSD) {
            runtime.qBOffset = static_cast<uint64_t>(batchIdx) * qHeads * qHeadStride;
            runtime.oBOffset = static_cast<uint64_t>(batchIdx) * qHeads * oHeadStride;
        } else {
            runtime.qBOffset = static_cast<uint64_t>(qPrefix) * strideQ;
            runtime.oBOffset = static_cast<uint64_t>(qPrefix) * strideO;
        }
        runtime.lseBOffset = INPUT_LAYOUT == FaiKenel::inputLayout::TND
            ? qPrefix : static_cast<uint64_t>(qPrefix) * qHeads;
        runtime.lseHeadStride = INPUT_LAYOUT == FaiKenel::inputLayout::TND ? totalQTokens : maxQSeqlen;
        runtime.kBOffset = 0;
        runtime.vBOffset = 0;
        runtime.blockBOffset = 0;
        if constexpr (!PAGED_CACHE_FLAG) {
            if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::BNSD) {
                runtime.kBOffset = static_cast<uint64_t>(batchIdx) * kvHeads * kvHeadStrideK;
                runtime.vBOffset = static_cast<uint64_t>(batchIdx) * kvHeads * kvHeadStrideV;
            } else {
                runtime.kBOffset = static_cast<uint64_t>(kvPrefix) * strideK;
                runtime.vBOffset = static_cast<uint64_t>(kvPrefix) * strideV;
            }
        } else {
            runtime.blockBOffset = static_cast<uint64_t>(batchIdx) * maxNumBlocksPerBatch;
        }
        runtime.kNewBOffset = static_cast<uint64_t>(batchIdx) * kvNewSeqlen * strideK;
        runtime.vNewBOffset = static_cast<uint64_t>(batchIdx) * kvNewSeqlen * strideV;
        runtime.curQSBlockTile = GetQSBlockTile(kvSeqlen);
        runtime.curQSBlockNum = CeilDiv(qSeqlen, runtime.curQSBlockTile);
        runtime.appendKVFlag = kvNewSeqlen != 0U;
        runtime.kvSeqlenOld = runtime.appendKVFlag ? kvSeqlen - kvNewSeqlen : kvSeqlen;
        runtime.kvStartOld = runtime.appendKVFlag
            ? CeilDiv(runtime.kvSeqlenOld, MAX_KV_STACK_LEN) : 0U;
        runtime.kvLoopNumTotalNew = runtime.appendKVFlag
            ? CeilDiv(kvNewSeqlen, MAX_KV_STACK_LEN) : 0U;
        // Append-KV uses two logical phases: old cache stacks followed by
        // new-KV stacks.  This must match the tiler's old+new stack count;
        // ceil((old + new) / stackLen) would merge the phase boundary.
        runtime.curKSBlockNum = runtime.appendKVFlag
            ? runtime.kvStartOld + runtime.kvLoopNumTotalNew
            : CeilDiv(kvSeqlen, MAX_KV_STACK_LEN);
        runtime.kvNewSeqlen = kvNewSeqlen;
    }

#ifdef __DAV_C220_CUBE__
    // CUBE only needs mask-derived KV bounds and row invalidation metadata.
    // The mask tensor itself is consumed by VEC during softmax.  Keep this
    // calculation local to the CUBE task path and specialize it by mask type
    // so unused causal/SWA branches do not enter the CUBE instruction body.
    __aicore__ inline void BuildCubeMaskGeometry(
        uint32_t& kvStart, uint32_t& kvStackCount, uint32_t& noSkipKvS,
        int32_t& invalidPrefixEndRow, int32_t& invalidSuffixStartRow,
        uint32_t qSeqlen, uint32_t kvSeqlen, uint32_t qSBlockIdx,
        uint32_t qSBlockTile, uint32_t qSBlockSize) const
    {
        kvStart = 0;
        kvStackCount = 0;
        noSkipKvS = kvSeqlen;
        invalidPrefixEndRow = static_cast<int32_t>(qSeqlen);
        invalidSuffixStartRow = 0;
        if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_CAUSAL) {
            int64_t end = static_cast<int64_t>((qSBlockIdx + 1U) * qSBlockTile) +
                          static_cast<int64_t>(kvSeqlen) - qSeqlen;
            end = end < 0 ? 0 : end;
            noSkipKvS = static_cast<uint32_t>(
                end < static_cast<int64_t>(kvSeqlen) ? end : static_cast<int64_t>(kvSeqlen));
            kvStackCount = CeilDiv(noSkipKvS, MAX_KV_STACK_LEN);
            if (qSeqlen > kvSeqlen) {
                invalidPrefixEndRow = static_cast<int32_t>(qSeqlen - kvSeqlen);
            }
        } else if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_SWA) {
            int32_t leftPoint = static_cast<int32_t>(kvSeqlen);
            int32_t rightPoint = 0;
            int32_t leftStart = 0;
            int32_t leftEnd = 0;
            int32_t rightStart = 0;
            int32_t rightEnd = 0;
            if (windowSizeLeft < 0 && -windowSizeLeft >= static_cast<int64_t>(qSeqlen)) {
                kvStart = 0;
            } else if (windowSizeLeft != fa_metadata::MASK_PLAN_WINDOW_MAX) {
                leftPoint = static_cast<int32_t>(kvSeqlen) - static_cast<int32_t>(qSeqlen) -
                            static_cast<int32_t>(windowSizeLeft);
                leftStart = static_cast<int32_t>(qSBlockIdx * qSBlockTile) + leftPoint;
                leftEnd = leftStart + static_cast<int32_t>(qSBlockSize);
                kvStart = static_cast<uint32_t>(
                    (leftStart > 0 ? leftStart : 0) /
                    static_cast<int32_t>(MAX_KV_STACK_LEN));
            }
            if (windowSizeRight < 0 && -windowSizeRight >= static_cast<int64_t>(kvSeqlen)) {
                kvStackCount = 0;
            } else if (windowSizeRight != fa_metadata::MASK_PLAN_WINDOW_MAX) {
                rightPoint = static_cast<int32_t>(kvSeqlen) - static_cast<int32_t>(qSeqlen) +
                             static_cast<int32_t>(windowSizeRight);
                rightStart = static_cast<int32_t>(qSBlockIdx * qSBlockTile) + rightPoint;
                rightEnd = rightStart + static_cast<int32_t>(qSBlockSize);
                int32_t end = rightEnd <= 0
                    ? 0
                    : ((rightEnd + MAX_KV_STACK_LEN - 1) /
                       MAX_KV_STACK_LEN) * MAX_KV_STACK_LEN;
                end = end > static_cast<int32_t>(kvSeqlen) ? static_cast<int32_t>(kvSeqlen) : end;
                noSkipKvS = static_cast<uint32_t>(end <= 0 ? kvSeqlen : end);
                kvStackCount = CeilDiv(noSkipKvS, MAX_KV_STACK_LEN);
            } else {
                kvStackCount = CeilDiv(noSkipKvS, MAX_KV_STACK_LEN);
            }
            if (leftEnd > static_cast<int32_t>(kvSeqlen) &&
                windowSizeLeft != fa_metadata::MASK_PLAN_WINDOW_MAX) {
                invalidSuffixStartRow = static_cast<int32_t>(kvSeqlen) - leftPoint;
            } else if (rightStart < 0 && windowSizeRight != fa_metadata::MASK_PLAN_WINDOW_MAX) {
                invalidPrefixEndRow = -rightPoint;
            }
        } else {
            kvStackCount = CeilDiv(noSkipKvS, MAX_KV_STACK_LEN);
        }
    }
#endif

    __aicore__ inline void RunCubeTaskLoop(
        const __gm__ FAInferTilingData* tilingData, uint32_t coreIdx, uint32_t coreNum,
        GlobalTensorBundle& globalTensors,
        StackDescriptor (&descriptors)[STACK_SLOTS], DelayedPVTaskContext (&taskContexts)[STACK_SLOTS],
        uint32_t& issuedStackCount,
        uint32_t& taskStateSequence,
        uint32_t& softmaxPingPongFlag)
    {
        const uint32_t blockStackNum = (MAX_KV_STACK_LEN - 1 + pagedBlockSize) / pagedBlockSize;
        auto& gActualQseqlen = globalTensors.gActualQseqlen;
        auto& gActualKvseqlen = globalTensors.gActualKvseqlen;
        if constexpr (FLASH_DECODE_FLAG) {
            const __gm__ coreNode* node = &tilingData->coreInfo[coreIdx];
            uint64_t gmOffsetLseFD = static_cast<uint64_t>(node->firstSplitKVTaskLseOffset);
            uint64_t gmOffsetOFD = static_cast<uint64_t>(node->firstSplitKVTaskOOffset);
            for (uint32_t b = node->startBIdx; b <= static_cast<uint32_t>(node->endBIdx); ++b) {
                const uint32_t qSeqlen = INPUT_LAYOUT == FaiKenel::inputLayout::TND
                    ? QSeqLenOf(b, gActualQseqlen, globalTensors.gSeqUsedQ) : maxQSeqlen;
                uint32_t kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(b));
                if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                    if constexpr (!PAGED_CACHE_FLAG) {
                        kvSeqlen = KvSeqLenOf(b, gActualKvseqlen, globalTensors.gSeqUsedKv);
                    }
                }
                kvSeqlen += kvNewSeqlen;
                BatchRuntime runtime{};
                const uint32_t qPrefix = INPUT_LAYOUT == FaiKenel::inputLayout::TND
                    ? static_cast<uint32_t>(gActualQseqlen.GetValue(b)) : b * maxQSeqlen;
                const uint32_t kvPrefix = kvNewSeqlen != 0U ? b * kvCacheSeqlen
                    : (INPUT_LAYOUT == FaiKenel::inputLayout::TND
                        ? static_cast<uint32_t>(gActualKvseqlen.GetValue(b))
                        : static_cast<uint32_t>(gActualKvseqlen.GetValue(0)) * b);
                BuildCubeBatchRuntime(runtime, b, qSeqlen, kvSeqlen, qPrefix, kvPrefix);
                const uint32_t qNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
                const uint32_t qNBlockNumPerGroup = CeilDiv(groupSize, qNBlockTile);
                const uint32_t qNBlockNum = qNBlockNumPerGroup * kvHeads;
                const uint32_t qSBlockNum = runtime.curQSBlockNum;
                const int32_t nBegin = b == static_cast<uint32_t>(node->startBIdx) ? node->startN1Idx : 0;
                const int32_t nEnd = b == static_cast<uint32_t>(node->endBIdx)
                    ? node->endN1Idx : static_cast<int32_t>(qNBlockNum) - 1;
                for (int32_t n = nBegin; n <= nEnd; ++n) {
                    const int32_t sBegin = b == static_cast<uint32_t>(node->startBIdx) && n == nBegin
                        ? node->startS1Idx : 0;
                    const int32_t sEnd = b == static_cast<uint32_t>(node->endBIdx) && n == nEnd
                        ? node->endS1Idx : static_cast<int32_t>(qSBlockNum) - 1;
                    for (int32_t s = sBegin; s <= sEnd; ++s) {
                        const int32_t stS2 = b == static_cast<uint32_t>(node->startBIdx) && n == nBegin && s == sBegin
                            ? node->startS2Idx : 0;
                        const int32_t enS2 = b == static_cast<uint32_t>(node->endBIdx) && n == nEnd && s == sEnd
                            ? node->endS2Idx : static_cast<int32_t>(runtime.curKSBlockNum);
                        const bool split = enS2 > stS2 && enS2 - stS2 < static_cast<int32_t>(runtime.curKSBlockNum);
                        const uint32_t qSSize = s + 1 == static_cast<int32_t>(qSBlockNum)
                            ? qSeqlen - static_cast<uint32_t>(s) * runtime.curQSBlockTile : runtime.curQSBlockTile;
                        const uint32_t nInGroup = static_cast<uint32_t>(n) % qNBlockNumPerGroup;
                        const uint32_t qNSize = nInGroup + 1 == qNBlockNumPerGroup
                            ? groupSize - nInGroup * qNBlockTile : qNBlockTile;
                        const uint32_t taskFlags = fa_metadata::NeedsCopyback(
                            b, static_cast<uint32_t>(s), qSBlockNum)
                            ? fa_metadata::TASK_FLAG_NEED_COPYBACK : 0U;
                        const uint32_t currentTaskStateSlot = taskStateSequence % STACK_SLOTS;
                        const uint32_t nInGroupStart = nInGroup * qNBlockTile;
                        const uint32_t qNStart = (static_cast<uint32_t>(n) / qNBlockNumPerGroup) * groupSize + nInGroupStart;
                        const bool hasTaskWork = runMainLoop(
                            coreIdx, qSeqlen, kvSeqlen, qNStart, qNSize, qNBlockTile,
                            static_cast<uint32_t>(n) / qNBlockNumPerGroup, static_cast<uint32_t>(s), taskFlags,
                            split, stS2, enS2, gmOffsetLseFD, gmOffsetOFD, runtime,
                            globalTensors, descriptors, taskContexts, currentTaskStateSlot, issuedStackCount,
                            softmaxPingPongFlag, blockStackNum,
                            b == static_cast<uint32_t>(node->endBIdx) && n == nEnd && s == sEnd
                                ? PRE_LAUNCH : 0U);
                        taskStateSequence += static_cast<uint32_t>(hasTaskWork);
                        if (split) {
                            gmOffsetLseFD += static_cast<uint64_t>(qSSize) * qNSize;
                            gmOffsetOFD += static_cast<uint64_t>(qSSize) * qNSize * embedV;
                        }
                    }
                }
            }
        } else {
            uint32_t curBatch = 0;
            uint32_t previousTaskCount = 0;
            uint32_t currentTaskEnd = tilingData->firstBatchTaskNum;
            uint32_t cachedBatch = UINT32_MAX;
            uint32_t qSeqlen = INPUT_LAYOUT == FaiKenel::inputLayout::TND
                ? QSeqLenOf(0, gActualQseqlen, globalTensors.gSeqUsedQ) : maxQSeqlen;
            uint32_t kvSeqlen = 0;
            uint32_t qNBlockTile = 0;
            uint32_t qNBlockNum = 0;
            BatchRuntime runtime{};
            for (uint32_t taskIdx = coreIdx; taskIdx < tilingData->totalTaskNum; taskIdx += coreNum) {
                while (taskIdx >= currentTaskEnd) {
                    ++curBatch;
                    previousTaskCount = currentTaskEnd;
                    qSeqlen = INPUT_LAYOUT == FaiKenel::inputLayout::TND
                        ? QSeqLenOf(curBatch, gActualQseqlen, globalTensors.gSeqUsedQ) : maxQSeqlen;
                    const uint32_t nextQTile = GetQNBlockTile(qSeqlen, groupSize);
                    const uint32_t nextQN = CeilDiv(groupSize, nextQTile) * kvHeads;
                    currentTaskEnd += nextQN * CeilDiv(qSeqlen, Q_TILE_CEIL);
                }
                if (curBatch != cachedBatch) {
                    kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
                    if constexpr (INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                        if constexpr (!PAGED_CACHE_FLAG) {
                            kvSeqlen = KvSeqLenOf(curBatch, gActualKvseqlen, globalTensors.gSeqUsedKv);
                        }
                    }
                    kvSeqlen += kvNewSeqlen;
                    const uint32_t qPrefix = INPUT_LAYOUT == FaiKenel::inputLayout::TND
                        ? static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch)) : curBatch * maxQSeqlen;
                    const uint32_t kvPrefix = kvNewSeqlen != 0U ? curBatch * kvCacheSeqlen
                        : (INPUT_LAYOUT == FaiKenel::inputLayout::TND
                            ? static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch))
                            : static_cast<uint32_t>(gActualKvseqlen.GetValue(0)) * curBatch);
                    BuildCubeBatchRuntime(runtime, curBatch, qSeqlen, kvSeqlen, qPrefix, kvPrefix);
                    qNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
                    qNBlockNum = CeilDiv(groupSize, qNBlockTile) * kvHeads;
                    cachedBatch = curBatch;
                }
                const uint32_t taskInBatch = taskIdx - previousTaskCount;
                const uint32_t qSBlockIdx = taskInBatch / qNBlockNum;
                const uint32_t qNBlockIdx = taskInBatch % qNBlockNum;
                const uint32_t qSSize = qSBlockIdx + 1 == runtime.curQSBlockNum
                    ? qSeqlen - qSBlockIdx * runtime.curQSBlockTile : runtime.curQSBlockTile;
                const uint32_t nInGroup = qNBlockIdx % (qNBlockNum / kvHeads);
                const uint32_t qNSize = nInGroup + 1 == qNBlockNum / kvHeads
                    ? groupSize - nInGroup * qNBlockTile : qNBlockTile;
                const uint32_t taskFlags = fa_metadata::NeedsCopyback(
                    curBatch, qSBlockIdx, runtime.curQSBlockNum)
                    ? fa_metadata::TASK_FLAG_NEED_COPYBACK : 0U;
                const uint32_t currentTaskStateSlot = taskStateSequence % STACK_SLOTS;
                const bool hasTaskWork = runMainLoop(
                    coreIdx, qSeqlen, kvSeqlen,
                    (qNBlockIdx / (qNBlockNum / kvHeads)) * groupSize + nInGroup * qNBlockTile,
                    qNSize, qNBlockTile, qNBlockIdx / (qNBlockNum / kvHeads), qSBlockIdx,
                    taskFlags, false, 0, 0, 0, 0, runtime, globalTensors,
                    descriptors, taskContexts, currentTaskStateSlot, issuedStackCount,
                    softmaxPingPongFlag, blockStackNum,
                    tilingData->totalTaskNum - taskIdx <= coreNum ? PRE_LAUNCH : 0U);
                taskStateSequence += static_cast<uint32_t>(hasTaskWork);
            }
        }
    }
#endif

    __aicore__ inline bool runMainLoop(uint32_t coreIdx, uint32_t qSeqlen, uint32_t kvSeqlenCur,
                                       uint32_t qNStartIdx, uint32_t qNBlockSize, uint32_t qNBlockTile,
                                       uint32_t kvNIdx, uint32_t qSBlockIdx, uint32_t taskFlags,
                                       bool isSplitKV, int32_t stS2IdxNow,
                                       int32_t enS2IdxNow, uint64_t gmOffsetLseFD, uint64_t gmOffsetOFD,
#ifdef __DAV_C220_VEC__
                                       const MaskPlan& maskGeometry, const BatchRuntime& batchRuntime,
#else
                                       const BatchRuntime& batchRuntime,
#endif
                                       GlobalTensorBundle& globalTensors,
                                       StackDescriptor (&descriptors)[STACK_SLOTS],
                                       DelayedPVTaskContext (&taskContexts)[STACK_SLOTS],
                                       const uint32_t& currentTaskStateSlot, uint32_t& issuedStackCount,
                                       uint32_t& softmaxPingPongFlag, uint32_t blockStackNum,
                                       uint32_t pipelineDrain)
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

#ifdef __DAV_C220_CUBE__
        uint32_t kvStart = 0;
        uint32_t kvSLoopNumTotal = 0;
        uint32_t noSkipKvS = 0;
        int32_t invalidPrefixEndRow = 0;
        int32_t invalidSuffixStartRow = 0;
        BuildCubeMaskGeometry(kvStart, kvSLoopNumTotal, noSkipKvS, invalidPrefixEndRow,
                              invalidSuffixStartRow, qSeqlen, kvSeqlen, qSBlockIdx,
                              curQSBlockTile, qSBlockSize);
#else
        uint32_t noSkipKvS = kvSeqlen;
        uint32_t kvSLoopNumTotal = CeilDiv(kvSeqlen, MAX_KV_STACK_LEN);
        uint32_t kvStart = 0;
        int32_t windowSizeLeftStartLen = 0;
        int32_t windowSizeLeftEndLen = 0;
        int32_t windowSizeRightStartLen = 0;
        int32_t windowSizeRightEndLen = 0;
        bool notPreMask = true;
        bool notNextMask = true;
        int32_t invalidSuffixStartRow = 0;
        int32_t invalidPrefixEndRow = static_cast<int32_t>(qSeqlen);
        if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_CAUSAL) {
            noSkipKvS = maskGeometry.causal.noSkipKvS;
            kvSLoopNumTotal = maskGeometry.causal.kvSLoopNumTotal;
            invalidPrefixEndRow = maskGeometry.causal.invalidPrefixEndRow;
        } else if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_SWA) {
            noSkipKvS = maskGeometry.swa.noSkipKvS;
            kvSLoopNumTotal = maskGeometry.swa.kvSLoopNumTotal;
            kvStart = maskGeometry.swa.kvStart;
            windowSizeLeftStartLen = maskGeometry.swa.windowSizeLeftStartLen;
            windowSizeRightStartLen = maskGeometry.swa.windowSizeRightStartLen;
            notPreMask = (maskGeometry.swa.flags & fa_metadata::MASK_PLAN_NOT_PRE_MASK) != 0;
            notNextMask = (maskGeometry.swa.flags & fa_metadata::MASK_PLAN_NOT_NEXT_MASK) != 0;
            windowSizeLeftEndLen = notPreMask ? 0 : windowSizeLeftStartLen + static_cast<int32_t>(qSBlockSize);
            windowSizeRightEndLen = notNextMask ? 0 : windowSizeRightStartLen + static_cast<int32_t>(qSBlockSize);
            invalidSuffixStartRow = maskGeometry.swa.invalidSuffixStartRow;
            invalidPrefixEndRow = maskGeometry.swa.invalidPrefixEndRow;
        }
#endif

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
            if (pipelineDrain == 0U) {
                return false;
            }
            // A final empty task still drains earlier descriptors and must not
            // overwrite their persistent task contexts.
            kvStart = kvEnd;
        }

#ifdef __DAV_C220_CUBE__
        if (!isEmptyTask) {
            LayoutQ layoutQTemp(rowNum, embed);
            blockMmadQK.resetBlockStart(kvStart, pagedBlockSize);
            blockMmadQK.loadQGM(globalTensors.gQ[gmOffsetQ], layoutQTemp, rowNum, qNBlockSize,
                                static_cast<uint32_t>(qHeadStride), static_cast<uint32_t>(strideQ));
        }
#endif
        typename EpilogueOnlineSoftmax::TaskPlan softmaxTaskPlan{};
        if (!isEmptyTask) {
            // Reuse a context slot only after its previous delayed PV has been
            // consumed. Empty tasks leave all live slots intact.
#ifdef __DAV_C220_VEC__
            softmaxTaskPlan = EpilogueOnlineSoftmax::MakeTaskPlan(rowNum, qSBlockSize, qNBlockSize);
            const auto rescaleTaskPlan = EpilogueRescaleO::MakeTaskPlan(rowNum, qSBlockSize, qNBlockSize, embedRound);
#else
            const typename EpilogueRescaleO::TaskPlan rescaleTaskPlan{};
#endif
            taskContexts[currentTaskStateSlot] = DelayedPVTaskContext{
                gmOffsetV, gmOffsetVNew, gmOffsetO, gmOffsetLse, gmOffsetLseFD, gmOffsetOFD,
                blockBOffset, rowNum, qSeqlen, qSBlockSize, qNBlockSize, kvSLoopNumTotal, noSkipKvS,
                invalidSuffixStartRow, invalidPrefixEndRow, static_cast<int32_t>(qSBlockIdx),
                static_cast<int32_t>(qNBlockTile), isSplitKV, rescaleTaskPlan};
        }

        // One producer/consumer loop covers prefill, steady work and final
        // drain. Dummy tail iterations submit no QK and touch only old slots.
        const uint32_t stackSeqTilePad = MAX_KV_STACK_LEN;
        for (uint32_t kvSIdx = kvStart; kvSIdx < kvEnd + pipelineDrain; ++kvSIdx) {
            if (kvSIdx < kvEnd) {
                StackDescriptor& desc = descriptors[issuedStackCount % STACK_SLOTS];
                auto& gQ = globalTensors.gQ;
                auto& gK = globalTensors.gK;
                auto& gBlockTable = globalTensors.gBlockTable;
                auto& gS = globalTensors.gS;
                auto& gP = globalTensors.gP;
                auto& gKNew = globalTensors.gKNew;

                const bool isAppendBlock = appendKVFlag && (kvSIdx >= kvStartOld);
                const bool doCopyback = isAppendBlock &&
                    ((taskFlags & fa_metadata::TASK_FLAG_NEED_COPYBACK) != 0U);
                const uint32_t kvSIdxLocal = isAppendBlock ? kvSIdx - kvStartOld : kvSIdx;
                auto gQKSrc = isAppendBlock ? gKNew : gK;
                const uint64_t qkSrcOffset = isAppendBlock ? gmOffsetKNew : gmOffsetK;
                const uint64_t qkCacheOffset =
                    doCopyback ? static_cast<uint64_t>(kvSeqlenOld) + kvSIdxLocal * MAX_KV_STACK_LEN : 0;
                const uint32_t curStackTileMod = issuedStackCount % STACK_SLOTS;
                desc.stackSeqTile = AscendC::Std::min(
                    MAX_KV_STACK_LEN, GetStackSeqTile(isAppendBlock, kvSIdx, kvSIdxLocal, kvLoopNumTotalNew,
                                                      kvSLoopNumTotalOld, noSkipKvS, kvSeqlenOld,
                                                      kvNewSeqlen));
                desc.pvCacheOffset = qkCacheOffset;
                desc.isAppendBlock = isAppendBlock;
                desc.doCopyback = doCopyback;
                desc.kvSIdx = kvSIdxLocal;
                desc.slot = curStackTileMod;
                desc.taskStateSlot = currentTaskStateSlot;
                desc.firstStack = kvSIdx == kvStart;
                desc.lastStack = kvSIdx + 1 >= kvEnd;

                const uint64_t gmOffsetS = static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * STACK_SLOTS +
                                                                 curStackTileMod * WORKSPACE_BLOCK_SIZE_DB);
                GemmCoord actualBlockShapeQK{rowNum, desc.stackSeqTile, embed};
                LayoutS layOutS(rowNum, desc.stackSeqTile, stackSeqTilePad);
#ifdef __DAV_C220_CUBE__
                LayoutQ layoutQTemp(rowNum, embed);
                LayoutK layoutKTemp(strideK, MAX_KV_STACK_LEN);
                if constexpr (PAGED_CACHE_FLAG) {
                    blockMmadQK(gQ[gmOffsetQ], gQKSrc[qkSrcOffset], gS[gmOffsetS],
                                isAppendBlock ? gBlockTable : gBlockTable[blockBOffset], layoutQTemp, layoutKTemp,
                                layOutS, actualBlockShapeQK, kvSIdxLocal, kvSLoopNumTotal, pagedBlockSize, strideK,
                                doCopyback, gK[gmOffsetK], qkCacheOffset, gBlockTable,
                                isAppendBlock ? pagedBlockSize : 0U, static_cast<uint32_t>(blockBOffset));
                } else {
                    blockMmadQK(gQ[gmOffsetQ], gQKSrc[qkSrcOffset], gS[gmOffsetS], gBlockTable, layoutQTemp,
                                layoutKTemp, layOutS, actualBlockShapeQK, kvSIdxLocal, kvSLoopNumTotal, pagedBlockSize,
                                strideK, doCopyback, gK[gmOffsetK], qkCacheOffset);
                }
                if (kvSIdx == kvEnd - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
                }
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#endif
#ifdef __DAV_C220_VEC__
                LayoutP layOutP(rowNum, desc.stackSeqTile, stackSeqTilePad);
                const uint32_t kvSStartIdx =
                    isAppendBlock ? kvSeqlenOld + kvSIdxLocal * MAX_KV_STACK_LEN : kvSIdx * MAX_KV_STACK_LEN;
                const uint32_t kvSEndIdx = kvSStartIdx + desc.stackSeqTile;
                if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_CAUSAL) {
                    const int64_t triUp = static_cast<int64_t>(noSkipKvS) - static_cast<int64_t>(qSBlockSize);
                    const bool doTriUMask = triUp < static_cast<int64_t>(kvSEndIdx - 1U);
                    const int64_t diagS0 = triUp - static_cast<int64_t>(kvSStartIdx);
                    // Both causal tile modes use the same row pipeline, including unmasked tiles.
                    epilogueOnlineSoftmax(gP[gmOffsetS], gS[gmOffsetS], layOutP, layOutS,
                                          actualBlockShapeQK, softmaxTaskPlan, desc.firstStack,
                                          qSBlockSize, qNBlockSize,
                                          curStackTileMod, desc.taskStateSlot, qkReady, softmaxPingPongFlag,
                                          diagS0, doTriUMask, isSplitKV);
                } else if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_SWA) {
                    const int32_t kvStartI = static_cast<int32_t>(kvSStartIdx);
                    const int32_t kvEndI = static_cast<int32_t>(kvSEndIdx);
                    const bool doTriUPreMask =
                        (maskType != 2 || notPreMask) ? false
                        : (windowSizeLeftStartLen >= kvStartI && windowSizeLeftStartLen < kvEndI) ||
                              (windowSizeLeftEndLen > kvStartI && windowSizeLeftEndLen <= kvEndI) ||
                              (windowSizeLeftStartLen <= kvStartI && windowSizeLeftEndLen >= kvEndI);
                    const bool doTriUNextMask =
                        (maskType != 2 || notNextMask) ? false
                        : (windowSizeRightStartLen >= kvStartI && windowSizeRightStartLen < kvEndI) ||
                              (windowSizeRightEndLen > kvStartI && windowSizeRightEndLen <= kvEndI) ||
                              (windowSizeRightStartLen <= kvStartI && windowSizeRightEndLen >= kvEndI);
                    const bool doTriUMask = doTriUPreMask || doTriUNextMask;
                    const int64_t leftS0 = static_cast<int64_t>(windowSizeLeftStartLen) - kvSStartIdx;
                    const int64_t rightS0 = static_cast<int64_t>(windowSizeRightStartLen) - kvSStartIdx;
                    epilogueOnlineSoftmax(gP[gmOffsetS], gS[gmOffsetS], layOutP, layOutS,
                                          actualBlockShapeQK, softmaxTaskPlan, desc.firstStack,
                                          qSBlockSize, qNBlockSize,
                                          curStackTileMod, desc.taskStateSlot, qkReady, softmaxPingPongFlag,
                                          rightS0, doTriUMask, isSplitKV,
                                          leftS0, doTriUPreMask, doTriUNextMask);
                } else {
                    epilogueOnlineSoftmax(gP[gmOffsetS], gS[gmOffsetS], layOutP, layOutS, actualBlockShapeQK,
                                          softmaxTaskPlan,
                                          desc.firstStack, 0, qSBlockSize, qNBlockSize, curStackTileMod,
                                          desc.taskStateSlot, qkReady, softmaxPingPongFlag, isSplitKV, false, false);
                }
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
            }
            if (issuedStackCount >= PRE_LAUNCH) {
                StackDescriptor& pvDesc = descriptors[(issuedStackCount - PRE_LAUNCH) % STACK_SLOTS];
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
                const DelayedPVTaskContext& task = taskContexts[pvDesc.taskStateSlot];

                uint64_t gmOffsetOTmp = static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * STACK_SLOTS +
                                                               pvDesc.slot * WORKSPACE_BLOCK_SIZE_DB);
                GemmCoord actualBlockShapePV{task.rowNum, embedV, pvDesc.stackSeqTile};
                LayoutOTmp layoutOTmp(task.rowNum, embedV, embedRoundV);
#ifdef __DAV_C220_CUBE__
                auto gPVSrc = pvDesc.isAppendBlock ? gVNew : gV;
                const uint64_t gmOffsetVSrc = pvDesc.isAppendBlock ? task.gmOffsetVNew : task.gmOffsetV;
                // BlockMmad takes a mutable reference; keep each consumption local
                // rather than exposing the shared task snapshot to modification.
                uint32_t kvSLoopNumTotal = task.kvSLoopNumTotal;
                LayoutV layoutVTemp(MAX_KV_STACK_LEN, strideV);
                LayoutP layoutPTemp(task.rowNum, pvDesc.stackSeqTile, stackSeqTilePad);
                uint64_t gmOffsetP =
                    coreIdx * WORKSPACE_BLOCK_SIZE_DB * STACK_SLOTS + pvDesc.slot * WORKSPACE_BLOCK_SIZE_DB;
                if constexpr (PAGED_CACHE_FLAG) {
                    // Delayed PV derives its page cursor from its own stack, not the current QK task.
                    blockMmadPV.resetBlockStart(pvDesc.kvSIdx, pagedBlockSize);
                    blockMmadPV(gP[gmOffsetP], gPVSrc[gmOffsetVSrc], gOTmp[gmOffsetOTmp],
                                pvDesc.isAppendBlock ? gBlockTable : gBlockTable[task.blockBOffset], layoutPTemp,
                                layoutVTemp, layoutOTmp, actualBlockShapePV, pvDesc.kvSIdx, kvSLoopNumTotal,
                                pagedBlockSize, task.noSkipKvS, strideV, blockStackNum, softmaxReady,
                                pvDesc.doCopyback, gV[task.gmOffsetV], pvDesc.pvCacheOffset, gBlockTable,
                                pvDesc.isAppendBlock ? pagedBlockSize : 0U, static_cast<uint32_t>(task.blockBOffset));
                } else {
                    blockMmadPV(gP[gmOffsetP], gPVSrc[gmOffsetVSrc], gOTmp[gmOffsetOTmp], gBlockTable,
                                layoutPTemp, layoutVTemp, layoutOTmp, actualBlockShapePV, pvDesc.kvSIdx,
                                kvSLoopNumTotal, pagedBlockSize, task.noSkipKvS, strideV, blockStackNum,
                                softmaxReady, pvDesc.doCopyback, gV[task.gmOffsetV], pvDesc.pvCacheOffset);
                }
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
#endif
#ifdef __DAV_C220_VEC__
                LayoutO layoutO(task.qSeqlen, embed * qHeads);
                LayoutUpdate layoutUpdate(task.rowNum, embed, embedRound);
                // Head-major LSE: stride(0) is maxQSeqlen for BSND and totalQTokens for TND.
                LayoutLse layoutLse(qHeads, (INPUT_LAYOUT == FaiKenel::inputLayout::TND) ? totalQTokens : maxQSeqlen);
                uint64_t gmOffsetUpdate = static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB);

                if constexpr (FLASH_DECODE_FLAG) {
                    LayoutLse layoutgmLse(task.qSBlockSize, task.qNBlockSize);
                    LayoutLse layoutgmLo(task.qSBlockSize, embed * task.qNBlockSize);
                    typename EpilogueRescaleO::SplitKVParams splitParams;
                    splitParams.isSplitkv = task.isSplitKV;
                    splitParams.gCombineLse = gLseFD[task.gmOffsetLseFD];
                    splitParams.gCombineo = gOFD[task.gmOffsetOFD];
                    splitParams.layoutgmLse = &layoutgmLse;
                    splitParams.layoutgmLo = &layoutgmLo;

                    epilogueRescaleO(gO[task.gmOffsetO], gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate],
                                     gLse[task.gmOffsetLse], layoutO, layoutOTmp, layoutUpdate, layoutLse,
                                     actualBlockShapePV, task.rescaleTaskPlan, pvReady, task.qSBlockSize, task.qNBlockSize,
                                     pvDesc.firstStack, pvDesc.lastStack, pvDesc.slot, pvDesc.taskStateSlot,
                                     splitParams, 0, 0, 0, 0, 1,
                                     static_cast<uint32_t>(strideO), static_cast<uint32_t>(oHeadStride));
                } else {
                    epilogueRescaleO(
                        gO[task.gmOffsetO], gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate], gLse[task.gmOffsetLse],
                        layoutO, layoutOTmp, layoutUpdate, layoutLse, actualBlockShapePV, task.rescaleTaskPlan, pvReady,
                        task.qSBlockSize,
                        task.qNBlockSize, pvDesc.firstStack, pvDesc.lastStack, pvDesc.slot, pvDesc.taskStateSlot,
                        typename EpilogueRescaleO::SplitKVParams(), task.invalidSuffixStartRow,
                        task.invalidPrefixEndRow, task.qSeqlen, task.qSBlockIdx, task.curQNBlockTile,
                        static_cast<uint32_t>(strideO), static_cast<uint32_t>(oHeadStride));
                }
#endif
            }
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

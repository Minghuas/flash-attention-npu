/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026
 */

#ifndef FA_METADATA_ARGS_H
#define FA_METADATA_ARGS_H

#include <cstdint>
#include "tilingdata.h"

#if defined(__DAV_C220_CUBE__) || defined(__DAV_C220_VEC__)
#define FA_METADATA_INLINE __aicore__ inline
#else
#define FA_METADATA_INLINE inline
#endif

namespace fa_metadata {
constexpr uint32_t TASK_METADATA_VERSION = 3;
constexpr uint32_t TASK_METADATA_OK = 0;
constexpr uint32_t TASK_METADATA_OVERFLOW = 1;
constexpr uint32_t TASK_FLAG_NEED_COPYBACK = 1u << 0;
constexpr uint32_t CORE_FIRST_TASK_SPLIT = 1u << 0;
constexpr uint32_t CORE_LAST_TASK_SPLIT = 1u << 1;
constexpr uint32_t APPEND_PLAN_ENABLED = 1u << 0;
constexpr uint32_t MASK_PLAN_NOT_PRE_MASK = 1u << 0;
constexpr uint32_t MASK_PLAN_NOT_NEXT_MASK = 1u << 1;
constexpr uint64_t TASK_METADATA_ALIGN = 64;
constexpr uint32_t MASK_DIM = 2048;
constexpr uint64_t MASK_BYTES = static_cast<uint64_t>(MASK_DIM) * MASK_DIM;
constexpr uint32_t MASK_PLAN_STACK_LEN = 512;
constexpr int64_t MASK_PLAN_WINDOW_MAX = 2147483647;

// Append-KV copyback is selected per query-stack tile, for both ordinary and
// FlashDecode tasks.  Keep the rule in one place so SetOrdinaryTask()'s flag
// clearing cannot make the ordinary path diverge from the legacy logic.
static FA_METADATA_INLINE constexpr bool NeedsCopyback(uint32_t batchIdx, uint32_t qSBlockIdx, uint32_t qSBlockNum)
{
    return qSBlockNum != 0U && qSBlockIdx == batchIdx % qSBlockNum;
}

static FA_METADATA_INLINE uint32_t MaskPlanCeilDiv(uint32_t a, uint32_t b)
{
    return b == 0 ? 0 : (a + b - 1) / b;
}

static FA_METADATA_INLINE MaskPlan BuildMaskPlan(uint32_t qSeqlen, uint32_t kvSeqlen, uint32_t qSBlockIdx,
                                                 uint32_t qSBlockTile, uint32_t qSBlockSize, uint32_t maskType,
                                                 int64_t windowLeft, int64_t windowRight)
{
    MaskPlan p{};
    if (maskType == 1U) {
        p.causal.invalidPrefixEndRow = qSeqlen > kvSeqlen ? qSeqlen - kvSeqlen : qSeqlen;
        int64_t end = static_cast<int64_t>((qSBlockIdx + 1U) * qSBlockTile) +
                      static_cast<int64_t>(kvSeqlen) - qSeqlen;
        end = end < 0 ? 0 : end;
        p.causal.noSkipKvS = static_cast<uint32_t>(end < kvSeqlen ? end : kvSeqlen);
        p.causal.kvSLoopNumTotal = MaskPlanCeilDiv(p.causal.noSkipKvS, MASK_PLAN_STACK_LEN);
    } else if (maskType == 2U || maskType == 4U) {
        p.swa.noSkipKvS = kvSeqlen;
        p.swa.invalidPrefixEndRow = static_cast<int32_t>(qSeqlen);
        bool notPreMask = true;
        bool notNextMask = true;
        int32_t leftPoint = static_cast<int32_t>(kvSeqlen);
        int32_t rightPoint = 0;
        int32_t leftEnd = 0;
        if (!(windowLeft < 0 && -windowLeft >= static_cast<int64_t>(qSeqlen)) &&
            windowLeft != MASK_PLAN_WINDOW_MAX) {
            leftPoint = static_cast<int32_t>(kvSeqlen) - static_cast<int32_t>(qSeqlen) -
                        static_cast<int32_t>(windowLeft);
            p.swa.windowSizeLeftStartLen = qSBlockIdx * qSBlockTile + leftPoint;
            leftEnd = p.swa.windowSizeLeftStartLen + static_cast<int32_t>(qSBlockSize);
            p.swa.kvStart = static_cast<uint32_t>((p.swa.windowSizeLeftStartLen > 0 ?
                p.swa.windowSizeLeftStartLen : 0) / MASK_PLAN_STACK_LEN);
            notPreMask = false;
        }
        if (windowRight < 0 && -windowRight >= static_cast<int64_t>(kvSeqlen)) {
            p.swa.kvSLoopNumTotal = 0;
        } else if (windowRight != MASK_PLAN_WINDOW_MAX) {
            rightPoint = static_cast<int32_t>(kvSeqlen) - static_cast<int32_t>(qSeqlen) +
                         static_cast<int32_t>(windowRight);
            p.swa.windowSizeRightStartLen = qSBlockIdx * qSBlockTile + rightPoint;
            const int32_t rightEnd = p.swa.windowSizeRightStartLen + static_cast<int32_t>(qSBlockSize);
            int32_t end = rightEnd <= 0 ? 0 :
                ((rightEnd + MASK_PLAN_STACK_LEN - 1) / MASK_PLAN_STACK_LEN) * MASK_PLAN_STACK_LEN;
            end = end > static_cast<int32_t>(kvSeqlen) ? static_cast<int32_t>(kvSeqlen) : end;
            p.swa.noSkipKvS = static_cast<uint32_t>(end <= 0 ? kvSeqlen : end);
            p.swa.kvSLoopNumTotal = MaskPlanCeilDiv(p.swa.noSkipKvS, MASK_PLAN_STACK_LEN);
            notNextMask = false;
        } else {
            p.swa.kvSLoopNumTotal = MaskPlanCeilDiv(p.swa.noSkipKvS, MASK_PLAN_STACK_LEN);
        }
        if (leftEnd > static_cast<int32_t>(kvSeqlen) && windowLeft != MASK_PLAN_WINDOW_MAX) {
            p.swa.invalidSuffixStartRow = static_cast<int32_t>(kvSeqlen) - leftPoint;
        } else if (p.swa.windowSizeRightStartLen < 0 && windowRight != MASK_PLAN_WINDOW_MAX) {
            p.swa.invalidPrefixEndRow = -rightPoint;
        }
        p.swa.flags = (notPreMask ? MASK_PLAN_NOT_PRE_MASK : 0U) |
                      (notNextMask ? MASK_PLAN_NOT_NEXT_MASK : 0U);
    }
    return p;
}

constexpr uint64_t SCHEDULER_HEADER_OFFSET =
    (sizeof(FAInferTilingData) + TASK_METADATA_ALIGN - 1) / TASK_METADATA_ALIGN * TASK_METADATA_ALIGN;
constexpr uint64_t CORE_RANGE_OFFSET = SCHEDULER_HEADER_OFFSET + sizeof(SchedulerHeader);

// scheduler_metadata is tiling-only. The 2048x2048 compressed triu(1) mask is
// generated locally on the device by MakeDeviceTriuMask when required.
inline uint64_t TilingOffset(bool /*has_mask*/)
{
    return 0;
}
constexpr uint64_t AlignTaskMetadata(uint64_t value)
{
    return (value + TASK_METADATA_ALIGN - 1) / TASK_METADATA_ALIGN * TASK_METADATA_ALIGN;
}

constexpr uint64_t SchedulerHeaderOffset()
{
    return AlignTaskMetadata(sizeof(FAInferTilingData));
}

constexpr uint64_t CoreRangeOffset()
{
    return SchedulerHeaderOffset() + sizeof(SchedulerHeader);
}

constexpr uint64_t PlanDataOffset(uint32_t blockDim)
{
    return AlignTaskMetadata(CoreRangeOffset() + static_cast<uint64_t>(blockDim) * sizeof(CoreTaskRange));
}

constexpr uint64_t BatchPlanOffset(uint32_t blockDim, uint32_t /*batch*/, uint32_t /*maxQSeqlen*/,
                                   uint32_t /*numHeads*/, uint32_t /*numSplits*/ = 1U)
{
    return PlanDataOffset(blockDim);
}

constexpr uint32_t HeadPlanCapacity(uint32_t batch, uint32_t numHeads)
{
    return batch * numHeads;
}

constexpr uint64_t HeadPlanOffset(uint32_t blockDim, uint32_t batch, uint32_t maxQSeqlen,
                                  uint32_t numHeads, uint32_t numSplits = 1U)
{
    return AlignTaskMetadata(BatchPlanOffset(blockDim, batch, maxQSeqlen, numHeads, numSplits) +
                             static_cast<uint64_t>(batch) * sizeof(BatchPlan));
}

constexpr uint64_t HeadPlanEndOffset(uint32_t blockDim, uint32_t batch, uint32_t maxQSeqlen,
                                      uint32_t numHeads, uint32_t headPlanCapacity,
                                      uint32_t numSplits = 1U)
{
    return AlignTaskMetadata(HeadPlanOffset(blockDim, batch, maxQSeqlen, numHeads, numSplits) +
                             static_cast<uint64_t>(headPlanCapacity) * sizeof(HeadPlan));
}

constexpr uint32_t MaskPlanStride(uint32_t maskType)
{
    return maskType == 1U ? sizeof(CausalMaskPlan)
                        : ((maskType == 2U || maskType == 4U) ? sizeof(SwaMaskPlan) : 0U);
}

constexpr uint64_t MaskPlanCapacity(uint32_t batch, uint32_t maxQSeqlen, uint32_t maskType)
{
    if (maskType == 0U) return 0;
    const uint64_t qBlocks = (static_cast<uint64_t>(maxQSeqlen) + 127) / 128;
    return static_cast<uint64_t>(batch) * (qBlocks == 0 ? 1 : qBlocks);
}

constexpr uint64_t MaskPlanStorageBytes(uint32_t capacity, uint32_t maskType)
{
    return static_cast<uint64_t>(capacity) * MaskPlanStride(maskType);
}

constexpr uint64_t MaskPlanOffsetForCapacity(uint32_t blockDim, uint32_t batch, uint32_t maxQSeqlen,
                                             uint32_t numHeads, uint32_t headPlanCapacity,
                                             uint32_t fdPlanCapacity, uint32_t numSplits = 1U)
{
    (void)fdPlanCapacity;
    return HeadPlanEndOffset(blockDim, batch, maxQSeqlen, numHeads, headPlanCapacity, numSplits);
}

constexpr uint64_t AppendPlanOffsetForCapacity(uint32_t blockDim, uint32_t batch, uint32_t maxQSeqlen,
                                               uint32_t numHeads, uint32_t headPlanCapacity,
                                               uint32_t fdPlanCapacity, uint64_t maskPlanBytesCapacity,
                                               uint32_t numSplits = 1U)
{
    return AlignTaskMetadata(MaskPlanOffsetForCapacity(blockDim, batch, maxQSeqlen, numHeads,
                                                        headPlanCapacity, fdPlanCapacity, numSplits) +
                             maskPlanBytesCapacity);
}

constexpr uint64_t AddressPlanOffsetForCapacity(uint32_t blockDim, uint32_t batch, uint32_t maxQSeqlen,
                                                uint32_t numHeads, uint32_t headPlanCapacity,
                                                uint32_t fdPlanCapacity, uint64_t maskPlanBytesCapacity,
                                                uint32_t appendPlanCapacity, uint32_t numSplits = 1U)
{
    return AlignTaskMetadata(AppendPlanOffsetForCapacity(blockDim, batch, maxQSeqlen, numHeads,
                                                         headPlanCapacity, fdPlanCapacity, maskPlanBytesCapacity,
                                                         numSplits) +
                             static_cast<uint64_t>(appendPlanCapacity) * sizeof(AppendPlan));
}

inline uint64_t MetadataBytes(bool /*has_mask*/, uint32_t batch, uint32_t maxQSeqlen, uint32_t numHeads,
                              uint32_t blockDim, uint32_t headPlanCapacity,
                              uint32_t /*fdPlanCapacity*/ = 0, uint64_t maskPlanBytesCapacity = 0,
                              uint32_t appendPlanCapacity = 0, uint32_t addressPlanCapacity = 0,
                              uint32_t numSplits = 1U)
{
    return AddressPlanOffsetForCapacity(blockDim, batch, maxQSeqlen, numHeads, headPlanCapacity,
                                        0, maskPlanBytesCapacity, appendPlanCapacity, numSplits) +
           static_cast<uint64_t>(addressPlanCapacity) * sizeof(AddressPlan);
}

constexpr uint64_t WORKSPACE_BLOCK_SIZE_DB = static_cast<uint64_t>(128) * 512;
constexpr uint64_t PRELAUNCH_NUM = 3;
inline uint64_t Mm1OutSize(uint64_t blockDim)
{
    return blockDim * WORKSPACE_BLOCK_SIZE_DB * 4 * PRELAUNCH_NUM;
}
inline uint64_t SmOnlineOutSize(uint64_t blockDim)
{
    return blockDim * WORKSPACE_BLOCK_SIZE_DB * 2 * PRELAUNCH_NUM;
}
inline uint64_t Mm2OutSize(uint64_t blockDim)
{
    return blockDim * WORKSPACE_BLOCK_SIZE_DB * 4 * PRELAUNCH_NUM;
}
inline uint64_t UpdateOutSize(uint64_t blockDim)
{
    return blockDim * WORKSPACE_BLOCK_SIZE_DB * 4 * PRELAUNCH_NUM;
}
inline uint64_t WorkSpaceSize(uint64_t blockDim)
{
    return Mm1OutSize(blockDim) + SmOnlineOutSize(blockDim) + Mm2OutSize(blockDim) + UpdateOutSize(blockDim);
}
} // namespace fa_metadata

struct FAMetadataArgs {
    uint64_t seqlensQAddr;
    uint64_t cacheSeqlensAddr;
    uint64_t metaOutAddr;
    uint32_t batch;
    uint32_t numHeads;
    uint32_t numHeadsK;
    uint32_t embeddingSize;
    uint32_t embeddingSizeV;
    uint32_t numBlocks;
    uint32_t blockSize;
    uint32_t maxNumBlocksPerBatch;
    uint32_t maxQSeqlen;
    uint32_t maskType;
    uint32_t blockDim;
    uint32_t isVarlen;
    uint32_t isVarlenKv;
    uint32_t pagedKV;
    uint32_t numSplits;
    int64_t windowSizeLeft;
    int64_t windowSizeRight;
    // Append-KV tiling fields (0 = append disabled), mirrored by ComputeFAMetadata.
    uint32_t kvNewSeqlen;
    uint32_t kvCacheSeqlen;
    uint32_t taskCapacityFactor;
    // Version 3 reserves taskCapacityFactor/fdPlanCapacity as 1/0; neither
    // affects allocation. HeadPlan is bounded by batch*numHeads.
    uint32_t headPlanCapacity;
    uint32_t fdPlanCapacity;
    uint32_t maskPlanCapacity;
};

#undef FA_METADATA_INLINE

#endif

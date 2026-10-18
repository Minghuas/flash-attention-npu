/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef FLASH_ATTENTION_REGULAR_H
#define FLASH_ATTENTION_REGULAR_H

#include <cstdint>

struct coreNode {
    int startBIdx;
    int startN1Idx;
    int startS1Idx;
    int startS2Idx;
    int endBIdx;
    int endN1Idx;
    int endS1Idx;
    int endS2Idx;
    int64_t firstSplitKVTaskLseOffset;
    int64_t firstSplitKVTaskOOffset;
};

struct splitNode {
    int batchIdx;
    int headStartIdx;
    int headEndIdx;
    int qStartIdx;
    int qEndIdx;
    int splitNum;
    int64_t lseTaskOffset;
    int64_t oTaskOffset;
};

struct alignas(64) TaskDesc {
    uint32_t batchIdx;
    uint32_t qNBlockIdx;
    uint32_t qSBlockIdx;
    uint32_t kvStart;
    uint32_t kvEnd;
    int32_t startS2Idx;
    int32_t endS2Idx;
    uint32_t isSplitKV;
    uint32_t pipelineDrain;
    uint32_t fdPlanIndex;
    uint32_t headPlanIndex;
    uint32_t maskPlanIndex;
    uint32_t flags;
    uint32_t reserved;
};

static_assert(sizeof(TaskDesc) == 64, "TaskDesc must occupy one 64-byte cache line");

// Batch-shared geometry.  These values are generated once per batch and are
// intentionally kept separate from TaskDesc so the same q/kv geometry is not
// copied into every query/head task.
struct BatchPlan {
    uint32_t qSeqlen;
    uint32_t kvSeqlen;
    uint32_t qNBlockTile;
    uint32_t qNBlockNumPerGroup;
    uint32_t qNBlockNum;
    uint32_t qSBlockTile;
    uint32_t qSBlockNum;
    uint32_t kSBlockTile;
    uint32_t kSBlockNum;
    uint32_t kSBlockTail;
    uint32_t appendStartStack;
    uint32_t flags;
    uint32_t headPlanBase;
    uint32_t headPlanCount;
    uint32_t maskPlanBase;
    uint32_t maskPlanCount;
};

static_assert(sizeof(BatchPlan) == 64, "BatchPlan layout must remain compact");

struct HeadPlan {
    uint32_t qNStart;
    uint32_t qNSize;
    uint32_t kvHeadOffset;
};

struct MaskPlan {
    int32_t windowSizeLeftStartLen;
    int32_t windowSizeLeftEndLen;
    int32_t windowSizeRightStartLen;
    int32_t windowSizeRightEndLen;
    int32_t invalidSuffixStartRow;
    int32_t invalidPrefixEndRow;
    uint32_t noSkipKvS;
    uint32_t kvStart;
    uint32_t kvSLoopNumTotal;
    uint32_t flags;
};

static_assert(sizeof(MaskPlan) == 40, "MaskPlan must remain compact");

struct FDPlan {
    // lseOffset is an LSE element offset; oOffset is an output element offset.
    uint64_t lseOffset;
    uint64_t oOffset;
};

// Batch-shared append-KV phase geometry.  Stack-local mask state remains in
// the KV loop; these values are invariant for all tasks of one batch.
struct AppendPlan {
    uint32_t kvSeqlenOld;
    uint32_t kvStartOld;
    uint32_t kvLoopNumTotalNew;
    uint32_t kvNewSeqlen;
    uint32_t flags;
    uint32_t reserved[3];
};

static_assert(sizeof(AppendPlan) == 32, "AppendPlan must remain compact");

// Batch-shared GM bases. Q/K/V/O are byte offsets; lseBOffset is an element
// offset and blockBOffset is a block-table entry offset.
struct AddressPlan {
    uint64_t qBOffset;
    uint64_t kBOffset;
    uint64_t vBOffset;
    uint64_t oBOffset;
    uint64_t lseBOffset;
    uint64_t blockBOffset;
    uint64_t kNewBOffset;
    uint64_t vNewBOffset;
    uint32_t lseHeadStride;
};

static_assert(sizeof(AddressPlan) == 72, "AddressPlan layout must remain compact");

struct CoreTaskRange {
    uint32_t taskOffset;
    uint32_t taskCount;
};

struct alignas(64) SchedulerHeader {
    uint32_t version;
    uint32_t coreNum;
    uint32_t actualTaskCount;
    uint32_t taskCapacity;
    uint64_t taskDataOffset;
    uint64_t taskDataBytes;
    uint64_t batchPlanOffset;
    uint64_t batchPlanBytes;
    uint64_t headPlanOffset;
    uint64_t headPlanBytes;
    uint64_t fdPlanOffset;
    uint64_t fdPlanBytes;
    uint64_t maskPlanOffset;
    uint64_t maskPlanBytes;
    uint64_t appendPlanOffset;
    uint64_t appendPlanBytes;
    uint64_t addressPlanOffset;
    uint64_t addressPlanBytes;
    uint32_t status;
    uint32_t flashDecodeFlag;
    uint32_t batchPlanCount;
    uint32_t headPlanCount;
    uint32_t fdPlanCount;
    uint32_t maskPlanCount;
    uint32_t appendPlanCount;
    uint32_t addressPlanCount;
};

static_assert(sizeof(SchedulerHeader) == 192, "SchedulerHeader must occupy three cache lines");

struct FAInferTilingData {
    uint32_t numHeads;
    uint32_t embeddingSize;
    uint32_t embeddingSizeV;
    uint32_t numBlocks;
    uint32_t blockSize;
    uint32_t maxQSeqlen;
    uint32_t maxKvSeqlen;
    uint32_t kvHeads;
    uint32_t batch;
    uint32_t maxNumBlocksPerBatch;
    uint32_t firstBatchTaskNum;
    uint32_t totalTaskNum;
    uint32_t maskType;
    uint64_t mm1OutSize;
    uint64_t smOnlineOutSize;
    uint64_t mm2OutSize;
    uint64_t UpdateSize;
    uint64_t workSpaceSize;
    uint64_t padding1;
    uint64_t padding2;
    uint32_t padding3;
    int64_t windowSizeLeft = 0;
    int64_t windowSizeRight = 0;
    uint64_t splitLseTotalSize;
    uint64_t splitOTotalSize;
    uint32_t totalSplitNodeNum;
    uint32_t needCoreNum;
    uint32_t flashDecodeFlag;
    uint32_t numSplits;

    uint32_t kvNewSeqlen;
    uint32_t kvCacheSeqlen;
    coreNode coreInfo[25];
    splitNode splitInfo[25];

    uint32_t get_numHeads() const
    {
        return numHeads;
    }
    uint32_t get_embeddingSize() const
    {
        return embeddingSize;
    }
    uint32_t get_embeddingSizeV() const
    {
        return embeddingSizeV;
    }
    uint32_t get_numBlocks() const
    {
        return numBlocks;
    }
    uint32_t get_blockSize() const
    {
        return blockSize;
    }
    uint32_t get_maxQSeqlen() const
    {
        return maxQSeqlen;
    }
    uint32_t get_maxKvSeqlen() const
    {
        return maxKvSeqlen;
    }
    uint32_t get_kvHeads() const
    {
        return kvHeads;
    }
    uint32_t get_batch() const
    {
        return batch;
    }
    uint32_t get_maxNumBlocksPerBatch() const
    {
        return maxNumBlocksPerBatch;
    }
    uint32_t get_firstBatchTaskNum() const
    {
        return firstBatchTaskNum;
    }
    uint32_t get_totalTaskNum() const
    {
        return totalTaskNum;
    }
    uint32_t get_maskType() const
    {
        return maskType;
    }
    uint64_t get_mm1OutSize() const
    {
        return mm1OutSize;
    }
    uint64_t get_smOnlineOutSize() const
    {
        return smOnlineOutSize;
    }
    uint64_t get_mm2OutSize() const
    {
        return mm2OutSize;
    }
    uint64_t get_UpdateSize() const
    {
        return UpdateSize;
    }
    uint64_t get_workSpaceSize() const
    {
        return workSpaceSize;
    }
    uint64_t get_padding1() const
    {
        return padding1;
    }
    uint64_t get_padding2() const
    {
        return padding2;
    }
    uint32_t get_padding3() const
    {
        return padding3;
    }
    uint64_t get_splitLseTotalSize() const
    {
        return splitLseTotalSize;
    }
    uint64_t get_splitOTotalSize() const
    {
        return splitOTotalSize;
    }
    uint32_t get_totalSplitNodeNum() const
    {
        return totalSplitNodeNum;
    }
    uint32_t get_needCoreNum() const
    {
        return needCoreNum;
    }
    uint32_t get_flashDecodeFlag() const
    {
        return flashDecodeFlag;
    }
    uint32_t get_numSplits() const
    {
        return numSplits;
    }
    uint32_t get_kvNewSeqlen() const
    {
        return kvNewSeqlen;
    }
    uint32_t get_kvCacheSeqlen() const
    {
        return kvCacheSeqlen;
    }

    void set_numHeads(uint32_t value)
    {
        numHeads = value;
    }
    void set_embeddingSize(uint32_t value)
    {
        embeddingSize = value;
    }
    void set_embeddingSizeV(uint32_t value)
    {
        embeddingSizeV = value;
    }
    void set_numBlocks(uint32_t value)
    {
        numBlocks = value;
    }
    void set_blockSize(uint32_t value)
    {
        blockSize = value;
    }
    void set_maxQSeqlen(uint32_t value)
    {
        maxQSeqlen = value;
    }
    void set_maxKvSeqlen(uint32_t value)
    {
        maxKvSeqlen = value;
    }
    void set_kvHeads(uint32_t value)
    {
        kvHeads = value;
    }
    void set_batch(uint32_t value)
    {
        batch = value;
    }
    void set_maxNumBlocksPerBatch(uint32_t value)
    {
        maxNumBlocksPerBatch = value;
    }
    void set_firstBatchTaskNum(uint32_t value)
    {
        firstBatchTaskNum = value;
    }
    void set_totalTaskNum(uint32_t value)
    {
        totalTaskNum = value;
    }
    void set_maskType(uint32_t value)
    {
        maskType = value;
    }

    void set_windowSizeLeft(int64_t value)
    {
        windowSizeLeft = value;
    }
    void set_windowSizeRight(int64_t value)
    {
        windowSizeRight = value;
    }
    void set_mm1OutSize(uint64_t value)
    {
        mm1OutSize = value;
    }
    void set_smOnlineOutSize(uint64_t value)
    {
        smOnlineOutSize = value;
    }
    void set_mm2OutSize(uint64_t value)
    {
        mm2OutSize = value;
    }
    void set_UpdateSize(uint64_t value)
    {
        UpdateSize = value;
    }
    void set_workSpaceSize(uint64_t value)
    {
        workSpaceSize = value;
    }
    void set_padding1(uint64_t value)
    {
        padding1 = value;
    }
    void set_padding2(uint64_t value)
    {
        padding2 = value;
    }
    void set_padding3(uint32_t value)
    {
        padding3 = value;
    }
    void set_splitLseTotalSize(uint64_t value)
    {
        splitLseTotalSize = value;
    }
    void set_splitOTotalSize(uint64_t value)
    {
        splitOTotalSize = value;
    }
    void set_totalSplitNodeNum(uint32_t value)
    {
        totalSplitNodeNum = value;
    }
    void set_needCoreNum(uint32_t value)
    {
        needCoreNum = value;
    }
    void set_flashDecodeFlag(uint32_t value)
    {
        flashDecodeFlag = value;
    }
    void set_numSplits(uint32_t value)
    {
        numSplits = value;
    }
    void set_kvNewSeqlen(uint32_t value)
    {
        kvNewSeqlen = value;
    }
    void set_kvCacheSeqlen(uint32_t value)
    {
        kvCacheSeqlen = value;
    }
};

// Per-forward values accompany the reusable device tiling at kernel launch.
struct FAInferTilingParams {
    uint64_t tilingDataAddr;
    float scaleValue;
    float softcapValue;
};

#endif

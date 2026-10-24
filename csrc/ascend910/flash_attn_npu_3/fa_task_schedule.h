/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef FA_TASK_SCHEDULE_H
#define FA_TASK_SCHEDULE_H

#include "fa_metadata_args.h"

// Host/AICPU producer shared by both metadata paths. This header is not
// included by the device kernel: it replaces task serialization with seeds.
namespace fa_metadata {
struct TaskBoundary {
    uint32_t batch;
    uint32_t head;
    uint32_t query;
    int32_t startS2;
    int32_t endS2;
};

inline bool BoundaryIsSplit(const TaskBoundary& task, const BatchPlan* batches)
{
    return task.endS2 > task.startS2 &&
        static_cast<uint32_t>(task.endS2 - task.startS2) < batches[task.batch].kSBlockNum;
}

inline bool BuildTaskSchedule(SchedulerHeader& header, CoreTaskRange* ranges,
                              BatchPlan* batches, const HeadPlan* heads,
                              const FAInferTilingData& tiling)
{
    header.status = TASK_METADATA_OVERFLOW;
    if (header.coreNum == 0U) return false;
    const bool fd = tiling.flashDecodeFlag != 0U;
    uint64_t total = 0;
    for (uint32_t b = 0; b < header.batchPlanCount; ++b) {
        BatchPlan& plan = batches[b];
        plan.taskBase = static_cast<uint32_t>(total);
        total += static_cast<uint64_t>(plan.qNBlockNum) * plan.qSBlockNum;
        if (total > UINT32_MAX) return false;
        const uint32_t innerCount = fd ? plan.qSBlockNum : plan.qNBlockNum;
        const uint32_t step = fd ? 1U : header.coreNum;
        plan.taskStepOuter = innerCount == 0U ? 0U : step / innerCount;
        plan.taskStepInner = innerCount == 0U ? 0U : step % innerCount;
        plan.copybackQSBlock = plan.qSBlockNum == 0U ? UINT32_MAX : b % plan.qSBlockNum;
    }
    uint64_t scheduled = 0;
    uint32_t batchCursor = 0;
    for (uint32_t core = 0; core < header.coreNum; ++core) {
        CoreTaskRange& range = ranges[core];
        range = {};
        if (!fd) {
            range.taskStart = core;
            range.taskCount = total <= core ? 0U :
                static_cast<uint32_t>((total - 1U - core) / header.coreNum + 1U);
            if (range.taskCount == 0U) continue;
            while (core >= batches[batchCursor].taskBase +
                   batches[batchCursor].qNBlockNum * batches[batchCursor].qSBlockNum) ++batchCursor;
            const BatchPlan& plan = batches[batchCursor];
            const uint32_t local = core - plan.taskBase;
            range.firstBatch = batchCursor;
            range.firstOuter = local / plan.qNBlockNum;
            range.firstInner = local % plan.qNBlockNum;
        } else {
            if (core >= sizeof(tiling.coreInfo) / sizeof(tiling.coreInfo[0]) ||
                (tiling.needCoreNum != 0U && core >= tiling.needCoreNum)) continue;
            const coreNode& node = tiling.coreInfo[core];
            if (node.startBIdx > node.endBIdx) continue;
            if (node.startBIdx < 0 || static_cast<uint32_t>(node.endBIdx) >= header.batchPlanCount) return false;
            TaskBoundary first{}, last{};
            uint32_t previousEnd = 0;
            for (int32_t b = node.startBIdx; b <= node.endBIdx; ++b) {
                const BatchPlan& plan = batches[b];
                const int32_t nBegin = b == node.startBIdx ? node.startN1Idx : 0;
                const int32_t nEnd = b == node.endBIdx ? node.endN1Idx : static_cast<int32_t>(plan.qNBlockNum) - 1;
                if (nBegin > nEnd) continue;
                if (nBegin < 0 || static_cast<uint32_t>(nEnd) >= plan.qNBlockNum) return false;
                for (int32_t n = nBegin; n <= nEnd; ++n) {
                    const bool firstHead = b == node.startBIdx && n == nBegin;
                    const bool lastHead = b == node.endBIdx && n == nEnd;
                    const int32_t sBegin = firstHead ? node.startS1Idx : 0;
                    const int32_t sEnd = lastHead ? node.endS1Idx : static_cast<int32_t>(plan.qSBlockNum) - 1;
                    if (sBegin > sEnd) continue;
                    if (sBegin < 0 || static_cast<uint32_t>(sEnd) >= plan.qSBlockNum) return false;
                    const uint32_t begin = plan.taskBase + static_cast<uint32_t>(n) * plan.qSBlockNum + sBegin;
                    const uint32_t end = plan.taskBase + static_cast<uint32_t>(n) * plan.qSBlockNum + sEnd;
                    if (range.taskCount != 0U && begin != previousEnd + 1U) return false;
                    if (range.taskCount == 0U) {
                        range.taskStart = begin;
                        range.firstBatch = static_cast<uint32_t>(b);
                        range.firstInner = static_cast<uint32_t>(sBegin);
                        range.firstOuter = static_cast<uint32_t>(n);
                        first = {static_cast<uint32_t>(b), static_cast<uint32_t>(n), static_cast<uint32_t>(sBegin),
                            firstHead ? node.startS2Idx : 0,
                            lastHead && sBegin == sEnd ? node.endS2Idx : static_cast<int32_t>(plan.kSBlockNum)};
                    }
                    last = {static_cast<uint32_t>(b), static_cast<uint32_t>(n), static_cast<uint32_t>(sEnd),
                        firstHead && sBegin == sEnd ? node.startS2Idx : 0,
                        lastHead ? node.endS2Idx : static_cast<int32_t>(plan.kSBlockNum)};
                    range.taskCount += end - begin + 1U;
                    previousEnd = end;
                }
            }
            if (range.taskCount == 0U) continue;
            range.firstS2 = first.startS2;
            range.lastS2 = last.endS2;
            range.firstLseOffset = static_cast<uint64_t>(node.firstSplitKVTaskLseOffset);
            range.firstOOffset = static_cast<uint64_t>(node.firstSplitKVTaskOOffset);
            range.lastLseOffset = range.firstLseOffset;
            range.lastOOffset = range.firstOOffset;
            if (BoundaryIsSplit(first, batches)) {
                range.flags |= CORE_FIRST_TASK_SPLIT;
                if (range.taskCount > 1U) {
                    const BatchPlan& plan = batches[first.batch];
                    const uint32_t qSize = first.query + 1U == plan.qSBlockNum
                        ? plan.qSeqlen - first.query * plan.qSBlockTile : plan.qSBlockTile;
                    const uint64_t rows = static_cast<uint64_t>(qSize) * heads[plan.headPlanBase + first.head].qNSize;
                    range.lastLseOffset += rows;
                    range.lastOOffset += rows * tiling.embeddingSizeV;
                }
            }
            if (BoundaryIsSplit(last, batches)) range.flags |= CORE_LAST_TASK_SPLIT;
        }
        scheduled += range.taskCount;
        if (scheduled > UINT32_MAX) return false;
    }
    header.actualTaskCount = static_cast<uint32_t>(scheduled);
    header.status = TASK_METADATA_OK;
    return true;
}
} // namespace fa_metadata
#endif

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KERNEL_COMMON
#define KERNEL_COMMON

#include "kernel_operator.h"

constexpr int32_t NUM1 = 1;
constexpr int32_t NUM4 = 4;

constexpr int32_t NUM64 = 64;
constexpr int32_t NUM512 = 512;
constexpr int32_t NUM576 = 576;
constexpr int32_t BASIC_BLOCK_SIZE = 256;
constexpr int32_t Q_BLK = 256;
constexpr int32_t MAX_STACK_LEN = 512;

constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
constexpr uint32_t Q_M_TILE_MAX = 128;
constexpr uint32_t Q_N_SPLIT_ALIGN = 2;

constexpr uint32_t UNIT_BLOCK_STACK_NUM = 4;

struct FAIKernelParams {
    // Data members
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR mask;
    GM_ADDR blockTables;
    GM_ADDR actualQseqlen;
    GM_ADDR actualKvseqlen;
    GM_ADDR o;
    GM_ADDR lse;
    GM_ADDR workSpace;
    GM_ADDR tiling;

    // Methods
    __aicore__ inline FAIKernelParams() {}

    __aicore__ inline FAIKernelParams(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR mask_, GM_ADDR blockTables_,
            GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_, GM_ADDR o_, GM_ADDR lse_, GM_ADDR workSpace_, GM_ADDR tiling_)
        : q(q_), k(k_), v(v_), mask(mask_), blockTables(blockTables_), actualQseqlen(actualQseqlen_),
            actualKvseqlen(actualKvseqlen_), o(o_), lse(lse_), workSpace(workSpace_), tiling(tiling_) {}
};

enum class Format
{
    TND = 0,
    BSND = 1
};

enum class CacheMode 
{
    normalCache = 0,
    pagedCache = 1,
};

enum class PageShape 
{
    BnBsND = 0,
    BnNBsD = 1,
    normalShape = 2,
};

enum class MaskCategory 
{
    NO_MASK = 0,
    MASK_CAUSAL = 1,
    MASK_SWA = 4,
};

enum class CacheLayout : uint8_t
{
    nd = 0,
    nz = 1,
};

// The grouped-Q path keeps the logical M order as [head][S].  FixPipe owns a
// private UB view per AIV, so its physical extent must be rounded *after*
// logical ownership has been decided.  Do not replace this with
// RoundUp(rowNum, align) / subBlockNum: that loses the head boundary for odd
// groups and small S tiles.
struct FAIGroupedRowPartition {
    uint32_t logicalRowStart;
    uint32_t storageRowStart;
    uint32_t validRows;
    uint32_t physicalRows;
};

__aicore__ inline FAIGroupedRowPartition GetFAIGroupedRowPartition(
    uint32_t qSBlockSize, uint32_t qNBlockSize, uint32_t align)
{
    uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
    uint32_t subBlockNum = AscendC::GetSubBlockNum();
    uint32_t rowNum = qSBlockSize * qNBlockSize;
    uint32_t split = 0;
    if (qNBlockSize == 1U) {
        // Preserve the original single-head FixPipe partition: the M extent
        // is aligned before splitting, so a 20-row tail is handled as 12+8,
        // not 10+10.
        split = (rowNum + align - 1U) / align * align / subBlockNum;
        split = rowNum < split ? rowNum : split;
    } else {
        split = qSBlockSize * (qNBlockSize / subBlockNum);
    }
    uint32_t logicalRowStart = subBlockIdx == 0U ? 0U : split;
    // QK/PV FixPipe splits grouped work into two equal physical M regions.
    // non-DN pads each AIV half to 16 rows; DN pads each half to 32 rows.
    // For an odd number of heads, AIV1 owns one more logical head; reserve
    // both regions to the larger pad-aligned extent.
    uint32_t storageRowStart = logicalRowStart;
    if (qNBlockSize > 1U && subBlockIdx != 0U) {
        uint32_t padUnit = align >= 32U ? 32U : 16U;
        uint32_t firstPhysicalRows = ((split + padUnit - 1U) / padUnit) * padUnit;
        uint32_t secondLogicalRows = rowNum - split;
        uint32_t secondPhysicalRows =
            ((secondLogicalRows + padUnit - 1U) / padUnit) * padUnit;
        storageRowStart = firstPhysicalRows > secondPhysicalRows ?
            firstPhysicalRows : secondPhysicalRows;
    }
    uint32_t validRows = subBlockIdx == 0U ? split : rowNum - split;
    uint32_t physicalRows = (validRows + align - 1U) / align * align;
    return {logicalRowStart, storageRowStart, validRows, physicalRows};
}

__aicore__ inline uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize,
                                          bool restrictMergedRowsForLargeD = false)
{
    uint32_t tile = qSeqlen == 0U ? Q_M_TILE_MAX :
        (Q_M_TILE_MAX / qSeqlen) / Q_N_SPLIT_ALIGN * Q_N_SPLIT_ALIGN;
    if (restrictMergedRowsForLargeD && qSeqlen != 0U) {
        constexpr uint32_t MAX_M_FOR_LARGE_D = Q_M_TILE_MAX / 2U;
        uint32_t maxTile = MAX_M_FOR_LARGE_D / qSeqlen;
        maxTile = maxTile > 0U ? maxTile : 1U;
        tile = tile < maxTile ? tile : maxTile;
    }
    tile = tile < groupSize ? tile : groupSize;
    return tile < 1U ? 1U : tile;
}


#endif

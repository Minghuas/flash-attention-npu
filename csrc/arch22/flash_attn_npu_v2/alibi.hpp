/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */
#ifndef COMMON_ALIBI_BIAS_HPP
#define COMMON_ALIBI_BIAS_HPP

namespace Alibi {
enum class AlibiMaskType : uint32_t {
    NO_MASK = 0,
    MASK_CAUSAL = 1,
    MASK_SWA = 4
};

__aicore__ inline void RescaleBiasRow(AscendC::LocalTensor<float> &workUb,
                                      float slope, float preSlope, int32_t count)
{
    AscendC::Muls<float>(workUb, workUb, slope / preSlope, count);  
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void BuildAbsBiasRow(AscendC::LocalTensor<float> &workUb,
                                       int64_t baseColIdx, float slope, int32_t count)
{
    AscendC::CreateVecIndex<float>(workUb, static_cast<float>(-baseColIdx), count);
    AscendC::PipeBarrier<PIPE_V>();
    // if (baseColIdx >= 0) {
    //     int32_t absCnt = baseColIdx < count ? baseColIdx : count;
    //     AscendC::Abs<float>(workUb, workUb, absCnt);
    //     AscendC::PipeBarrier<PIPE_V>();
    // }
    AscendC::Abs<float>(workUb, workUb, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls<float>(workUb, workUb, -slope, count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void AddBiasToRow(AscendC::LocalTensor<float> &scoreUb, uint32_t rowOff,
                                    AscendC::LocalTensor<float> &workUb, int32_t count)
{
    AscendC::Add<float>(scoreUb[rowOff], scoreUb[rowOff], workUb, count);
    AscendC::PipeBarrier<PIPE_V>();
}

// ============================================================================
// ApplyAlibiRows — adds ALiBi bias to each score row.
//
// Team-confirmed formulas (2026-07-23):
//   qSIdx   = qSBlockBaseIdx + token       (Q-sequence position)
//   qKvSIdx  = alibiDiffS + qSIdx          (mapped to K-coordinate)
//   qNIdx   = qNBlockBaseIdx + head        (global head index)
//   baseColIdx = qKvSIdx - kvSStartIdx     (bias V-shape center, can be negative)
//   slope   = slopesGm[slopesBatchOffset + qNIdx]
//   bias[col] = -slope * |baseColIdx - col|
//
// Type convention: ALiBi position indices (qSBlockBaseIdx/alibiDiffS/qNBlockBaseIdx/
// kvSStartIdx/baseColIdx) are int64_t (avoid uint32_t underflow / signed-mixed UB).
// slopesBatchOffset is int64_t (GM byte offset, can be large).
// Existing kernel counts (columnNum/rowNumCurLoop/rowStride/scoreOffset/absRowStart/
// qSBlockSize) stay uint32_t; helper `count` params are int32_t (AscendC intrinsic 约定).
// ============================================================================


template <AlibiMaskType MASK_TYPE>
__aicore__ inline void ApplyAlibiRows(
    AscendC::LocalTensor<float> &scoreUb, uint32_t scoreOffset,
    uint32_t rowStride, uint32_t columnNum,
    int64_t absRowStart, uint32_t rowNumCurLoop,
    uint32_t qSBlockSize, int64_t qSBlockBaseIdx, 
    int64_t qNBlockBaseIdx, int64_t alibiDiffS, 
    AscendC::GlobalTensor<float> &slopesGm, int64_t slopesBatchOffset,
    AscendC::LocalTensor<float> &workUb, int64_t kvSStartIdx);

// template <>
// __aicore__ inline void ApplyAlibiRows<AlibiMaskType::MASK_CAUSAL>(
    // AscendC::LocalTensor<float> &scoreUb, uint32_t scoreOffset,
    // uint32_t rowStride, uint32_t columnNum,
    // int64_t absRowStart, uint32_t rowNumCurLoop,
    // uint32_t qSBlockSize, int64_t qSBlockBaseIdx, 
    // int64_t qNBlockBaseIdx, int64_t alibiDiffS, 
    // AscendC::GlobalTensor<float> &slopesGm, int64_t slopesBatchOffset,
    // AscendC::LocalTensor<float> &workUb, int64_t kvSStartIdx)
// {
//     if (rowNumCurLoop == 0 || qSBlockSize == 0 || columnNum == 0) {
//         return;
//     }
//     int32_t count = static_cast<int32_t>(columnNum);

//     AscendC::CreateVecIndex<float>(workUb, static_cast<float>(kvSStartIdx), count); 
//     AscendC::PipeBarrier<PIPE_V>();
//     float preSlope = 1.0f;

//     for (uint32_t ri = 0; ri < rowNumCurLoop; ++ri) {
//         int64_t absRow = static_cast<int64_t>(absRowStart) + ri;
//         int64_t head   = absRow / qSBlockSize;
//         int64_t qNIdx  = qNBlockBaseIdx + head;
//         float slope  = slopesGm.GetValue(slopesBatchOffset + qNIdx);  
//         if (slope != preSlope) {
//             RescaleBiasRow(workUb, slope, preSlope, count);
//             preSlope = slope;
//         }
//         AddBiasToRow(scoreUb, scoreOffset + ri * rowStride, workUb, count);
//     }
// }


template <>
__aicore__ inline void ApplyAlibiRows<AlibiMaskType::NO_MASK>(
    AscendC::LocalTensor<float> &scoreUb, uint32_t scoreOffset,
    uint32_t rowStride, uint32_t columnNum,
    int64_t absRowStart, uint32_t rowNumCurLoop,
    uint32_t qSBlockSize, int64_t qSBlockBaseIdx, 
    int64_t qNBlockBaseIdx, int64_t alibiDiffS, 
    AscendC::GlobalTensor<float> &slopesGm, int64_t slopesBatchOffset,
    AscendC::LocalTensor<float> &workUb, int64_t kvSStartIdx)
{
    if (rowNumCurLoop == 0 || qSBlockSize == 0 || columnNum == 0) {
        return;
    }
    for (uint32_t ri = 0; ri < rowNumCurLoop; ++ri) {
        int64_t absRow = absRowStart + static_cast<int64_t>(ri);
        int64_t qNIdx = qNBlockBaseIdx + absRow / qSBlockSize;
        int64_t baseColIdx = alibiDiffS + qSBlockBaseIdx + (absRow % qSBlockSize) - kvSStartIdx;
        float slope = slopesGm.GetValue(slopesBatchOffset + qNIdx);

        BuildAbsBiasRow(workUb, baseColIdx, slope, columnNum);
        AddBiasToRow(scoreUb, scoreOffset + ri * rowStride, workUb, columnNum);
    }
}

template <>
__aicore__ inline void ApplyAlibiRows<AlibiMaskType::MASK_SWA>(
    AscendC::LocalTensor<float> &scoreUb, uint32_t scoreOffset,
    uint32_t rowStride, uint32_t columnNum,
    int64_t absRowStart, uint32_t rowNumCurLoop,
    uint32_t qSBlockSize, int64_t qSBlockBaseIdx, 
    int64_t qNBlockBaseIdx, int64_t alibiDiffS, 
    AscendC::GlobalTensor<float> &slopesGm, int64_t slopesBatchOffset,
    AscendC::LocalTensor<float> &workUb, int64_t kvSStartIdx)
{
    // TODO: implement SWA-specific optimized handling.
    ApplyAlibiRows<AlibiMaskType::NO_MASK>(scoreUb, scoreOffset, rowStride, columnNum,
        absRowStart, rowNumCurLoop, qSBlockSize,
        qSBlockBaseIdx, qNBlockBaseIdx, alibiDiffS, 
        slopesGm, slopesBatchOffset, workUb, kvSStartIdx);
}

template <>
__aicore__ inline void ApplyAlibiRows<AlibiMaskType::MASK_CAUSAL>(
    AscendC::LocalTensor<float> &scoreUb, uint32_t scoreOffset,
    uint32_t rowStride, uint32_t columnNum,
    int64_t absRowStart, uint32_t rowNumCurLoop,
    uint32_t qSBlockSize, int64_t qSBlockBaseIdx, 
    int64_t qNBlockBaseIdx, int64_t alibiDiffS, 
    AscendC::GlobalTensor<float> &slopesGm, int64_t slopesBatchOffset,
    AscendC::LocalTensor<float> &workUb, int64_t kvSStartIdx)
{
    
    ApplyAlibiRows<AlibiMaskType::NO_MASK>(scoreUb, scoreOffset, rowStride, columnNum,
        absRowStart, rowNumCurLoop, qSBlockSize,
        qSBlockBaseIdx, qNBlockBaseIdx, alibiDiffS, 
        slopesGm, slopesBatchOffset, workUb, kvSStartIdx);
}

}
#endif

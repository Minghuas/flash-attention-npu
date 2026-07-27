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
                                      const float slope, const float preSlope, uint32_t N)
{
    AscendC::Muls<float>(workUb, workUb, slope / preSlope, N);  
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void BuildAbsBiasRow(AscendC::LocalTensor<float> &workUb,
                                       int32_t baseColIdx, const float slope, uint32_t N)
{
    AscendC::CreateVecIndex<float>(workUb, static_cast<float>(-baseColIdx), N);
    AscendC::PipeBarrier<PIPE_V>();
    // if (baseColIdx >= 0) {
    //     int32_t absCnt = baseColIdx < N ? baseColIdx : N;
    //     AscendC::Abs<float>(workUb, workUb, absCnt);
    //     AscendC::PipeBarrier<PIPE_V>();
    // }
    AscendC::Abs<float>(workUb, workUb, N);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls<float>(workUb, workUb, -slope, N);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void AddBiasToRow(AscendC::LocalTensor<float> &scoreUb, uint32_t rowOff,
                                    AscendC::LocalTensor<float> &workUb, uint32_t N)
{
    AscendC::Add<float>(scoreUb[rowOff], scoreUb[rowOff], workUb, N);
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
// Type convention: position indices are uint32_t; slopesBatchOffset is int64_t
// (the only signed parameter — it's a GM byte offset that can be large).
// ============================================================================


template <AlibiMaskType MASK_TYPE>
__aicore__ inline void ApplyAlibiRows(
    AscendC::LocalTensor<float> &scoreUb, uint32_t scoreOffset,
    uint32_t rowStride, uint32_t columnNum,
    uint32_t absRowStart, uint32_t rowNumCurLoop,
    uint32_t qSBlockSize,
    uint32_t qSBlockBaseIdx, uint32_t alibiDiffS,
    uint32_t qNBlockBaseIdx,
    AscendC::GlobalTensor<float> &slopesGm, int64_t slopesBatchOffset,
    AscendC::LocalTensor<float> &workUb,
    int32_t kvSStartIdx);

// template <>
// __aicore__ inline void ApplyAlibiRows<AlibiMaskType::MASK_CAUSAL>(
//     AscendC::LocalTensor<float> &scoreUb, uint32_t scoreOffset,
//     uint32_t rowStride, uint32_t columnNum,
//     uint32_t absRowStart, uint32_t rowNumCurLoop,
//     uint32_t qSBlockSize,
//     uint32_t qSBlockBaseIdx, uint32_t alibiDiffS,
//     uint32_t qNBlockBaseIdx,
//     AscendC::GlobalTensor<float> &slopesGm, int64_t slopesBatchOffset,
//     AscendC::LocalTensor<float> &workUb,
//     int32_t kvSStartIdx)
// {
//     if (rowNumCurLoop == 0 || qSBlockSize == 0 || columnNum == 0) {
//         return;
//     }
//     const int32_t count = static_cast<int32_t>(columnNum);

//     AscendC::CreateVecIndex<float>(workUb, static_cast<float>(kvSStartIdx), count); 
//     AscendC::PipeBarrier<PIPE_V>();
//     float preSlope = 1.0f;

//     for (uint32_t ri = 0; ri < rowNumCurLoop; ++ri) {
//         const uint32_t absRow = absRowStart + ri;
//         const uint32_t head   = absRow / qSBlockSize;          
//         uint32_t qNIdx      = qNBlockBaseIdx + head;
//         const float slope  = slopesGm.GetValue(slopesBatchOffset + qNIdx);  
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
    uint32_t absRowStart, uint32_t rowNumCurLoop,
    uint32_t qSBlockSize,
    uint32_t qSBlockBaseIdx, uint32_t alibiDiffS,
    uint32_t qNBlockBaseIdx,
    AscendC::GlobalTensor<float> &slopesGm, int64_t slopesBatchOffset,
    AscendC::LocalTensor<float> &workUb,
    int32_t kvSStartIdx)
{
    if (rowNumCurLoop == 0 || qSBlockSize == 0 || columnNum == 0) {
        return;
    }
    for (uint32_t ri = 0; ri < rowNumCurLoop; ++ri) {
        uint32_t absRow = absRowStart + ri;
        uint32_t head   = absRow / qSBlockSize;
        uint32_t token  = absRow % qSBlockSize;

        uint32_t qSIdx      = qSBlockBaseIdx + token;
        uint32_t qKvSIdx    = alibiDiffS + qSIdx;
        uint32_t qNIdx      = qNBlockBaseIdx + head;
        int32_t  baseColIdx = qKvSIdx - kvSStartIdx;   
        const float    slope      = slopesGm.GetValue(slopesBatchOffset + qNIdx);

        BuildAbsBiasRow(workUb, baseColIdx, slope, columnNum);
        AddBiasToRow(scoreUb, scoreOffset + ri * rowStride, workUb, columnNum);
    }
}

template <>
__aicore__ inline void ApplyAlibiRows<AlibiMaskType::MASK_SWA>(
    AscendC::LocalTensor<float> &scoreUb, uint32_t scoreOffset,
    uint32_t rowStride, uint32_t columnNum,
    uint32_t absRowStart, uint32_t rowNumCurLoop,
    uint32_t qSBlockSize,
    uint32_t qSBlockBaseIdx, uint32_t alibiDiffS,
    uint32_t qNBlockBaseIdx,
    AscendC::GlobalTensor<float> &slopesGm, int64_t slopesBatchOffset,
    AscendC::LocalTensor<float> &workUb,
    int32_t kvSStartIdx)
{
    // TODO: implement SWA-specific optimized handling.
    ApplyAlibiRows<AlibiMaskType::NO_MASK>(scoreUb, scoreOffset, rowStride, columnNum,
        absRowStart, rowNumCurLoop, qSBlockSize,
        qSBlockBaseIdx, alibiDiffS, qNBlockBaseIdx,
        slopesGm, slopesBatchOffset, workUb, kvSStartIdx);
}

template <>
__aicore__ inline void ApplyAlibiRows<AlibiMaskType::MASK_CAUSAL>(
    AscendC::LocalTensor<float> &scoreUb, uint32_t scoreOffset,
    uint32_t rowStride, uint32_t columnNum,
    uint32_t absRowStart, uint32_t rowNumCurLoop,
    uint32_t qSBlockSize,
    uint32_t qSBlockBaseIdx, uint32_t alibiDiffS,
    uint32_t qNBlockBaseIdx,
    AscendC::GlobalTensor<float> &slopesGm, int64_t slopesBatchOffset,
    AscendC::LocalTensor<float> &workUb,
    int32_t kvSStartIdx)
{
    
    ApplyAlibiRows<AlibiMaskType::NO_MASK>(scoreUb, scoreOffset, rowStride, columnNum,
        absRowStart, rowNumCurLoop, qSBlockSize,
        qSBlockBaseIdx, alibiDiffS, qNBlockBaseIdx,
        slopesGm, slopesBatchOffset, workUb, kvSStartIdx);
}

}
#endif

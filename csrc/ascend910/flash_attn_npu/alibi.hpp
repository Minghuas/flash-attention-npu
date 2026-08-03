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
                                       int64_t baseColIdx, float slope, int32_t count,
                                    AscendC::TEventID eventIdSToV, AscendC::TEventID eventIdVToS)
{
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);
    AscendC::CreateVecIndex<float>(workUb, static_cast<float>(-baseColIdx), count);

    AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIdSToV);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIdSToV);

    AscendC::PipeBarrier<PIPE_V>();
    // if (baseColIdx >= 0) {
    //     int32_t absCnt = baseColIdx < count ? baseColIdx : count;
    //     AscendC::Abs<float>(workUb, workUb, absCnt);
    //     AscendC::PipeBarrier<PIPE_V>();
    // }
    AscendC::Muls<float>(workUb, workUb, -slope, count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void BuildAbsBiasRow2(AscendC::LocalTensor<float> &workUb,
    int64_t baseColIdx, float slope, int32_t count,
    AscendC::TEventID eventIdSToV, AscendC::TEventID eventIdVToS)
{
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);

    int32_t colIdx = 0;
    for (; colIdx <= -baseColIdx && colIdx < count; colIdx++) {
        workUb.SetValue(colIdx, static_cast<float>(-baseColIdx - colIdx));
    }
    for (; colIdx < count; colIdx++) {
        workUb.SetValue(colIdx, static_cast<float>(colIdx + baseColIdx));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIdSToV);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIdSToV);
    AscendC::PipeBarrier<PIPE_V>();
    
    AscendC::Muls<float>(workUb, workUb, -slope, count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void BuildAbsBiasRow3(AscendC::LocalTensor<float> &workUb, 
    int64_t baseColIdx, float slope, int32_t count, 
    AscendC::TEventID eventIdSToV, AscendC::TEventID eventIdVToS)
{
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);

    int32_t colIdx = 0;
    float value = slope * static_cast<float>(baseColIdx);
    for (; colIdx < -baseColIdx && colIdx < count; ++colIdx) {
        workUb.SetValue(colIdx, value);
        value += slope;
    }
    value = -slope * static_cast<float>(colIdx + baseColIdx);
    for (; colIdx < count; ++colIdx) {
        workUb.SetValue(colIdx, value);
        value -= slope;
    }

    AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIdSToV);	 
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIdSToV);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void AddBiasToRow(AscendC::LocalTensor<float> &scoreUb, uint32_t rowOff,
                                    AscendC::LocalTensor<float> &workUb, int32_t count)
{
    AscendC::Add<float>(scoreUb[rowOff], scoreUb[rowOff], workUb, count);
    AscendC::PipeBarrier<PIPE_V>();
}

// 方式4：直接在 scoreUb 上累加 bias（不需要 workUb）
// 使用 operator() 返回 float& 引用（左值=SetValue, 右值=GetValue）
__aicore__ inline void ApplyAlibiBiasDirect(
    AscendC::LocalTensor<float> &scoreUb, uint32_t rowOff,
    int64_t baseColIdx, float slope, int32_t count,
    AscendC::TEventID eventIdSToV, AscendC::TEventID eventIdVToS)
{
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);

    auto scoreRowUb = scoreUb[rowOff];
    int32_t colIdx = 0;
    float bias = slope * static_cast<float>(baseColIdx);
    for (; colIdx < -baseColIdx && colIdx < count; ++colIdx) {
        scoreRowUb(colIdx) = scoreRowUb(colIdx) + bias;
        bias += slope;
    }
    bias = -slope * static_cast<float>(colIdx + baseColIdx);
    for (; colIdx < count; ++colIdx) {
        scoreRowUb(colIdx) = scoreRowUb(colIdx) + bias;
        bias -= slope;
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIdSToV);	 
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIdSToV);
    AscendC::PipeBarrier<PIPE_V>();
}




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
    if (rowNumCurLoop == 0 || columnNum == 0) {
        return;
    }

    AscendC::TEventID eventIdSToV = GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V);
    AscendC::TEventID eventIdVToS = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S);

    for (uint32_t ri = 0; ri < rowNumCurLoop; ++ri) {
        int64_t absRow = absRowStart + static_cast<int64_t>(ri);
        int64_t qNIdx = qNBlockBaseIdx + absRow / qSBlockSize;
        float slope = slopesGm.GetValue(slopesBatchOffset + qNIdx);

        // int64_t baseColIdx = kvSStartIdx - (alibiDiffS + qSBlockBaseIdx + (absRow % qSBlockSize));
        // // BuildAbsBiasRow2(workUb, baseColIdx, slope, columnNum, eventIdSToV, eventIdVToS);
        // BuildAbsBiasRow3(workUb, baseColIdx, slope, columnNum, eventIdSToV, eventIdVToS);
        // AddBiasToRow(scoreUb, scoreOffset + ri * rowStride, workUb, columnNum);

        int64_t baseColIdx = kvSStartIdx - (alibiDiffS + qSBlockBaseIdx + (absRow % qSBlockSize));
        ApplyAlibiBiasDirect(scoreUb, scoreOffset + ri * rowStride, baseColIdx, slope, columnNum, eventIdSToV, eventIdVToS);

        // TODO: 用两份workUb做pingpong优化
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
    // TODO: implement SWA-specific optimized method.
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

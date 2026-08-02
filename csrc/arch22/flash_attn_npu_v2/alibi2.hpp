/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */
#ifndef COMMON_ALIBI_BIAS_HPP
#define COMMON_ALIBI_BIAS_HPP

__aicore__ inline void RescaleBiasRow(AscendC::LocalTensor<float> &workUb,
                                      float slope, float preSlope, int32_t count)
{
    AscendC::Muls<float>(workUb, workUb, slope / preSlope, count);  
    AscendC::PipeBarrier<PIPE_V>();
}


__aicore__ inline void BuildAlibiBiasRowCausal(AscendC::LocalTensor<float> &workUb, 
    int64_t kvSStartIdx, float slope, int32_t count, 
    AscendC::TEventID eventIdSToV, AscendC::TEventID eventIdVToS)
{
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);
    // causal bias[colIdx] = slope * colIdx
    for (int32_t col = 0; col < count; ++col) {
        workUb.SetValue(col, slope * static_cast<float>(kvSStartIdx + col));
    }

    AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIdSToV);	 
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIdSToV);
    AscendC::PipeBarrier<PIPE_V>();
}


__aicore__ inline void BuildAlibiBiasRow2(AscendC::LocalTensor<float> &workUb,
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

__aicore__ inline void BuildAlibiBiasRow(AscendC::LocalTensor<float> &workUb, 
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

__aicore__ inline void BuildAlibiBiasRow3a(AscendC::LocalTensor<float> &workUb, 
    int64_t baseColIdx, float slope, int32_t count, 
    AscendC::TEventID eventIdSToV, AscendC::TEventID eventIdVToS)
{
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);

    int n = 8;  // FLOAT_PER_DATABLOCK，只对第 1 个 datablock 中的元素进行标量操作
    for (int i = 0; i < n; i++) {
        workUb.SetValue(i, static_cast<float>(baseColIdx + i) * slope);
    }

    AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIdSToV);  
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIdSToV);
    
    while (n < count) {
        if (2 * n < count) {
            AscendC::Adds(workUb[n], workUb, static_cast<float>(n) * slope, n);
            AscendC::PipeBarrier<PIPE_V>();
            n *= 2;
        } else {
            AscendC::Adds(workUb[n], workUb, static_cast<float>(n) * slope, count - n);
            AscendC::PipeBarrier<PIPE_V>();
            break;
        }
    }
    
    if (baseColIdx < 0) {
        AscendC::Abs(workUb, workUb, AscendC::Std::min(-baseColIdx, count));
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void AddBiasToScoreRow(AscendC::LocalTensor<float> &scoreUb, uint32_t rowOff,
                                    AscendC::LocalTensor<float> &workUb, int32_t count)
{
    AscendC::Add<float>(scoreUb[rowOff], scoreUb[rowOff], workUb, count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void SubBiasToScoreRow(AscendC::LocalTensor<float> &scoreUb, uint32_t rowOff,
                                    AscendC::LocalTensor<float> &workUb, int32_t count)
{
    AscendC::Sub<float>(scoreUb[rowOff], scoreUb[rowOff], workUb, count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void ApplyAlibi(
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
        int64_t baseColIdx = kvSStartIdx - (alibiDiffS + qSBlockBaseIdx + (absRow % qSBlockSize));
        // // BuildAlibiBiasRow(workUb, baseColIdx, slope, columnNum, eventIdSToV, eventIdVToS);
        // BuildAlibiBiasRow2(workUb, baseColIdx, slope, columnNum, eventIdSToV, eventIdVToS);
        // AddBiasToScoreRow(scoreUb, scoreOffset + ri * rowStride, workUb, columnNum);
    
        BuildAlibiBiasRow3a(workUb, baseColIdx, slope, columnNum, eventIdSToV, eventIdVToS);
        SubBiasToScoreRow(scoreUb, scoreOffset + ri * rowStride, workUb, columnNum);
    }

    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::S_V>(eventIdSToV);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_S>(eventIdVToS);
}

__aicore__ inline void ApplyAlibiCausal(
    AscendC::LocalTensor<float> &scoreUb, uint32_t scoreOffset,
    uint32_t rowStride, uint32_t columnNum,
    int64_t absRowStart, uint32_t rowNumCurLoop,
    uint32_t qSBlockSize, int64_t qSBlockBaseIdx, 
    int64_t qNBlockBaseIdx, int64_t alibiDiffS, 
    AscendC::GlobalTensor<float> &slopesGm, int64_t slopesBatchOffset,
    AscendC::LocalTensor<float> &workUb, int64_t kvSStartIdx)
{
    ApplyAlibi(scoreUb, scoreOffset,
    rowStride, columnNum,
    absRowStart, rowNumCurLoop,
    qSBlockSize, qSBlockBaseIdx, 
    qNBlockBaseIdx, alibiDiffS, 
    slopesGm, slopesBatchOffset,
    workUb, kvSStartIdx);
}

// Optimized method for causal mask type: mask out position that rowIdx < colIdx
__aicore__ inline void ApplyAlibiCausal2(
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

    int64_t qNIdx = qNBlockBaseIdx + absRowStart / qSBlockSize;
    float slope = slopesGm.GetValue(slopesBatchOffset + qNIdx);

    BuildAlibiBiasRowCausal(workUb, kvSStartIdx, slope, columnNum, eventIdSToV, eventIdVToS);
    AddBiasToScoreRow(scoreUb, scoreOffset, workUb, columnNum);
    
    for (uint32_t ri = 1; ri < rowNumCurLoop; ++ri) {
        int64_t absRow = absRowStart + static_cast<int64_t>(ri);
        qNIdx = qNBlockBaseIdx + absRow / qSBlockSize;
        float new_slope = slopesGm.GetValue(slopesBatchOffset + qNIdx);
        float diff = slope > new_slope ? slope - new_slope : new_slope - slope;
        if (diff > 1e-4f) {
            RescaleBiasRow(workUb, new_slope, slope, columnNum);
            slope = new_slope;
        } 
        AddBiasToScoreRow(scoreUb, scoreOffset + ri * rowStride, workUb, columnNum);
    }
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::S_V>(eventIdSToV);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_S>(eventIdVToS);
}

#endif

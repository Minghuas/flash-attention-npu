/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_RESCALE_O_HPP_T
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_RESCALE_O_HPP_T

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"

namespace Catlass::Epilogue::Block {

template <
    class ElementO_,
    class ElementOTmp_,
    class ElementS_,
    class TileCopy_,
    class OTmpSrcPos_ // the src TPosition of pv res, viable configurations: GM/L0C
>
class BlockEpilogue<
    EpilogueFARescaleO,
    ElementO_,
    ElementOTmp_,
    ElementS_,
    TileCopy_,
    OTmpSrcPos_>
{
public:
    using DispatchPolicy = EpilogueFARescaleO;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementO = ElementO_;
    using ElementOTmp = ElementOTmp_;
    using SMDtype = ElementS_;
    using TileCopy = TileCopy_;
    using OTmpSrcPos = OTmpSrcPos_;

    using CopyUbToGmO = typename TileCopy::CopyUbToGmO;

    static constexpr uint32_t UB_OTMP_BUF_STAGES = 2;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 32768;
    static constexpr uint32_t DM_UB_GLOBAL_ELEM_NUM = 64;
    static constexpr uint32_t RESCALE_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t RESCALE_COL_MAX_ELEM_NUM = 128;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t VECTOR_SIZE = 128;
    static constexpr uint32_t RESCALE_SKIP_SCRATCH_OFFSET = 250 * 1024;

    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource, bool enableRescaleSkip_ = false)
    {
        enableRescaleSkip = enableRescaleSkip_;
        constexpr uint32_t LO_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GO_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t LM_UB_TENSOR_OFFSET = 7 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = LM_UB_TENSOR_OFFSET + 64 * sizeof(float);
        constexpr uint32_t DM_UB_TENSOR_OFFSET = GM_UB_TENSOR_OFFSET + 64 * sizeof(float);
        constexpr uint32_t LL_UB_TENSOR_OFFSET = DM_UB_TENSOR_OFFSET + 3 * 64 * sizeof(float);
        constexpr uint32_t GL_UB_TENSOR_OFFSET = LL_UB_TENSOR_OFFSET +  64 * sizeof(float);

        for (uint32_t i = 0; i < UB_OTMP_BUF_STAGES; i++) {
            loUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                LO_UB_TENSOR_OFFSET + i * UB_UINT8_BLOCK_SIZE);
        }
        goUbTensor32 = resource.ubBuf.template GetBufferByByte<ElementOTmp>(GO_UB_TENSOR_OFFSET);
        goUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementO>(GO_UB_TENSOR_OFFSET);
        glUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        dmUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        redUbTensor = resource.ubBuf.template GetBufferByByte<float>(RESCALE_SKIP_SCRATCH_OFFSET);
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC set for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC wait for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <uint32_t ColBlocks, bool HeadDimAligned64, class TensorDst>
    __aicore__ inline
    void SubCoreCompute(TensorDst &gOTensorTlaTile,
                        uint32_t colStride,
                        uint32_t curTileMod,
                        uint32_t ubOTmpBufId,
                        bool isFirstKvSTile,
                        bool isLastKvSTile,
                        Arch::CrossCoreFlag pvReadyFlag,
                        uint32_t zeroRowCount,
                        uint32_t tailCols)
    {
        uint32_t rowNumCurSubCore = tla::get<0>(gOTensorTlaTile.shape());
        uint32_t colNumCurSubCore = tla::get<1>(gOTensorTlaTile.shape());

        __ubuf__ ElementOTmp *goUb = (__ubuf__ ElementOTmp *) goUbTensor32.GetPhyAddr();
        __ubuf__ ElementOTmp *loUb = (__ubuf__ ElementOTmp *) loUbTensor[ubOTmpBufId].GetPhyAddr();
        __ubuf__ ElementOTmp *glUb = ( __ubuf__ ElementOTmp *) glUbTensor32.GetPhyAddr();
        __ubuf__ ElementOTmp *dmUb =
            (__ubuf__ ElementOTmp *) dmUbTensor32[curTileMod * DM_UB_GLOBAL_ELEM_NUM].GetPhyAddr();
        
        WaitCrossCoreSync<4, PIPE_V>(pvReadyFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        if (isFirstKvSTile) {
            if (!isLastKvSTile) {
                uint32_t totalCopyElems = rowNumCurSubCore * colStride;
                AscendC::DataCopy(goUbTensor32, loUbTensor[ubOTmpBufId], totalCopyElems);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                DivFuncLastAndFirstGeneric<ElementOTmp, ColBlocks, HeadDimAligned64>(
                    goUb, loUb, glUb, rowNumCurSubCore, colNumCurSubCore, colStride, tailCols);
            }
        } else if (!isLastKvSTile) {
            bool skip = enableRescaleSkip &&
                DmAllOne(dmUbTensor32[curTileMod * DM_UB_GLOBAL_ELEM_NUM], rowNumCurSubCore);
            if (skip) {
                AddFunc<ElementOTmp, ColBlocks, HeadDimAligned64>(
                    goUb, loUb, rowNumCurSubCore, colNumCurSubCore, colStride, tailCols);
            } else {
                RescaleFuncGeneric<ElementOTmp, ColBlocks, HeadDimAligned64>(
                    goUb, loUb, dmUb, rowNumCurSubCore, colNumCurSubCore, colStride, tailCols);
            }
        } else {
            bool skip = enableRescaleSkip &&
                DmAllOne(dmUbTensor32[curTileMod * DM_UB_GLOBAL_ELEM_NUM], rowNumCurSubCore);
            if (skip) {
                AddDivFuncLastNotFirst<ElementOTmp, ColBlocks, HeadDimAligned64>(
                    goUb, loUb, glUb, rowNumCurSubCore, colNumCurSubCore, colStride, tailCols);
            } else {
                RescaleFuncLastNotFirstGeneric<ElementOTmp, ColBlocks, HeadDimAligned64>(
                    goUb, loUb, dmUb, glUb, rowNumCurSubCore, colNumCurSubCore, colStride, tailCols);
            }
        }
        if (zeroRowCount > 0) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(
                goUbTensor32,
                static_cast<ElementOTmp>(0),
                zeroRowCount * colStride);
        }
        // release lo buf
        SetCrossCoreSync<4, PIPE_V>(pvReadyFlag);
        if (isLastKvSTile) {
            AscendC::PipeBarrier<PIPE_V>();
            if (std::is_same<ElementO, bfloat16_t>::value) {
                AscendC::Cast(
                    goUbTensor16, goUbTensor32,
                    AscendC::RoundMode::CAST_RINT,
                    rowNumCurSubCore * colStride
                    );
            } else {
                AscendC::Cast(
                    goUbTensor16, goUbTensor32,
                    AscendC::RoundMode::CAST_NONE,
                    rowNumCurSubCore * colStride
                    );
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            auto ubOLayoutTla = tla::MakeLayout(
                tla::MakeShape(rowNumCurSubCore, colNumCurSubCore),
                tla::MakeStride(colStride, tla::Int<1>{})
            );
            auto ubOTensorTla = tla::MakeTensor(goUbTensor16, ubOLayoutTla, Arch::PositionUB{});
            copyUbToGmO(gOTensorTlaTile, ubOTensorTla);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
    }
    
    template <typename T, uint32_t ColBlocks, bool HeadDimAligned64 = true>
    __simd_vf__ inline void RescaleFuncGeneric(__ubuf__ T *goUb, __ubuf__ T *loUb, __ubuf__ T *dmUb,
                                               uint32_t row, uint32_t col, uint32_t rowStride,
                                               uint32_t tailCols)
    {
        using namespace AscendC::MicroAPI;
        const uint32_t VL_ELEM_NUM = 64;
        RegTensor<float> dm, go0, go1, go2, go3, lo0, lo1, lo2, lo3;
        RegTensor<float> mul0, mul1, mul2, mul3, out0, out1, out2, out3;
        MaskReg p0 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p1 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p2 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p3 = CreateMask<float, MaskPattern::ALL>();
        if constexpr (ColBlocks == 64) {
            p0 = UpdateMask<float>(tailCols);
        } else if constexpr (ColBlocks == 128) {
            p1 = UpdateMask<float>(tailCols);
        } else if constexpr (ColBlocks == 192) {
            p2 = UpdateMask<float>(tailCols);
        } else {
            p3 = UpdateMask<float>(tailCols);
        }

        for (uint32_t i = 0; i < row; ++i) {
            LoadAlign<T, LoadDist::DIST_BRC_B32>(dm, dmUb + i);
            __ubuf__ T *go;
            __ubuf__ T *lo;
            if constexpr (HeadDimAligned64) {
                go = goUb + i * ColBlocks;
                lo = loUb + i * ColBlocks;
            } else {
                go = goUb + i * rowStride;
                lo = loUb + i * rowStride;
            }
            LoadAlign<T, LoadDist::DIST_NORM>(go0, go);
            LoadAlign<T, LoadDist::DIST_NORM>(lo0, lo);
            if constexpr (ColBlocks >= 128) {
                LoadAlign<T, LoadDist::DIST_NORM>(go1, go + VL_ELEM_NUM);
                LoadAlign<T, LoadDist::DIST_NORM>(lo1, lo + VL_ELEM_NUM);
            }
            if constexpr (ColBlocks >= 192) {
                LoadAlign<T, LoadDist::DIST_NORM>(go2, go + VL_ELEM_NUM * 2);
                LoadAlign<T, LoadDist::DIST_NORM>(lo2, lo + VL_ELEM_NUM * 2);
            }
            if constexpr (ColBlocks >= 256) {
                LoadAlign<T, LoadDist::DIST_NORM>(go3, go + VL_ELEM_NUM * 3);
                LoadAlign<T, LoadDist::DIST_NORM>(lo3, lo + VL_ELEM_NUM * 3);
            }

            Mul(mul0, go0, dm, p0);
            if constexpr (ColBlocks >= 128) {
                Mul(mul1, go1, dm, p1);
            }
            if constexpr (ColBlocks >= 192) {
                Mul(mul2, go2, dm, p2);
            }
            if constexpr (ColBlocks >= 256) {
                Mul(mul3, go3, dm, p3);
            }
            Add(out0, mul0, lo0, p0);
            if constexpr (ColBlocks >= 128) {
                Add(out1, mul1, lo1, p1);
            }
            if constexpr (ColBlocks >= 192) {
                Add(out2, mul2, lo2, p2);
            }
            if constexpr (ColBlocks >= 256) {
                Add(out3, mul3, lo3, p3);
            }

            StoreAlign<T, StoreDist::DIST_NORM_B32>(go, out0, p0);
            if constexpr (ColBlocks >= 128) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM, out1, p1);
            }
            if constexpr (ColBlocks >= 192) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM * 2, out2, p2);
            }
            if constexpr (ColBlocks >= 256) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM * 3, out3, p3);
            }
        }
    }

    template <typename T, uint32_t ColBlocks, bool HeadDimAligned64 = true>
    __simd_vf__ inline void RescaleFuncLastNotFirstGeneric(__ubuf__ T *goUb, __ubuf__ T *loUb,
                                                           __ubuf__ T *dmUb, __ubuf__ T *glUb,
                                                           uint32_t row, uint32_t col, uint32_t rowStride,
                                                           uint32_t tailCols)
    {
        using namespace AscendC::MicroAPI;
        const uint32_t VL_ELEM_NUM = 64;
        RegTensor<float> dm, gl, go0, go1, go2, go3, lo0, lo1, lo2, lo3;
        RegTensor<float> mul0, mul1, mul2, mul3, sum0, sum1, sum2, sum3;
        RegTensor<float> out0, out1, out2, out3;
        MaskReg p0 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p1 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p2 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p3 = CreateMask<float, MaskPattern::ALL>();
        if constexpr (ColBlocks == 64) {
            p0 = UpdateMask<float>(tailCols);
        } else if constexpr (ColBlocks == 128) {
            p1 = UpdateMask<float>(tailCols);
        } else if constexpr (ColBlocks == 192) {
            p2 = UpdateMask<float>(tailCols);
        } else {
            p3 = UpdateMask<float>(tailCols);
        }

        for (uint32_t i = 0; i < row; ++i) {
            LoadAlign<T, LoadDist::DIST_BRC_B32>(dm, dmUb + i);
            LoadAlign<T, LoadDist::DIST_BRC_B32>(gl, glUb + i);
            __ubuf__ T *go;
            __ubuf__ T *lo;
            if constexpr (HeadDimAligned64) {
                go = goUb + i * ColBlocks;
                lo = loUb + i * ColBlocks;
            } else {
                go = goUb + i * rowStride;
                lo = loUb + i * rowStride;
            }
            LoadAlign<T, LoadDist::DIST_NORM>(go0, go);
            LoadAlign<T, LoadDist::DIST_NORM>(lo0, lo);
            if constexpr (ColBlocks >= 128) {
                LoadAlign<T, LoadDist::DIST_NORM>(go1, go + VL_ELEM_NUM);
                LoadAlign<T, LoadDist::DIST_NORM>(lo1, lo + VL_ELEM_NUM);
            }
            if constexpr (ColBlocks >= 192) {
                LoadAlign<T, LoadDist::DIST_NORM>(go2, go + VL_ELEM_NUM * 2);
                LoadAlign<T, LoadDist::DIST_NORM>(lo2, lo + VL_ELEM_NUM * 2);
            }
            if constexpr (ColBlocks >= 256) {
                LoadAlign<T, LoadDist::DIST_NORM>(go3, go + VL_ELEM_NUM * 3);
                LoadAlign<T, LoadDist::DIST_NORM>(lo3, lo + VL_ELEM_NUM * 3);
            }

            Mul(mul0, go0, dm, p0);
            if constexpr (ColBlocks >= 128) {
                Mul(mul1, go1, dm, p1);
            }
            if constexpr (ColBlocks >= 192) {
                Mul(mul2, go2, dm, p2);
            }
            if constexpr (ColBlocks >= 256) {
                Mul(mul3, go3, dm, p3);
            }
            Add(sum0, mul0, lo0, p0);
            if constexpr (ColBlocks >= 128) {
                Add(sum1, mul1, lo1, p1);
            }
            if constexpr (ColBlocks >= 192) {
                Add(sum2, mul2, lo2, p2);
            }
            if constexpr (ColBlocks >= 256) {
                Add(sum3, mul3, lo3, p3);
            }
            Div(out0, sum0, gl, p0);
            if constexpr (ColBlocks >= 128) {
                Div(out1, sum1, gl, p1);
            }
            if constexpr (ColBlocks >= 192) {
                Div(out2, sum2, gl, p2);
            }
            if constexpr (ColBlocks >= 256) {
                Div(out3, sum3, gl, p3);
            }

            StoreAlign<T, StoreDist::DIST_NORM_B32>(go, out0, p0);
            if constexpr (ColBlocks >= 128) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM, out1, p1);
            }
            if constexpr (ColBlocks >= 192) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM * 2, out2, p2);
            }
            if constexpr (ColBlocks >= 256) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM * 3, out3, p3);
            }
        }
    }

    template <typename T, uint32_t ColBlocks, bool HeadDimAligned64 = true>
    __simd_vf__ inline void DivFuncLastAndFirstGeneric(__ubuf__ T *goUb, __ubuf__ T *loUb,
                                                       __ubuf__ T *glUb, uint32_t row, uint32_t col,
                                                       uint32_t rowStride,
                                                       uint32_t tailCols)
    {
        using namespace AscendC::MicroAPI;
        const uint32_t VL_ELEM_NUM = 64;
        RegTensor<float> gl, in0, in1, in2, in3, out0, out1, out2, out3;
        MaskReg p0 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p1 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p2 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p3 = CreateMask<float, MaskPattern::ALL>();
        if constexpr (ColBlocks == 64) {
            p0 = UpdateMask<float>(tailCols);
        } else if constexpr (ColBlocks == 128) {
            p1 = UpdateMask<float>(tailCols);
        } else if constexpr (ColBlocks == 192) {
            p2 = UpdateMask<float>(tailCols);
        } else {
            p3 = UpdateMask<float>(tailCols);
        }

        for (uint32_t i = 0; i < row; ++i) {
            LoadAlign<T, LoadDist::DIST_BRC_B32>(gl, glUb + i);
            __ubuf__ T *go;
            __ubuf__ T *lo;
            if constexpr (HeadDimAligned64) {
                go = goUb + i * ColBlocks;
                lo = loUb + i * ColBlocks;
            } else {
                go = goUb + i * rowStride;
                lo = loUb + i * rowStride;
            }
            LoadAlign<T, LoadDist::DIST_NORM>(in0, lo);
            if constexpr (ColBlocks >= 128) {
                LoadAlign<T, LoadDist::DIST_NORM>(in1, lo + VL_ELEM_NUM);
            }
            if constexpr (ColBlocks >= 192) {
                LoadAlign<T, LoadDist::DIST_NORM>(in2, lo + VL_ELEM_NUM * 2);
            }
            if constexpr (ColBlocks >= 256) {
                LoadAlign<T, LoadDist::DIST_NORM>(in3, lo + VL_ELEM_NUM * 3);
            }

            Div(out0, in0, gl, p0);
            if constexpr (ColBlocks >= 128) {
                Div(out1, in1, gl, p1);
            }
            if constexpr (ColBlocks >= 192) {
                Div(out2, in2, gl, p2);
            }
            if constexpr (ColBlocks >= 256) {
                Div(out3, in3, gl, p3);
            }

            StoreAlign<T, StoreDist::DIST_NORM_B32>(go, out0, p0);
            if constexpr (ColBlocks >= 128) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM, out1, p1);
            }
            if constexpr (ColBlocks >= 192) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM * 2, out2, p2);
            }
            if constexpr (ColBlocks >= 256) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM * 3, out3, p3);
            }
        }
    }

    __aicore__ inline
    void SetMask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = len % FLOAT_VECTOR_SIZE;
        for (int64_t i = 0; i < temp; i++) {
            mask |= one << i;
        }
        if (len == VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else if (len >= FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
        } else {
            AscendC::SetVectorMask<int8_t>(0x0, mask);
        }
    }

    __aicore__ inline
    bool DmAllOne(AscendC::LocalTensor<float> dm, uint32_t row)
    {
        SetMask((int32_t)row);
        AscendC::WholeReduceMin<float, false>(
            redUbTensor, dm, (int32_t)0, 1, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID7);
        float minDm = redUbTensor.GetValue(0);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID7);
        // Restore the full vector mask so later legacy-API vector ops are unaffected.
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        return minDm >= 1.0f;
    }

    // Skip variant of RescaleFunc: dm == 1 for all rows, so go = go + lo.
    template <typename T, uint32_t ColBlocks, bool HeadDimAligned64 = true>
    __simd_vf__ inline void AddFunc(__ubuf__ T *goUb, __ubuf__ T *loUb,
                                    uint32_t row, uint32_t col, uint32_t rowStride,
                                    uint32_t tailCols)
    {
        using namespace AscendC::MicroAPI;
        const uint32_t VL_ELEM_NUM = 64;
        RegTensor<float> go0, go1, go2, go3, lo0, lo1, lo2, lo3;
        RegTensor<float> out0, out1, out2, out3;
        MaskReg p0 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p1 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p2 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p3 = CreateMask<float, MaskPattern::ALL>();
        if constexpr (ColBlocks == 64) {
            p0 = UpdateMask<float>(tailCols);
        } else if constexpr (ColBlocks == 128) {
            p1 = UpdateMask<float>(tailCols);
        } else if constexpr (ColBlocks == 192) {
            p2 = UpdateMask<float>(tailCols);
        } else {
            p3 = UpdateMask<float>(tailCols);
        }

        for (uint32_t i = 0; i < row; ++i) {
            __ubuf__ T *go;
            __ubuf__ T *lo;
            if constexpr (HeadDimAligned64) {
                go = goUb + i * ColBlocks;
                lo = loUb + i * ColBlocks;
            } else {
                go = goUb + i * rowStride;
                lo = loUb + i * rowStride;
            }
            LoadAlign<T, LoadDist::DIST_NORM>(go0, go);
            LoadAlign<T, LoadDist::DIST_NORM>(lo0, lo);
            if constexpr (ColBlocks >= 128) {
                LoadAlign<T, LoadDist::DIST_NORM>(go1, go + VL_ELEM_NUM);
                LoadAlign<T, LoadDist::DIST_NORM>(lo1, lo + VL_ELEM_NUM);
            }
            if constexpr (ColBlocks >= 192) {
                LoadAlign<T, LoadDist::DIST_NORM>(go2, go + VL_ELEM_NUM * 2);
                LoadAlign<T, LoadDist::DIST_NORM>(lo2, lo + VL_ELEM_NUM * 2);
            }
            if constexpr (ColBlocks >= 256) {
                LoadAlign<T, LoadDist::DIST_NORM>(go3, go + VL_ELEM_NUM * 3);
                LoadAlign<T, LoadDist::DIST_NORM>(lo3, lo + VL_ELEM_NUM * 3);
            }

            Add(out0, go0, lo0, p0);
            if constexpr (ColBlocks >= 128) {
                Add(out1, go1, lo1, p1);
            }
            if constexpr (ColBlocks >= 192) {
                Add(out2, go2, lo2, p2);
            }
            if constexpr (ColBlocks >= 256) {
                Add(out3, go3, lo3, p3);
            }

            StoreAlign<T, StoreDist::DIST_NORM_B32>(go, out0, p0);
            if constexpr (ColBlocks >= 128) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM, out1, p1);
            }
            if constexpr (ColBlocks >= 192) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM * 2, out2, p2);
            }
            if constexpr (ColBlocks >= 256) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM * 3, out3, p3);
            }
        }
    }

    // Skip variant of RescaleFuncLastNotFirst: go = (go + lo) / gl.
    template <typename T, uint32_t ColBlocks, bool HeadDimAligned64 = true>
    __simd_vf__ inline void AddDivFuncLastNotFirst(__ubuf__ T *goUb, __ubuf__ T *loUb,
                                                   __ubuf__ T *glUb, uint32_t row,
                                                   uint32_t col, uint32_t rowStride,
                                                   uint32_t tailCols)
    {
        using namespace AscendC::MicroAPI;
        const uint32_t VL_ELEM_NUM = 64;
        RegTensor<float> gl, go0, go1, go2, go3, lo0, lo1, lo2, lo3;
        RegTensor<float> sum0, sum1, sum2, sum3, out0, out1, out2, out3;
        MaskReg p0 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p1 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p2 = CreateMask<float, MaskPattern::ALL>();
        MaskReg p3 = CreateMask<float, MaskPattern::ALL>();
        if constexpr (ColBlocks == 64) {
            p0 = UpdateMask<float>(tailCols);
        } else if constexpr (ColBlocks == 128) {
            p1 = UpdateMask<float>(tailCols);
        } else if constexpr (ColBlocks == 192) {
            p2 = UpdateMask<float>(tailCols);
        } else {
            p3 = UpdateMask<float>(tailCols);
        }

        for (uint32_t i = 0; i < row; ++i) {
            LoadAlign<T, LoadDist::DIST_BRC_B32>(gl, glUb + i);
            __ubuf__ T *go;
            __ubuf__ T *lo;
            if constexpr (HeadDimAligned64) {
                go = goUb + i * ColBlocks;
                lo = loUb + i * ColBlocks;
            } else {
                go = goUb + i * rowStride;
                lo = loUb + i * rowStride;
            }
            LoadAlign<T, LoadDist::DIST_NORM>(go0, go);
            LoadAlign<T, LoadDist::DIST_NORM>(lo0, lo);
            if constexpr (ColBlocks >= 128) {
                LoadAlign<T, LoadDist::DIST_NORM>(go1, go + VL_ELEM_NUM);
                LoadAlign<T, LoadDist::DIST_NORM>(lo1, lo + VL_ELEM_NUM);
            }
            if constexpr (ColBlocks >= 192) {
                LoadAlign<T, LoadDist::DIST_NORM>(go2, go + VL_ELEM_NUM * 2);
                LoadAlign<T, LoadDist::DIST_NORM>(lo2, lo + VL_ELEM_NUM * 2);
            }
            if constexpr (ColBlocks >= 256) {
                LoadAlign<T, LoadDist::DIST_NORM>(go3, go + VL_ELEM_NUM * 3);
                LoadAlign<T, LoadDist::DIST_NORM>(lo3, lo + VL_ELEM_NUM * 3);
            }

            Add(sum0, go0, lo0, p0);
            if constexpr (ColBlocks >= 128) {
                Add(sum1, go1, lo1, p1);
            }
            if constexpr (ColBlocks >= 192) {
                Add(sum2, go2, lo2, p2);
            }
            if constexpr (ColBlocks >= 256) {
                Add(sum3, go3, lo3, p3);
            }

            Div(out0, sum0, gl, p0);
            if constexpr (ColBlocks >= 128) {
                Div(out1, sum1, gl, p1);
            }
            if constexpr (ColBlocks >= 192) {
                Div(out2, sum2, gl, p2);
            }
            if constexpr (ColBlocks >= 256) {
                Div(out3, sum3, gl, p3);
            }

            StoreAlign<T, StoreDist::DIST_NORM_B32>(go, out0, p0);
            if constexpr (ColBlocks >= 128) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM, out1, p1);
            }
            if constexpr (ColBlocks >= 192) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM * 2, out2, p2);
            }
            if constexpr (ColBlocks >= 256) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(go + VL_ELEM_NUM * 3, out3, p3);
            }
        }
    }

    template <class TensorDst>
    __aicore__ inline
    void operator()(TensorDst &gOTensor,
                    GemmCoord actualOriShape,
                    uint32_t curTileMod,
                    uint32_t gatheredKvSTileIdx,
                    bool isFirstKvSTile,
                    bool isLastKvSTile,
                    Arch::CrossCoreFlag pvReadyFlag,
                    bool isDN,
                    uint32_t fullyMaskedRows)
    {
        uint32_t rowNumOri = actualOriShape[0];
        uint32_t colNumOri = actualOriShape[1];
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t rowNumOriAligned = isDN ? RoundUp(rowNumOri, 32) : RoundUp(rowNumOri, 8);
        uint32_t colNumOriAligned16 = RoundUp(colNumOri, 16);

        uint32_t rowNumSplit = rowNumOriAligned / subBlockNum;
        rowNumSplit = (rowNumOri < rowNumSplit) ? rowNumOri : rowNumSplit;
        uint32_t rowNumCurSubCore = (subBlockIdx == 0) ? rowNumSplit : (rowNumOri - rowNumSplit);
        uint32_t rowOffsetCurSubCore = rowNumSplit * subBlockIdx;
        uint32_t colNumCurSubCore = colNumOri;
        uint32_t colStrideCurSubCore = colNumOriAligned16;
        uint32_t zeroRowCount = 0;
        if (rowOffsetCurSubCore < fullyMaskedRows) {
            uint32_t remainingRows = fullyMaskedRows - rowOffsetCurSubCore;
            zeroRowCount = remainingRows < rowNumCurSubCore ? remainingRows : rowNumCurSubCore;
        }

        auto gOTensorTlaTile = GetTile(gOTensor,
            tla::MakeCoord(rowOffsetCurSubCore, 0), tla::MakeShape(rowNumCurSubCore, colNumCurSubCore));
        uint32_t ubOTmpBufId = gatheredKvSTileIdx % UB_OTMP_BUF_STAGES;
        uint32_t vlElemNum = AscendC::GetVecLen() / sizeof(ElementOTmp);
        bool headdimAligned64 = (colNumOri % 64 == 0);
        if (rowNumCurSubCore > 0) {
            if (colNumCurSubCore <= vlElemNum) {
                if (headdimAligned64) {
                    SubCoreCompute<64, true>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore);
                } else {
                    SubCoreCompute<64, false>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore);
                }
            } else if (colNumCurSubCore <= 2 * vlElemNum) {
                if (headdimAligned64) {
                    SubCoreCompute<128, true>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore - vlElemNum);
                } else {
                    SubCoreCompute<128, false>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore - vlElemNum);
                }
            } else if (colNumCurSubCore <= 3 * vlElemNum) {
                if (headdimAligned64) {
                    SubCoreCompute<192, true>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore - 2 * vlElemNum);
                } else {
                    SubCoreCompute<192, false>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore - 2 * vlElemNum);
                }
            } else {
                if (headdimAligned64) {
                    SubCoreCompute<256, true>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore - 3 * vlElemNum);
                } else {
                    SubCoreCompute<256, false>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore - 3 * vlElemNum);
                }
            }
        } else {
            Arch::CrossCoreWaitFlag<4, PIPE_V>(pvReadyFlag);
            Arch::CrossCoreSetFlag<4, PIPE_V>(pvReadyFlag);
        }
    }
private:
    AscendC::LocalTensor<ElementOTmp> loUbTensor[UB_OTMP_BUF_STAGES];
    AscendC::LocalTensor<SMDtype> dmUbTensor16;
    AscendC::LocalTensor<SMDtype> glUbTensor16;
    AscendC::LocalTensor<float> dmUbTensor32;
    AscendC::LocalTensor<float> glUbTensor32;
    AscendC::LocalTensor<float> redUbTensor;
    bool enableRescaleSkip{false};
    AscendC::LocalTensor<ElementO> goUbTensor16;
    AscendC::LocalTensor<ElementOTmp> goUbTensor32;

    CopyUbToGmO copyUbToGmO;
};
}
#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_RESCALE_O_LOCAL_HPP

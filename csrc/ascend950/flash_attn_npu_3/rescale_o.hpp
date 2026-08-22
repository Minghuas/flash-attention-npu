/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_RESCALE_O_HPP_T
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_RESCALE_O_HPP_T

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include <limits>
#include <type_traits>
#include "kernel_common.hpp"
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
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;

    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {
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
        lmUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(LM_UB_TENSOR_OFFSET);
        gmUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        dmUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
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
                        uint32_t tailCols,
                        bool skipOutput = false)
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
                DivFuncLastAndFirst<ElementOTmp, ColBlocks, HeadDimAligned64>(
                    goUb, loUb, glUb, rowNumCurSubCore, colNumCurSubCore, colStride, tailCols);
            }
        } else if (!isLastKvSTile) {
            RescaleFunc<ElementOTmp, ColBlocks, HeadDimAligned64>(
                goUb, loUb, dmUb, rowNumCurSubCore, colNumCurSubCore, colStride, tailCols);
        } else {
            RescaleFuncLastNotFirst<ElementOTmp, ColBlocks, HeadDimAligned64>(
                goUb, loUb, dmUb, glUb, rowNumCurSubCore, colNumCurSubCore, colStride, tailCols);
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
            if (!skipOutput) {
                copyUbToGmO(gOTensorTlaTile, ubOTensorTla);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
    }

    template <class TensorDst>
    __aicore__ inline
    void ScatterGroupedOutput(TensorDst &gOTensor, uint32_t qSBlockSize,
                              uint32_t logicalRowStart, uint32_t rowCount,
                              uint32_t embedV, uint32_t outputStride,
                              uint32_t colStride, uint32_t fullyMaskedRowsPerHead)
    {
        uint32_t firstHead = logicalRowStart / qSBlockSize;
        uint32_t headCount = rowCount / qSBlockSize;
        auto gO = gOTensor.data();
        for (uint32_t headLocal = 0; headLocal < headCount; ++headLocal) {
            if (fullyMaskedRowsPerHead != 0U) {
                AscendC::Duplicate(goUbTensor16[headLocal * qSBlockSize * colStride],
                    static_cast<ElementO>(0), fullyMaskedRowsPerHead * colStride);
            }
            AscendC::DataCopyPad(
                gO[(firstHead + headLocal) * embedV],
                goUbTensor16[headLocal * qSBlockSize * colStride],
                AscendC::DataCopyExtParams(qSBlockSize, embedV * sizeof(ElementO), 0,
                    (outputStride - embedV) * sizeof(ElementO), 0));
        }
    }

    // Keep LSE behind the final O GM write.  goUbTensor32 is free after the
    // EVENT_ID4 wait, so it is a dedicated LSE expansion scratch and never
    // aliases OnlineSoftmax's maskUb on the following task.
    __aicore__ inline
    void WriteGroupedLse(AscendC::GlobalTensor<float> gLse,
                         uint32_t qSBlockSize, uint32_t qNBlockSize,
                         uint32_t qHeads, uint32_t fullyMaskedRowsPerHead)
    {
        auto partition = GetFAIGroupedRowPartition(qSBlockSize, qNBlockSize, 8U);
        uint32_t rowCount = partition.validRows;
        if (rowCount == 0U) {
            return;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Ln<float, false>(lmUbTensor32, glUbTensor32, (uint64_t)0,
            CeilDiv(rowCount, FLOAT_VECTOR_SIZE), AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float, false>(lmUbTensor32, lmUbTensor32, gmUbTensor32, (uint64_t)0,
            CeilDiv(rowCount, FLOAT_VECTOR_SIZE), AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(goUbTensor32.template ReinterpretCast<uint32_t>(),
            lmUbTensor32.template ReinterpretCast<uint32_t>(),
            CeilDiv(rowCount, FLOAT_BLOCK_SIZE), AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        // Follow Ascend910's InvalidLineLSEProcess convention: Brcb expands
        // every scalar LSE to one 32-byte block.  Mark invalid rows in that
        // block-expanded view so every vector destination is naturally aligned.
        if (fullyMaskedRowsPerHead != 0U) {
            if (qNBlockSize == 1U) {
                uint32_t firstLocalS = partition.logicalRowStart % qSBlockSize;
                if (firstLocalS < fullyMaskedRowsPerHead) {
                    uint32_t maskedRows = fullyMaskedRowsPerHead - firstLocalS;
                    maskedRows = maskedRows < rowCount ? maskedRows : rowCount;
                    AscendC::Duplicate(goUbTensor32, std::numeric_limits<float>::infinity(),
                        maskedRows * FLOAT_BLOCK_SIZE);
                }
            } else {
                uint32_t headCount = rowCount / qSBlockSize;
                uint32_t maskedRows = fullyMaskedRowsPerHead < qSBlockSize ?
                    fullyMaskedRowsPerHead : qSBlockSize;
                for (uint32_t headLocal = 0; headLocal < headCount; ++headLocal) {
                    AscendC::Duplicate(goUbTensor32[headLocal * qSBlockSize * FLOAT_BLOCK_SIZE],
                        std::numeric_limits<float>::infinity(), maskedRows * FLOAT_BLOCK_SIZE);
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
        constexpr uint32_t LSE_EXPANDED_SRC_GAP =
            (FLOAT_BLOCK_SIZE - 1U) * sizeof(float);
        if (qNBlockSize == 1U) {
            AscendC::DataCopyPad(gLse[partition.logicalRowStart * qHeads], goUbTensor32,
                AscendC::DataCopyExtParams(rowCount, sizeof(float), LSE_EXPANDED_SRC_GAP,
                    (qHeads - 1U) * sizeof(float), 0));
        } else {
            uint32_t firstHead = partition.logicalRowStart / qSBlockSize;
            uint32_t headCount = rowCount / qSBlockSize;
            for (uint32_t headLocal = 0; headLocal < headCount; ++headLocal) {
                AscendC::DataCopyPad(gLse[firstHead + headLocal],
                    goUbTensor32[headLocal * qSBlockSize * FLOAT_BLOCK_SIZE],
                    AscendC::DataCopyExtParams(qSBlockSize, sizeof(float), LSE_EXPANDED_SRC_GAP,
                        (qHeads - 1U) * sizeof(float), 0));
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
    }
    
    template <typename T, uint32_t ColBlocks, bool HeadDimAligned64 = true>
    __simd_vf__ inline void RescaleFunc(__ubuf__ T *goUb, __ubuf__ T *loUb, __ubuf__ T *dmUb,
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
    __simd_vf__ inline void RescaleFuncLastNotFirst(__ubuf__ T *goUb, __ubuf__ T *loUb,
                                        __ubuf__ T *dmUb, __ubuf__ T *glUb,
                                        uint32_t row, uint32_t col, uint32_t rowStride,
                                        uint32_t tailCols)
    {
        using namespace AscendC::MicroAPI;
        const uint32_t VL_ELEM_NUM = 64;

        RegTensor<float> dmVreg;
        RegTensor<float> glVreg;
        RegTensor<float> goPreVreg0;
        RegTensor<float> goPreVreg1;
        RegTensor<float> goPreVreg2;
        RegTensor<float> goPreVreg3;
        RegTensor<float> loVreg0;
        RegTensor<float> loVreg1;
        RegTensor<float> loVreg2;
        RegTensor<float> loVreg3;
        RegTensor<float> mulVreg0;
        RegTensor<float> mulVreg1;
        RegTensor<float> mulVreg2;
        RegTensor<float> mulVreg3;
        RegTensor<float> goCurVreg0;
        RegTensor<float> goCurVreg1;
        RegTensor<float> goCurVreg2;
        RegTensor<float> goCurVreg3;
        RegTensor<float> divVreg0;
        RegTensor<float> divVreg1;
        RegTensor<float> divVreg2;
        RegTensor<float> divVreg3;

        MaskReg preg0 = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg1 = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg2 = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg3 = CreateMask<float, MaskPattern::ALL>();
        if constexpr (ColBlocks == 64) {
            preg0 = UpdateMask<float>(tailCols);
        } else if constexpr (ColBlocks == 128) {
            preg1 = UpdateMask<float>(tailCols);
        } else if constexpr (ColBlocks == 192) {
            preg2 = UpdateMask<float>(tailCols);
        } else {
            preg3 = UpdateMask<float>(tailCols);
        }

        for (uint32_t i = 0; i < row; ++i) {
            LoadAlign<T, LoadDist::DIST_BRC_B32>(dmVreg, dmUb + i);
            LoadAlign<T, LoadDist::DIST_BRC_B32>(glVreg, glUb + i);

            __ubuf__ T *goRow;
            __ubuf__ T *loRow;
            if constexpr (HeadDimAligned64) {
                goRow = goUb + i * ColBlocks;
                loRow = loUb + i * ColBlocks;
            } else {
                goRow = goUb + i * rowStride;
                loRow = loUb + i * rowStride;
            }
            LoadAlign<T, LoadDist::DIST_NORM>(goPreVreg0, goRow);
            LoadAlign<T, LoadDist::DIST_NORM>(loVreg0, loRow);
            if constexpr (ColBlocks >= 128) {
                LoadAlign<T, LoadDist::DIST_NORM>(goPreVreg1, goRow + VL_ELEM_NUM);
                LoadAlign<T, LoadDist::DIST_NORM>(loVreg1, loRow + VL_ELEM_NUM);
            }
            if constexpr (ColBlocks >= 192) {
                LoadAlign<T, LoadDist::DIST_NORM>(goPreVreg2, goRow + VL_ELEM_NUM * 2);
                LoadAlign<T, LoadDist::DIST_NORM>(loVreg2, loRow + VL_ELEM_NUM * 2);
            }
            if constexpr (ColBlocks >= 256) {
                LoadAlign<T, LoadDist::DIST_NORM>(goPreVreg3, goRow + VL_ELEM_NUM * 3);
                LoadAlign<T, LoadDist::DIST_NORM>(loVreg3, loRow + VL_ELEM_NUM * 3);
            }

            Mul(mulVreg0, goPreVreg0, dmVreg, preg0);
            if constexpr (ColBlocks >= 128) {
                Mul(mulVreg1, goPreVreg1, dmVreg, preg1);
            }
            if constexpr (ColBlocks >= 192) {
                Mul(mulVreg2, goPreVreg2, dmVreg, preg2);
            }
            if constexpr (ColBlocks >= 256) {
                Mul(mulVreg3, goPreVreg3, dmVreg, preg3);
            }

            Add(goCurVreg0, mulVreg0, loVreg0, preg0);
            if constexpr (ColBlocks >= 128) {
                Add(goCurVreg1, mulVreg1, loVreg1, preg1);
            }
            if constexpr (ColBlocks >= 192) {
                Add(goCurVreg2, mulVreg2, loVreg2, preg2);
            }
            if constexpr (ColBlocks >= 256) {
                Add(goCurVreg3, mulVreg3, loVreg3, preg3);
            }

            Div(divVreg0, goCurVreg0, glVreg, preg0);
            if constexpr (ColBlocks >= 128) {
                Div(divVreg1, goCurVreg1, glVreg, preg1);
            }
            if constexpr (ColBlocks >= 192) {
                Div(divVreg2, goCurVreg2, glVreg, preg2);
            }
            if constexpr (ColBlocks >= 256) {
                Div(divVreg3, goCurVreg3, glVreg, preg3);
            }

            StoreAlign<T, StoreDist::DIST_NORM_B32>(goRow, divVreg0, preg0);
            if constexpr (ColBlocks >= 128) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(goRow + VL_ELEM_NUM, divVreg1, preg1);
            }
            if constexpr (ColBlocks >= 192) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(goRow + VL_ELEM_NUM * 2, divVreg2, preg2);
            }
            if constexpr (ColBlocks >= 256) {
                StoreAlign<T, StoreDist::DIST_NORM_B32>(goRow + VL_ELEM_NUM * 3, divVreg3, preg3);
            }
        }
    }

    template <typename T, uint32_t ColBlocks, bool HeadDimAligned64 = true>
    __simd_vf__ inline void DivFuncLastAndFirst(__ubuf__ T *goUb, __ubuf__ T *loUb, __ubuf__ T *glUb,
                                                uint32_t row, uint32_t col, uint32_t rowStride,
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

    template <class TensorDst>
    __aicore__ inline
    void operator()(TensorDst &gOTensor,
                    AscendC::GlobalTensor<float> gLse,
                    GemmCoord actualOriShape,
                    uint32_t curTileMod,
                    uint32_t gatheredKvSTileIdx,
                    bool isFirstKvSTile,
                    bool isLastKvSTile,
                    Arch::CrossCoreFlag pvReadyFlag,
                    bool isDN,
                    uint32_t fullyMaskedRowsPerHead,
                    uint32_t qSBlockSize,
                    uint32_t qNBlockSize,
                    uint32_t outputStride)
    {
        uint32_t rowNumOri = actualOriShape[0];
        uint32_t colNumOri = actualOriShape[1];
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t colNumOriAligned16 = RoundUp(colNumOri, 16);
        auto partition = GetFAIGroupedRowPartition(qSBlockSize, qNBlockSize, isDN ? 32U : 8U);
        uint32_t rowNumCurSubCore = partition.validRows;
        uint32_t rowOffsetCurSubCore = partition.logicalRowStart;
        uint32_t colNumCurSubCore = colNumOri;
        uint32_t colStrideCurSubCore = colNumOriAligned16;
        uint32_t zeroRowCount = 0;
        if (qNBlockSize == 1U && rowOffsetCurSubCore < fullyMaskedRowsPerHead) {
            uint32_t remainingRows = fullyMaskedRowsPerHead - rowOffsetCurSubCore;
            zeroRowCount = remainingRows < rowNumCurSubCore ? remainingRows : rowNumCurSubCore;
        }

        // For qN>1 the final GM addresses are not affine in logical M.  The
        // tile below supplies only shape to SubCoreCompute; final output uses
        // ScatterGroupedOutput after the per-AIV computation finishes.
        auto gOTensorTlaTile = GetTile(gOTensor,
            tla::MakeCoord(qNBlockSize == 1U ? rowOffsetCurSubCore : 0U, 0),
            tla::MakeShape(rowNumCurSubCore, colNumCurSubCore));
        uint32_t ubOTmpBufId = gatheredKvSTileIdx % UB_OTMP_BUF_STAGES;
        uint32_t vlElemNum = AscendC::GetVecLen() / sizeof(ElementOTmp);
        bool headdimAligned64 = (colNumOri % 64 == 0);
        if (rowNumCurSubCore > 0) {
            if (colNumCurSubCore <= vlElemNum) {
                if (headdimAligned64) {
                    SubCoreCompute<64, true>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore, qNBlockSize > 1U);
                } else {
                    SubCoreCompute<64, false>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore, qNBlockSize > 1U);
                }
            } else if (colNumCurSubCore <= 2 * vlElemNum) {
                if (headdimAligned64) {
                    SubCoreCompute<128, true>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore - vlElemNum, qNBlockSize > 1U);
                } else {
                    SubCoreCompute<128, false>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore - vlElemNum, qNBlockSize > 1U);
                }
            } else if (colNumCurSubCore <= 3 * vlElemNum) {
                if (headdimAligned64) {
                    SubCoreCompute<192, true>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore - 2 * vlElemNum, qNBlockSize > 1U);
                } else {
                    SubCoreCompute<192, false>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore - 2 * vlElemNum, qNBlockSize > 1U);
                }
            } else {
                if (headdimAligned64) {
                    SubCoreCompute<256, true>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore - 3 * vlElemNum, qNBlockSize > 1U);
                } else {
                    SubCoreCompute<256, false>(gOTensorTlaTile, colStrideCurSubCore, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, pvReadyFlag, zeroRowCount, colNumCurSubCore - 3 * vlElemNum, qNBlockSize > 1U);
                }
            }
        } else {
            Arch::CrossCoreWaitFlag<4, PIPE_V>(pvReadyFlag);
            Arch::CrossCoreSetFlag<4, PIPE_V>(pvReadyFlag);
        }
        if (qNBlockSize > 1U && rowNumCurSubCore > 0U && isLastKvSTile) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            ScatterGroupedOutput(gOTensor, qSBlockSize, rowOffsetCurSubCore,
                rowNumCurSubCore, colNumOri, outputStride, colStrideCurSubCore,
                fullyMaskedRowsPerHead);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        }
        if constexpr (std::is_same_v<SMDtype, float>) {
            if (rowNumCurSubCore > 0U && isLastKvSTile) {
                WriteGroupedLse(gLse, qSBlockSize, qNBlockSize, outputStride / colNumOri,
                    fullyMaskedRowsPerHead);
            }
        }
    }
private:
    AscendC::LocalTensor<ElementOTmp> loUbTensor[UB_OTMP_BUF_STAGES];
    AscendC::LocalTensor<SMDtype> dmUbTensor16;
    AscendC::LocalTensor<SMDtype> glUbTensor16;
    AscendC::LocalTensor<float> dmUbTensor32;
    AscendC::LocalTensor<float> glUbTensor32;
    AscendC::LocalTensor<float> lmUbTensor32;
    AscendC::LocalTensor<float> gmUbTensor32;
    AscendC::LocalTensor<ElementO> goUbTensor16;
    AscendC::LocalTensor<ElementOTmp> goUbTensor32;

    CopyUbToGmO copyUbToGmO;
};
}
#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_RESCALE_O_HPP_T

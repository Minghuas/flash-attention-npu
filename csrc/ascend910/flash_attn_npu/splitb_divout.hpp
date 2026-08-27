/**
 * Copyright (c) 2026, perf-shortSeqLargeBatch project.
 *
 * SplitB 单遍 divout epilogue（方案 B 重写，devlog #34）。
 *
 * 封装形态照抄 FAInfer rescale_o.hpp 的 operator()/SubCoreCompute（用户 FIXME #6）：
 *   - init(resource)：UB 偏移与 FAInfer 一致（GO@128KB、TV@160KB、GM/GL/LSE@168KB+）
 *   - operator() 每调用处理一个 (qSBlockIdx,qNBlockIdx) tile：内部按 FAInfer 公式把行
 *     分摊给两个 AIV（与 SplitBSoftmax 同一公式），行块循环 + SubCoreCompute
 * 与 rescale_o.hpp 的差异（四段批结构所需）：
 *   - 无在线状态机（isFirst/isLast 恒真）：OTmp GM 直读 GO；行 max/sum 从 GM stats 读入
 *     UB（GM@168KB+10K/GL@168KB+12K）——与 SplitBSoftmax 的 stats 写布局严格配对
 *   - 无 SWA 行置零（delStartRow/delEndRow/InvalidLineLSEProcess，S4 补）与 FD SplitKV
 *   - 0 行子核防护：stub MTE3 写 + MTE3_MTE2(6) set 保下一 tile wait 自配对（devlog #23）
 * O/LSE 散射（打包行还原 BSND 头主序）与 Brcb 布局参数照抄 FAInfer 原值。
 */

#ifndef SPLITB_DIVOUT_HPP
#define SPLITB_DIVOUT_HPP

#include "kernel_operator.h"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"

namespace SplitB {

using namespace Catlass;

// ============================ 单遍 divout epilogue ============================
template <typename DType>
class SplitBDivOut {
public:
    using ElementO = DType;
    using ElementOTmp = float;
    using ElementLse = float;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;
    using LayoutLse = layout::RowMajor;

    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t MAX_UB_O_ELEM_NUM = 8192;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
    static constexpr uint32_t ROW_NUM_MAX = 128;   // = Q_TILE_CEIL；stats 行距（与 host 公式一致）
    // ⑤ LSE 广播区在 tvUb 内的基址偏移（devlog #44.49 根因修复）：
    // ② 除数广播占 tv[0, 每AIV行数×8)，每 AIV 行数 ≤ ROW_NUM_MAX/2=64 → 上限 512 float；
    // ⑤ 原偏移仅 FLOAT_VECTOR_SIZE=64 → 与 ② 区在 [64,512) 重叠——下一 tile 的 ② Brcb
    // 覆写重叠行，多头 tile 的 LSE gather（仅多头分支读 tv）读到被污染值 → LSE 错
    //（实证：坏行上边界恰落在重叠边界，G=2 时 s0-23 / Sq=48 时 s0-39；B≥3 才触发
    //   = b+2 同 ping-pong 槽的流水窗口；--dump 会改变时序掩盖）。偏移取 ② 区上限
    //   512 即彻底不相交（tv 区 [160KB,170KB) 共 2560 float，余量充足）。
    static constexpr uint32_t LSE_TV_FLOAT_OFFSET = (ROW_NUM_MAX / 2) * FLOAT_BLOCK_SIZE;

    __aicore__ inline SplitBDivOut() {}

    __aicore__ inline void init(Arch::Resource<Arch::AtlasA2> &resource)
    {
        // UB 偏移照抄 rescale_o.hpp init()（与 SplitBSoftmax 分时复用，无冲突）
        constexpr uint32_t GO_UB_TENSOR_OFFSET = 8 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;
        // LSE 与 GL 同址（FAInfer 同）：Ln 就地读 gl 写 lse，时序复用无冲突
        constexpr uint32_t LSE_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;

        goUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementO>(GO_UB_TENSOR_OFFSET);
        goUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(GO_UB_TENSOR_OFFSET);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        lseUbTensor = resource.ubBuf.template GetBufferByByte<float>(LSE_UB_TENSOR_OFFSET);
    }

    __aicore__ inline
    void SetMask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = static_cast<uint64_t>(len) % static_cast<uint64_t>(FLOAT_VECTOR_SIZE);
        for (uint64_t i = 0; i < temp; i++) {
            mask |= one << i;
        }

        if (len == 128) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else if (len >= FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
        } else {
            AscendC::SetVectorMask<int8_t>(0x0, mask);
        }
    }

    // stats GM→UB：max→[rowOffsetGm]，sum→[ROW_NUM_MAX+rowOffsetGm]，读入 [0..curRowNumRound)
    // MTE2→V 自配对事件（相邻 Set+Wait，无需预置；与 FAInfer go←GM 同型）
    __aicore__ inline
    void LoadStats(AscendC::GlobalTensor<float> gStats, uint32_t rowOffsetGm, uint32_t rowNumCurLoopRound)
    {
        AscendC::DataCopyParams statParams;
        statParams.blockCount = 1;
        statParams.blockLen = rowNumCurLoopRound / FLOAT_BLOCK_SIZE;   // 32B 块单位（devlog #29）
        statParams.srcStride = 0;
        statParams.dstStride = 0;
        AscendC::DataCopy(gmUbTensor, gStats[rowOffsetGm], statParams);
        AscendC::DataCopy(glUbTensor, gStats[ROW_NUM_MAX + rowOffsetGm], statParams);
        // PipeBarrier<PIPE_MTE2>（devlog #44.10）：防 Scalar 的 set_flag 早于 MTE2
        // 拷贝完成发射（-O2 Scalar/MTE2 发射乱序窗口）
        AscendC::PipeBarrier<PIPE_MTE2>();
        // 子核事件分域（devlog #44.40）：AIV0 用 0、AIV1 用 2（详见 SubCoreCompute 头注释）
        const uint32_t evId = 2 * AscendC::GetSubBlockIdx();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evId);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // O 散射写 GM（照抄 rescale_o.hpp CopyOToGm）：打包行（头主序）还原 BSND 头主序
    // proToken/integralHead/epiToken 三部分 + oHiddenSize=D×H 行距
    __aicore__ inline
    void CopyOToGm(AscendC::GlobalTensor<ElementO> gOutput, uint32_t proTokenIdx, uint32_t proTokenNum,
                   uint32_t epiTokenNum, uint32_t integralHeadNum, uint32_t qSThisSubBlock,
                   uint32_t embed, uint32_t embedRound, uint32_t oHiddenSize)
    {
        uint32_t innerOGmOffset = 0;
        uint32_t innerGOUbOffset = 0;
        if (proTokenNum != 0U) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset + proTokenIdx * oHiddenSize],
                goUbTensor16[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    proTokenNum, embed * 2, 0, (oHiddenSize - embed) * 2, 0));
            innerOGmOffset += embed;
            innerGOUbOffset += proTokenNum * embedRound;
        }
        for (uint32_t qN_idx = 0; qN_idx < integralHeadNum; qN_idx++) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],
                goUbTensor16[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    qSThisSubBlock, embed * 2, 0, (oHiddenSize - embed) * 2, 0));
            innerOGmOffset += embed;
            innerGOUbOffset += qSThisSubBlock * embedRound;
        }
        if (epiTokenNum != 0U) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],
                goUbTensor16[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    epiTokenNum, embed * 2, 0, (oHiddenSize - embed) * 2, 0));
        }
    }

    __aicore__ inline
    void SubCoreCompute(
        AscendC::GlobalTensor<ElementO> gOutput, AscendC::GlobalTensor<ElementOTmp> gInput,
        AscendC::GlobalTensor<ElementLse> gLse,
        const LayoutO &layoutOutput, const LayoutOTmp &layoutInput, const LayoutLse &layoutLse,
        uint32_t qNThisSubBlock, uint32_t qSThisSubBlock, uint32_t totalRowNum,
        uint32_t isLastRowLoop, uint32_t rowOffsetLoop,
        uint32_t proTokenIdx, uint32_t proTokenNum, uint32_t epiTokenNum, uint32_t integralHeadNum,
        uint32_t curRowNum)
    {
        const uint32_t embed = layoutInput.shape(1);
        const uint32_t embedRound = layoutInput.stride(0);
        const uint32_t curRowNumRound = RoundUp(curRowNum, FLOAT_BLOCK_SIZE);
        // 多头 tile 时 qSBlockSize==Sq==tile 每头行数（GetQNBlockTile 不变量，见 FAInfer 注释）
        const uint32_t qSBlockSize = layoutOutput.shape(0);
        const uint32_t oHiddenSize = layoutOutput.shape(1);

        // 子核事件分域（devlog #44.40，softmax #44.24 同款）：HardEvent 为核级共享，
        // 固定 ID（原 EVENT_ID0/1/2/4/6）会被另一 AIV 的同 ID Set 越权释放。
        // evId = 2×subIdx：AIV0 用 0、AIV1 用 2；各 HardEvent 类型的标志位空间独立，
        // 同 id 不同类型不冲突（softmax 先例）。Set+Wait 相邻闭合对与跨 tile 的
        // MTE3_MTE2 共用本核 id 均安全（无生命周期交叠）。
        // 跨 tile 的 Wait<MTE3_MTE2> 已移至 operator() 的 LoadStats 之前（#44.40：
        // 原在此入口——晚于 LoadStats 执行，护不住 gl/gm 覆写）。
        const uint32_t evId = 2 * AscendC::GetSubBlockIdx();

        // ① go = OTmp 块（单遍 isFirst 分支：GM 直读 GO，照抄 FAInfer）
        AscendC::DataCopy(
            goUbTensor32, gInput,
            AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
        // PipeBarrier<PIPE_MTE2>（devlog #44.10）：同 ① LoadStats 加固
        AscendC::PipeBarrier<PIPE_MTE2>();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evId);

        // ② go = go / gl（单遍 isLast 分支：gl 已由 LoadStats 从 GM 读入 UB）
        AscendC::Brcb(
            tvUbTensor.ReinterpretCast<uint32_t>(),
            glUbTensor.ReinterpretCast<uint32_t>(),
            curRowNumRound / FLOAT_BLOCK_SIZE,
            AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();  // FIXME: 补充同步！
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evId);
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        for (uint32_t vdiv_idx = 0; vdiv_idx < embed / FLOAT_VECTOR_SIZE; ++vdiv_idx) {
            AscendC::Div<float, false>(
                goUbTensor32[vdiv_idx * FLOAT_VECTOR_SIZE],
                goUbTensor32[vdiv_idx * FLOAT_VECTOR_SIZE],
                tvUbTensor,
                (uint64_t)0,
                curRowNum,
                AscendC::BinaryRepeatParams(
                    1, 1, 0, embedRound / FLOAT_BLOCK_SIZE, embedRound / FLOAT_BLOCK_SIZE, 1));
        }
        if (embed % FLOAT_VECTOR_SIZE > 0) {
            SetMask(embed % FLOAT_VECTOR_SIZE);
            AscendC::Div<float, false>(
                goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                tvUbTensor,
                (uint64_t)0,
                curRowNum,
                AscendC::BinaryRepeatParams(
                    1, 1, 0, embedRound / FLOAT_BLOCK_SIZE, embedRound / FLOAT_BLOCK_SIZE, 1));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // ③ cast fp32→fp16
        if (std::is_same<ElementO, bfloat16_t>::value) {
            AscendC::Cast<ElementO, float, false>(
                goUbTensor16, goUbTensor32,
                AscendC::RoundMode::CAST_RINT, (uint64_t)0,
                (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                AscendC::UnaryRepeatParams(1, 1, 4, 8));
        } else {
            AscendC::Cast<ElementO, float, false>(
                goUbTensor16, goUbTensor32,
                AscendC::RoundMode::CAST_NONE, (uint64_t)0,
                (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                AscendC::UnaryRepeatParams(1, 1, 4, 8));
        }

        // PipeBarrier<PIPE_V>（devlog #44.10）：防 Scalar 的 set_flag 早于 V 完成
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evId);

        // ④ O 散射写 GM（gOutput 基址已含本行块的头偏移，照抄 FAInfer）
        CopyOToGm(
            gOutput, proTokenIdx, proTokenNum, epiTokenNum, integralHeadNum,
            qSThisSubBlock, embed, embedRound, oHiddenSize);

        // ⑤ LSE = ln(gl)+gm（单遍 isLast && OUT_ONLY 恒走；写头主序 BNS）
        if (isLastRowLoop) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Ln<float, false>(
                lseUbTensor,
                glUbTensor,
                (uint64_t)0, CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                AscendC::UnaryRepeatParams(1, 1, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add<float, false>(
                lseUbTensor,
                lseUbTensor,
                gmUbTensor,
                (uint64_t)0, CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();

            // V→MTE2 自配对：Ln/Add 写 lseUb 完成后 Brcb 才读（跨 pipe RAW，devlog #44.6）
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evId);
            // LSE 广播用独立目标区（tvUb 偏移 LSE_TV_FLOAT_OFFSET）：与 ② Div 的除数
            // 广播区 tv[0,512) 彻底不相交。原偏移 FLOAT_VECTOR_SIZE(64) 时两区在
            // [64,512) 重叠——下一 tile 的 ② Brcb 覆写重叠行污染 ⑤ 的 gather 源
            //（devlog #44.49 GQA LSE 根因；#44.6 的 WAR 修复经验延续：分区不相交
            //   优于靠事件链排序）
            AscendC::Brcb(
                tvUbTensor[LSE_TV_FLOAT_OFFSET].ReinterpretCast<uint32_t>(),
                lseUbTensor.ReinterpretCast<uint32_t>(),
                CeilDiv(totalRowNum, FLOAT_BLOCK_SIZE),
                AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();  // FIXME: 补充同步！
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evId);

            if (qNThisSubBlock == 0U) {
                // 单头 tile：连续行直写（gLse 基址已含 token 偏移，照抄 FAInfer）
                AscendC::DataCopyPad(
                    gLse, lseUbTensor,
                    AscendC::DataCopyExtParams(1, totalRowNum * sizeof(float), 0, 0, 0));
            } else {
                // 多头 tile：逐 token 跨头 gather + 头主序 scatter（参数照抄 FAInfer；
                        // 源偏移与上方 Brcb dst 成对 = LSE_TV_FLOAT_OFFSET）
                const uint32_t lseHeadStrideGm = layoutLse.stride(0);
                for (uint32_t sIdx = 0; sIdx < qSBlockSize; sIdx++) {
                    AscendC::DataCopyPad(
                        gLse[sIdx],
                        tvUbTensor[sIdx * FLOAT_BLOCK_SIZE + LSE_TV_FLOAT_OFFSET],
                        AscendC::DataCopyExtParams(
                            qNThisSubBlock, sizeof(float),
                            qSBlockSize - 1,
                            (lseHeadStrideGm - 1) * sizeof(float), 0));
                }
            }
        }
        // PipeBarrier<PIPE_MTE3>（devlog #44.10）：防 Scalar 的 set_flag 早于 MTE3
        // 的 O/LSE 拷贝完成发射
        AscendC::PipeBarrier<PIPE_MTE3>();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evId);
    }

    __aicore__ inline
    void operator()(
        AscendC::GlobalTensor<ElementO> gOutput, AscendC::GlobalTensor<ElementOTmp> gInput,
        AscendC::GlobalTensor<ElementLse> gLse, AscendC::GlobalTensor<float> gStats,
        const LayoutO &layoutOutput, const LayoutOTmp &layoutInput, const LayoutLse &layoutLse,
        GemmCoord actualBlockShape,
        uint32_t qSBlockSizeTile, uint32_t qNBlockSize)
    {
        const uint32_t rowNum = actualBlockShape.m();
        const uint32_t embed = actualBlockShape.n();

        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();

        // 双 AIV 行分摊（FAInfer 精确公式，与 SplitBSoftmax 同款）
        const uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        const uint32_t qNThisSubBlock = (qNBlockSize == 1U) ? 0U :
            ((subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock);
        // Bug③a 修复（devlog #44.52，与 SplitBSoftmax 成对）：分摊边界对齐 8——
        // LoadStats 的 GM 读起点同样以 32B 块为单位，非对齐 split（Sq=31 → 15）会
        // 非对齐读；与 softmax 侧同公式保持写/读分摊一致。
        const uint32_t inRowSplitRaw = (qNBlockSize == 1U) ?
            (qSBlockSizeTile / subBlockNum) : (qSBlockSizeTile * qNSplitSubBlock);
        const uint32_t inRowSplitSubBlock = RoundDown(inRowSplitRaw, FLOAT_BLOCK_SIZE);
        const uint32_t inRowActualThisSubBlock = (subBlockIdx == 1U) ?
            (rowNum - inRowSplitSubBlock) : inRowSplitSubBlock;
        const uint32_t inRowOffsetThisSubBlock = subBlockIdx * inRowSplitSubBlock;
        const uint32_t outRowOffsetThisSubBlock = (qNBlockSize == 1U) ? inRowOffsetThisSubBlock : 0;
        const uint32_t outColOffsetThisSubBlock = (qNBlockSize == 1U) ? 0 :
            subBlockIdx * qNSplitSubBlock * embed;
        const uint32_t qSThisSubBlock = (qNBlockSize == 1U) ? inRowActualThisSubBlock : qSBlockSizeTile;
        const int64_t outOffsetSubBlock =
            layoutOutput.GetOffset(MatrixCoord(outRowOffsetThisSubBlock, outColOffsetThisSubBlock));

        // LSE 头主序：多头 tile 按头偏移；单头 tile 按行偏移（token），照抄 FAInfer
        const uint32_t outLseRowOffsetThisSubBlock = (qNBlockSize == 1U) ? 0 :
            subBlockIdx * qNSplitSubBlock;
        const uint32_t outLseColOffsetThisSubBlock = (qNBlockSize == 1U) ?
            inRowOffsetThisSubBlock : 0;
        const int64_t offsetLse =
            layoutLse.GetOffset(MatrixCoord(outLseRowOffsetThisSubBlock, outLseColOffsetThisSubBlock));
        auto gLseThisSubBlock = gLse[offsetLse];

        if (inRowActualThisSubBlock == 0U) {
            // 0 行子核：stub MTE3 写 + MTE3_MTE2(6) set 保下一 tile 的 wait 自配对（devlog #23/#34）
            AscendC::DataCopyParams stubParams;
            stubParams.blockCount = 1;
            stubParams.blockLen = 8 * sizeof(ElementO);
            stubParams.srcStride = 0;
            stubParams.dstStride = 0;
            AscendC::DataCopyPad(gOutput, goUbTensor16, stubParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(2 * AscendC::GetSubBlockIdx());   // #44.40 分域
            return;
        }

        const uint32_t maxRowNumPerLoop = MAX_UB_O_ELEM_NUM / embed;
        const uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        const uint32_t rowLoopNum = CeilDiv(inRowActualThisSubBlock, rowNumTile);

        // 行块循环（实际恒 1 次：子核行数 ≤64 ≤ rowNumTile(≥64)；保留循环保 FAInfer 形态）
        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum; rowLoopIdx++) {
            const uint32_t rowOffsetLoop = rowLoopIdx * rowNumTile;
            const uint32_t rowOffsetCurLoop = inRowOffsetThisSubBlock + rowOffsetLoop;
            const uint32_t rowActualCurLoop =
                (rowLoopIdx == (rowLoopNum - 1U)) ? inRowActualThisSubBlock - rowLoopIdx * rowNumTile : rowNumTile;

            // 上一 tile 的 O/LSE MTE3 读（goUb16 / lseUb≡glUb）全部完成后，才允许
            // LoadStats 的 MTE2 写覆写 gl/gm（devlog #44.40：原 Wait 在 SubCoreCompute
            // 入口，晚于 LoadStats → 护不住；-O2 下末尾的 PipeBarrier<MTE3> 不足，
            // t1 的 stats 覆写抢在 t0 LSE 拷贝读之前 → b1 h0 s16-31 的 LSE 写出 h1
            // 的原始 sum，t55 逐位实证）。evId 分域防另一 AIV 的 Set 越权放行本 wait
            //（#44.24/#44.40）；首轮由 kernel init 预置（MTE3_MTE2 ID 0/2 已 set）。
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(2 * AscendC::GetSubBlockIdx());

            // stats GM→UB（全局行偏移；与 SplitBSoftmax::CopyStatsToGm 布局严格配对）
            LoadStats(gStats, rowOffsetCurLoop, rowActualCurLoop);

            const int64_t offsetOutput =
                static_cast<int64_t>(rowLoopIdx * rowNumTile / qSThisSubBlock * embed) + outOffsetSubBlock;
            auto gOutputCurLoop = gOutput[offsetOutput];

            const int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetCurLoop, 0));
            auto gInputCurLoop = gInput[offsetInput];
            auto layoutInputCurLoop = layoutInput.GetTileLayout(MatrixCoord(rowActualCurLoop, embed));

            const uint32_t proTokenIdx = rowOffsetLoop % qSThisSubBlock;
            const uint32_t proTokenNum =
                AscendC::Std::min(rowActualCurLoop, (qSThisSubBlock - proTokenIdx)) % qSThisSubBlock;
            const uint32_t integralHeadNum = (rowActualCurLoop - proTokenNum) / qSThisSubBlock;
            const uint32_t epiTokenNum = rowActualCurLoop - proTokenNum - integralHeadNum * qSThisSubBlock;

            SubCoreCompute(
                gOutputCurLoop, gInputCurLoop, gLseThisSubBlock,
                layoutOutput, layoutInputCurLoop, layoutLse,
                qNThisSubBlock, qSThisSubBlock, inRowActualThisSubBlock,
                (rowLoopIdx == rowLoopNum - 1U),
                rowOffsetLoop,
                proTokenIdx, proTokenNum, epiTokenNum, integralHeadNum,
                rowActualCurLoop);
        }
    }

private:
    AscendC::LocalTensor<ElementO> goUbTensor16;
    AscendC::LocalTensor<float> goUbTensor32;
    AscendC::LocalTensor<float> tvUbTensor;
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> lseUbTensor;
};

} // namespace SplitB

#endif // SPLITB_DIVOUT_HPP

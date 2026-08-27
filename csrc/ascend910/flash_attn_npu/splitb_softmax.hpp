/**
 * Copyright (c) 2026, perf-shortSeqLargeBatch project.
 *
 * SplitB 单遍 softmax epilogue（方案 B 重写，devlog #34）。
 *
 * 封装形态照抄 FAInfer online_softmax.hpp 的 no-mask operator()（用户 FIXME #5）：
 *   - init(resource, scale, softcap)：UB 偏移与 FAInfer 一致（LS 2×32KB ping/pong、
 *     LP 2×16KB、TV@160KB、LM/LL/SOFTCAP@168KB+）
 *   - operator() 每调用处理一个 (qSBlockIdx,qNBlockIdx) tile：内部按 FAInfer 公式把行
 *     分摊给两个 AIV（qNBlockSize>1 按头对半；==1 按 qSBlockSize/2 行对半），行块
 *     ping/pong 预取 + SubCoreCompute
 * 与 online_softmax.hpp 的差异（四段批结构所需）：
 *   - 无在线状态机（S2 不切分，单遍即全局）：行 max/sum 直接写 GM stats 供段4 divout
 *     消费——stats 布局：tile 块内 [0,ROW_NUM_MAX)=rowmax, [ROW_NUM_MAX,2×ROW_NUM_MAX)=
 *     rowsum，行距 ROW_NUM_MAX=Q_TILE_CEIL=128（与 splitb_host.cpp workspace 公式一致）
 *   - 0 行子核防护：rowActualThisSubBlock==0 时补最小真 MTE3 写再返回（devlog #23 竞态）
 *   - 去 MTE3_V(EVENT_ID4) 等待：跨段 GM/UB 冲突已由 CrossCoreFlag 链 + 管道内序保证
 *     （该等待在 FAInfer 由 rescale 的 LSE 拷贝 set 配对，SplitB 下无此配对需求）
 * 行归约原语为 RowmaxTAILTILE/RowsumTAILTILE 形态（S2≤128 恒走 tail 分支，devlog #18/#20）。
 */

#ifndef SPLITB_SOFTMAX_HPP
#define SPLITB_SOFTMAX_HPP

#include "kernel_operator.h"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"

namespace SplitB {

using namespace Catlass;

// ---------------- 行归约原语（TailTile 变体，照抄 FAInfer RowmaxTAILTILE/RowsumTAILTILE） ----------------
__aicore__ inline uint32_t SBCeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

constexpr uint32_t SB_FLOAT_BLOCK_SIZE = 8;     // fp32 一个 block 的元素数（32B/4B）
constexpr uint32_t SB_FLOAT_VECTOR_SIZE = 64;   // fp32 一条 vector 指令的元素数（256B/4B）
constexpr uint32_t SB_REDUCE_UB_SIZE = 1024;    // 归约中间区大小（floats）

__aicore__ inline void SplitBSetVecMask(int32_t len)
{
    uint64_t mask = 0;
    uint64_t one = 1;
    uint64_t temp = len % SB_FLOAT_VECTOR_SIZE;
    for (int64_t i = 0; i < temp; i++) {
        mask |= one << i;
    }
    if (len == 128 || len == 0) {
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    } else if (len >= SB_FLOAT_VECTOR_SIZE) {
        AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
    } else {
        AscendC::SetVectorMask<int8_t>(0x0, mask);
    }
}

__aicore__ inline void SplitBSetBlockReduceMask(int32_t len)
{
    if (len > 8 || len < 1) {
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        return;
    }
    uint64_t subMask = ((uint64_t)1 << len) - 1;
    uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask + (subMask << 56) +
                         (subMask << 40) + (subMask << 24) + (subMask << 8);
    AscendC::SetVectorMask<int8_t>(maskValue, maskValue);
}

__aicore__ inline void
SplitBRowMax(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowmaxUb,
             const AscendC::LocalTensor<float> &tvUb, uint32_t numRowsRound, uint32_t numElems,
             uint32_t numElemsAligned)
{
    if (numElems >= SB_FLOAT_VECTOR_SIZE) {
        AscendC::BlockReduceMax<float, false>(
            tvUb, srcUb, numRowsRound, 0, 1, 1, numElemsAligned / SB_FLOAT_BLOCK_SIZE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceMax<float, false>(
            rowmaxUb, tvUb, SBCeilDiv(numRowsRound * SB_FLOAT_BLOCK_SIZE, SB_FLOAT_VECTOR_SIZE), 0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint64_t rowIdx = 1; rowIdx < (uint64_t)numElems / SB_FLOAT_VECTOR_SIZE; ++rowIdx) {
            AscendC::BlockReduceMax<float, false>(
                tvUb, srcUb[rowIdx * SB_FLOAT_VECTOR_SIZE], numRowsRound, 0, 1, 1,
                numElemsAligned / SB_FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceMax<float, false>(
                tvUb[SB_REDUCE_UB_SIZE], tvUb,
                SBCeilDiv(numRowsRound * SB_FLOAT_BLOCK_SIZE, SB_FLOAT_VECTOR_SIZE), 0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            SplitBSetVecMask(numRowsRound);
            AscendC::Max<float, false>(
                rowmaxUb, rowmaxUb, tvUb[SB_REDUCE_UB_SIZE], (uint64_t)0, 1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }
    if (numElems % SB_FLOAT_VECTOR_SIZE > 0) {
        SplitBSetVecMask(numElems % SB_FLOAT_VECTOR_SIZE);
        AscendC::BlockReduceMax<float, false>(
            tvUb, srcUb[numElems / SB_FLOAT_VECTOR_SIZE * SB_FLOAT_VECTOR_SIZE], numRowsRound, 0, 1, 1,
            numElemsAligned / SB_FLOAT_BLOCK_SIZE);
        AscendC::PipeBarrier<PIPE_V>();
        SplitBSetBlockReduceMask(SBCeilDiv(numElems % SB_FLOAT_VECTOR_SIZE, SB_FLOAT_BLOCK_SIZE));
        if (numElems < SB_FLOAT_VECTOR_SIZE) {
            AscendC::BlockReduceMax<float, false>(
                rowmaxUb, tvUb, SBCeilDiv(numRowsRound * SB_FLOAT_BLOCK_SIZE, SB_FLOAT_VECTOR_SIZE), 0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::BlockReduceMax<float, false>(
                tvUb[SB_REDUCE_UB_SIZE], tvUb,
                SBCeilDiv(numRowsRound * SB_FLOAT_BLOCK_SIZE, SB_FLOAT_VECTOR_SIZE), 0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            SplitBSetVecMask(numRowsRound);
            AscendC::Max<float, false>(
                rowmaxUb, rowmaxUb, tvUb[SB_REDUCE_UB_SIZE], (uint64_t)0, 1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
}

__aicore__ inline void
SplitBRowSum(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowsumUb,
             const AscendC::LocalTensor<float> &tvUb, uint32_t numRowsRound, uint32_t numElems,
             uint32_t numElemsAligned)
{
    if (numElems >= SB_FLOAT_VECTOR_SIZE) {
        AscendC::BlockReduceSum<float, false>(
            tvUb, srcUb, numRowsRound, 0, 1, 1, numElemsAligned / SB_FLOAT_BLOCK_SIZE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceSum<float, false>(
            rowsumUb, tvUb, SBCeilDiv(numRowsRound * SB_FLOAT_BLOCK_SIZE, SB_FLOAT_VECTOR_SIZE), 0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint64_t rowIdx = 1; rowIdx < (uint64_t)numElems / SB_FLOAT_VECTOR_SIZE; ++rowIdx) {
            AscendC::BlockReduceSum<float, false>(
                tvUb, srcUb[rowIdx * SB_FLOAT_VECTOR_SIZE], numRowsRound, 0, 1, 1,
                numElemsAligned / SB_FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceSum<float, false>(
                tvUb[SB_REDUCE_UB_SIZE], tvUb,
                SBCeilDiv(numRowsRound * SB_FLOAT_BLOCK_SIZE, SB_FLOAT_VECTOR_SIZE), 0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            SplitBSetVecMask(numRowsRound);
            AscendC::Add<float, false>(
                rowsumUb, rowsumUb, tvUb[SB_REDUCE_UB_SIZE], (uint64_t)0, 1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }
    if (numElems % SB_FLOAT_VECTOR_SIZE > 0) {
        SplitBSetVecMask(numElems % SB_FLOAT_VECTOR_SIZE);
        AscendC::BlockReduceSum<float, false>(
            tvUb, srcUb[numElems / SB_FLOAT_VECTOR_SIZE * SB_FLOAT_VECTOR_SIZE], numRowsRound, 0, 1, 1,
            numElemsAligned / SB_FLOAT_BLOCK_SIZE);
        AscendC::PipeBarrier<PIPE_V>();
        SplitBSetBlockReduceMask(SBCeilDiv(numElems % SB_FLOAT_VECTOR_SIZE, SB_FLOAT_BLOCK_SIZE));
        if (numElems < SB_FLOAT_VECTOR_SIZE) {
            AscendC::BlockReduceSum<float, false>(
                rowsumUb, tvUb, SBCeilDiv(numRowsRound * SB_FLOAT_BLOCK_SIZE, SB_FLOAT_VECTOR_SIZE), 0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::BlockReduceSum<float, false>(
                tvUb[SB_REDUCE_UB_SIZE], tvUb,
                SBCeilDiv(numRowsRound * SB_FLOAT_BLOCK_SIZE, SB_FLOAT_VECTOR_SIZE), 0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            SplitBSetVecMask(numRowsRound);
            AscendC::Add<float, false>(
                rowsumUb, rowsumUb, tvUb[SB_REDUCE_UB_SIZE], (uint64_t)0, 1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
}

// ============================ 单遍 softmax epilogue ============================
template <typename DType, bool HAS_SOFTCAP>
class SplitBSoftmax {
public:
    using ElementS = float;
    using ElementP = DType;
    using LayoutP = layout::RowMajor;
    using LayoutS = layout::RowMajor;

    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 8192;   // LS ping/pong 单槽（floats）
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
    static constexpr uint32_t ROW_NUM_MAX = 128;          // = Q_TILE_CEIL；stats 行距（与 host 公式一致）

    // kernel 骨架段（stage<3）的最小 stub 写需要 LP 偏移（devlog #23）
    static constexpr uint32_t LP_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;

    __aicore__ inline SplitBSoftmax() {}

    __aicore__ inline void init(Arch::Resource<Arch::AtlasA2> &resource, float scaleValue_,
                                float softcapValue_)
    {
        // UB 偏移照抄 online_softmax.hpp init()（与 SplitBDivOut 分时复用，无冲突）
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t LM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 8 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t LL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 11 * UB_UINT8_VECTOR_SIZE;
        // SOFTCAP 与 LM 同址（FAInfer 同）：ApplySoftcap 先于 rowmax，时序复用无冲突
        constexpr uint32_t SOFTCAP_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 8 * UB_UINT8_VECTOR_SIZE;

        scaleValue = scaleValue_;
        softcapValue = softcapValue_;
        lsUbTensor = resource.ubBuf.template GetBufferByByte<float>(LS_UB_TENSOR_OFFSET);
        lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementP>(LP_UB_TENSOR_OFFSET);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<float>(LM_UB_TENSOR_OFFSET);
        llUbTensor = resource.ubBuf.template GetBufferByByte<float>(LL_UB_TENSOR_OFFSET);
        softcapUbTensor = resource.ubBuf.template GetBufferByByte<float>(SOFTCAP_UB_TENSOR_OFFSET);
    }

    __aicore__ inline
    void CopySGmToUb(AscendC::GlobalTensor<ElementS> gInput, uint32_t sUbOffset,
                     uint32_t rowNumCurLoop, uint32_t columnNumRound, uint32_t columnNumPad)
    {
        AscendC::DataCopy(
            lsUbTensor[sUbOffset],
            gInput,
            AscendC::DataCopyParams(
                rowNumCurLoop, columnNumRound / FLOAT_BLOCK_SIZE,
                (columnNumPad - columnNumRound) / FLOAT_BLOCK_SIZE, 0));
    }

    __aicore__ inline
    void ScaleS(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        AscendC::Muls<float, false>(
            lsUbTensor[sUbOffset],
            lsUbTensor[sUbOffset],
            scaleValue,
            (uint64_t)0,
            SBCeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void ApplySoftcap(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        // 照抄 online_softmax.hpp ApplySoftcap（S4 启用；S3 HAS_SOFTCAP=false 不实例化）
        uint32_t repeatTimes = SBCeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE);
        AscendC::UnaryRepeatParams repeatParams(1, 1, 8, 8);

        AscendC::Maxs<float, false>(
            lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], -8.8f, (uint64_t)0, repeatTimes, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls<float, false>(
            lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], -2.0f, (uint64_t)0, repeatTimes, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp<float, false>(
            lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], (uint64_t)0, repeatTimes, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds<float, false>(
            lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], 1.0f, (uint64_t)0, repeatTimes, repeatParams);
        AscendC::Duplicate<float, false>(softcapUbTensor, 2 * softcapValue, (uint64_t)0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div<float, false>(
            lsUbTensor[sUbOffset], softcapUbTensor, lsUbTensor[sUbOffset], (uint64_t)0, repeatTimes,
            AscendC::BinaryRepeatParams(1, 1, 1, 8, 0, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds<float, false>(
            lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], -softcapValue, (uint64_t)0, repeatTimes, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // 单遍：lm 即全局 max（无 gm/hm 在线状态）
    __aicore__ inline
    void CalcLocalRowMax(uint32_t sUbOffset, uint32_t rowNumCurLoopRound,
                         uint32_t columnNum, uint32_t columnNumRound)
    {
        SplitBRowMax(lsUbTensor[sUbOffset], lmUbTensor, tvUbTensor,
                     rowNumCurLoopRound, columnNum, columnNumRound);
    }

    // 广播减 + exp（hm=lm 单遍语义；行批 ≤16 走 FAInfer 验证区间，devlog #18/#20）
    __aicore__ inline
    void CalcExp(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound,
                 uint32_t columnNum, uint32_t columnNumRound)
    {
        // max 广播 = Brcb（FAInfer 权威形态，devlog #44.19 回退说明）：
        // BrcbRepeatParams(1,8)——每个 src 块（8 个 max）广播成 8 个 dst 块，
        // repeatTimes=rows×8/64 恰好覆盖 rows 行 × 每行 8 元素；该布局与下方 Sub 的
        // src1RepStride=4 块语义严格配对，不可替换为逐行 Duplicate（Duplicate 单
        // repeat 写 64 元素会覆盖后续行槽位，破坏布局）。tvUb 与 SplitBRowMax 的
        // 复用由 RowMax 尾部 PipeBarrier<PIPE_V> 保证序（V 管道内序 + 屏障）。
        AscendC::Brcb(
            tvUbTensor.ReinterpretCast<uint32_t>(),
            lmUbTensor.ReinterpretCast<uint32_t>(),
            rowNumCurLoopRound / FLOAT_BLOCK_SIZE,
            AscendC::BrcbRepeatParams(1, 8));  // FIXME: 这里和CalcLocalRowMax复用了tvUbTensor，需要检查是否安全? 需要加同步吧？
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t subIdx = 0; subIdx < columnNum / FLOAT_VECTOR_SIZE; ++subIdx) {
            AscendC::Sub<float, false>(
                lsUbTensor[sUbOffset][subIdx * FLOAT_VECTOR_SIZE],
                lsUbTensor[sUbOffset][subIdx * FLOAT_VECTOR_SIZE],
                tvUbTensor,
                (uint64_t)0,
                rowNumCurLoop,
                AscendC::BinaryRepeatParams(
                    1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE, columnNumRound / FLOAT_BLOCK_SIZE, 1));
        }
        if (columnNum % FLOAT_VECTOR_SIZE > 0) {
            SplitBSetVecMask(columnNum % FLOAT_VECTOR_SIZE);
            AscendC::Sub<float, false>(
                lsUbTensor[sUbOffset][columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                lsUbTensor[sUbOffset][columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                tvUbTensor,
                (uint64_t)0,
                rowNumCurLoop,
                AscendC::BinaryRepeatParams(
                    1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE, columnNumRound / FLOAT_BLOCK_SIZE, 1));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp<float, false>(
            lsUbTensor[sUbOffset],
            lsUbTensor[sUbOffset],
            (uint64_t)0,
            SBCeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void CalcLocalRowSum(uint32_t sUbOffset, uint32_t rowNumCurLoopRound,
                         uint32_t columnNum, uint32_t columnNumRound)
    {
        SplitBRowSum(lsUbTensor[sUbOffset], llUbTensor, tvUbTensor,
                     rowNumCurLoopRound, columnNum, columnNumRound);
    }

    __aicore__ inline
    void DownCastP(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        if (std::is_same<ElementP, bfloat16_t>::value) {
            AscendC::Cast<ElementP, float, false>(
                lpUbTensor[sUbOffset],
                lsUbTensor[sUbOffset],
                AscendC::RoundMode::CAST_RINT,
                (uint64_t)0,
                SBCeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
                AscendC::UnaryRepeatParams(1, 1, 4, 8));
        } else {
            AscendC::Cast<ElementP, float, false>(
                lpUbTensor[sUbOffset],
                lsUbTensor[sUbOffset],
                AscendC::RoundMode::CAST_NONE,
                (uint64_t)0,
                SBCeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
                AscendC::UnaryRepeatParams(1, 1, 4, 8));
        }
    }

    __aicore__ inline
    void CopyPUbToGm(AscendC::GlobalTensor<ElementP> gOutput, uint32_t sUbOffset,
                     uint32_t rowNumCurLoop, uint32_t columnNumRound, uint32_t columnNumPad)
    {
        AscendC::DataCopy(
            gOutput,
            lpUbTensor[sUbOffset],
            AscendC::DataCopyParams(
                rowNumCurLoop, columnNumRound / BLOCK_SIZE, 0, (columnNumPad - columnNumRound) / BLOCK_SIZE));
    }

    // stats → GM：max→[rowOffsetGm]，sum→[ROW_NUM_MAX+rowOffsetGm]
    // 置于 P 拷贝之后（MTE3 序内）：数据依赖 cast/rowmax/rowsum 经 V_MTE3 事件链保证完成
    __aicore__ inline
    void CopyStatsToGm(AscendC::GlobalTensor<float> gStats, uint32_t rowOffsetGm,
                       uint32_t rowNumCurLoopRound)
    {
        // plain DataCopy：blockLen 单位 = 32B 块 = 8 floats（devlog #29）
        // lmUbTensor：行最大值 
        // llUbTensor：行和
        // 保存到GM，供第四阶段的计算使用
        AscendC::DataCopyParams statParams;
        statParams.blockCount = 1;
        statParams.blockLen = rowNumCurLoopRound / FLOAT_BLOCK_SIZE;
        statParams.srcStride = 0;
        statParams.dstStride = 0;
        AscendC::DataCopy(gStats[rowOffsetGm], lmUbTensor, statParams);
        AscendC::DataCopy(gStats[ROW_NUM_MAX + rowOffsetGm], llUbTensor, statParams);
    }

    __aicore__ inline
    void SubCoreCompute(
        AscendC::GlobalTensor<ElementP> gOutput, const LayoutP &layoutOutput,
        AscendC::GlobalTensor<float> gStats,
        uint32_t rowOffsetCurLoop, uint32_t rowOffsetThisSubBlock,
        uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound,
        uint32_t columnNum, uint32_t columnNumRound, uint32_t columnNumPad,
        uint32_t pingpongFlag)
    {
        // 子核事件分域（devlog #44.24）：双 AIV 并发跑同一自配对链，若 HardEvent 为
        // 核级共享，AIV1 的 Set 可提前释放 AIV0 的 Wait（AIV0 的 MTE2 半程被读 →
        // s8-15 旧值，b1t0 实证）。id = pingpong + 2×subIdx：AIV0 用 0/1、AIV1 用 2/3。
        const uint32_t evId = pingpongFlag + 2 * AscendC::GetSubBlockIdx();
        const uint32_t sUbOffset = pingpongFlag * MAX_UB_S_ELEM_NUM;
        // MTE3_V wait 提前到 RowMax 前（devlog #44.8）：RowMax 写 lmUb，上一 tile 的
        // CopyStatsToGm（MTE3 读 lmUb/llUb）必须在本 tile RowMax 前完成。原 wait 在
        // CalcExp 后只保护 lpUb（DownCastP 的写），lmUb 的写竞态仍在 → 上一 tile
        // stats 拷贝流末尾行（如 s63）被污染 → divout 末行除错（t26 b2 h0 s63
        // err=6.25 实证；t16 h1-h6 中等误差同源）。CalcExp 读的 lsUb 槽由 V_MTE2
        // 链（V 侧）保护，MTE3 不碰，提前 wait 无副作用。
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evId);
        CalcLocalRowMax(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound);
        CalcExp(sUbOffset, rowNumCurLoop, rowNumCurLoopRound, columnNum, columnNumRound);
        DownCastP(sUbOffset, rowNumCurLoop, columnNumRound);
        CalcLocalRowSum(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound);
        // V→MTE3 同步必须覆盖 CalcLocalRowSum（devlog #44.2）：CopyStatsToGm 读 llUb，
        // 若 SetFlag<V_MTE3> 只覆盖 DownCastP，则 MTE3 的 stats 拷贝可能与 V 的 RowSum
        // 并发执行，读到旧 sum → DivO 按错 sum 归一化（-O2 才暴露的隐式依赖）。
        // 置于 RowSum 之后：P 拷贝（读 lpUb）与 stats 拷贝（读 lmUb/llUb）均得到保证。
        // PipeBarrier<PIPE_V>（Scalar 等 V 排空，devlog #44.10）：set_flag 是 Scalar
        // 发射指令（CCE_SCALAR 实证），-O2 下 Scalar 与 V 的发射可乱序——若 set_flag
        // 先于 RowSum 尾部发射，事件提前置位 → 拷贝读未完成的 llUb（判刀 W 残留
        // 错误的候选根因）。
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evId);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evId);
        CopyPUbToGm(gOutput, sUbOffset, rowNumCurLoop, columnNumRound, columnNumPad);
        CopyStatsToGm(gStats, rowOffsetThisSubBlock + rowOffsetCurLoop, rowNumCurLoopRound);
        // MTE3_V 置于两个拷贝之后：覆盖 P+stats 双写。原位于 P 拷贝后（devlog #44.6）：
        // stats 拷贝尚在飞行时下一 tile 的 WaitFlag<MTE3_V> 即放行，其 CalcLocalRowMax/
        // CalcLocalRowSum（V，写 lmUb/llUb）与 stats 拷贝（读 lmUb/llUb）竞态 → 污染
        // 前一 tile 的 stats（t16 h1-h6 中等误差；全 1 输入因 rowmax 处处相同被掩盖）。
        // PipeBarrier<PIPE_MTE3>（devlog #44.10）：同理防 Scalar 的 set_flag 早于
        // MTE3 拷贝完成发射。
        AscendC::PipeBarrier<PIPE_MTE3>();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evId);
    }

    __aicore__ inline
    void operator()(
        AscendC::GlobalTensor<ElementP> gOutput, AscendC::GlobalTensor<ElementS> gInput,
        AscendC::GlobalTensor<float> gStats,
        const LayoutP &layoutOutput, const LayoutS &layoutInput,
        GemmCoord actualBlockShape,
        uint32_t qSBlockSize, uint32_t qNBlockSize)
    {
        const uint32_t rowNum = actualBlockShape.m();
        const uint32_t columnNum = actualBlockShape.n();
        const uint32_t columnNumRound = RoundUp(columnNum, BLOCK_SIZE);
        const uint32_t columnNumPad = layoutInput.stride(0);

        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();

        // 双 AIV 行分摊（FAInfer 精确公式：多头 tile 按头对半；单头 tile 按 qSBlockSize/2 行对半）
        const uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        // Bug③a 修复（devlog #44.52）：分摊边界对齐 8（FLOAT_BLOCK_SIZE）。stats 区行距
        // 1 float，CopyStatsToGm 的 blockLen 以 32B 块为单位——不对齐时：①AIV0 的块长
        // 取整多写的 1 行是 UB 中从未装载的 rounding 行（t11 dump 实测 max=2.6e11/
        // sum=inf），恰好落进 AIV1 首行 gStats[split] → 双写竞争；②AIV1 写起点
        // （如 Sq=31 → split=15 → +60B）非 32B 对齐，非对齐 MTE3 写抹脏邻块 → 受害行
        // 在 8-14 / 15 间随时序漂移。对齐后 AIV0 写 [0,8k) 恰好整块无垃圾尾行，AIV1
        // 起点对齐、垃圾尾行 ≥Sq 不被 divout 读取。FAInfer 原版 qSBlockSize 恒 2 的幂
        // （split 天然对齐）故未暴露；RoundDown 到 0 时走既有 0 行子核路径。
        const uint32_t rowSplitRaw = (qNBlockSize == 1) ?
            (qSBlockSize / 2) : (qSBlockSize * qNSplitSubBlock);
        const uint32_t rowSplitSubBlock = RoundDown(rowSplitRaw, FLOAT_BLOCK_SIZE);
        const uint32_t rowActualThisSubBlock = (subBlockIdx == 1) ?
            (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        const uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;

        if (rowActualThisSubBlock == 0) {
            return;
        }

        const uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
        const uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);
        const uint32_t preLoad = 1;

        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            if (rowLoopIdx < rowLoopNum) {
                // 预取下一行块（GM→UB；V_MTE2 首轮由 init 预置满足）
                const uint32_t pingpongFlag = rowLoopIdx % 2;
                const uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                const uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                const uint32_t rowNumCurLoop = (rowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                const int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInput];

                const uint32_t evId = pingpongFlag + 2 * AscendC::GetSubBlockIdx();
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evId);
                CopySGmToUb(gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM),
                        rowNumCurLoop, columnNumRound, columnNumPad);
                // PipeBarrier<PIPE_MTE2>（devlog #44.10）：防 Scalar 的 set_flag 早于
                // MTE2 拷贝完成发射（-O2 下 Scalar/MTE2 发射乱序窗口）
                AscendC::PipeBarrier<PIPE_MTE2>();
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evId);
            }
            if (rowLoopIdx >= preLoad) {
                const uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                const uint32_t pingpongFlag = delayedRowLoopIdx % 2;
                const uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                const uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                const uint32_t rowNumCurLoop =
                    (delayedRowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                
                // FIXME: 参数不一致
                const uint32_t rowNumCurLoopRound = RoundUp(rowNumCurLoop, FLOAT_BLOCK_SIZE);

                const int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));

                const uint32_t evId = pingpongFlag + 2 * AscendC::GetSubBlockIdx();
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evId);
                // ScaleS 恢复（devlog #44.42：调试期曾禁用对齐脚本；GM 的 S 保持 raw QK
                // 输出，scale 在 UB 内做——S dump(100 系) 对脚本 s_raw 仍然成立）
                ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                if constexpr (HAS_SOFTCAP) {
                    ApplySoftcap((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                }
                SubCoreCompute(
                    gOutputCurLoop, layoutOutputCurLoop, gStats,
                    rowOffsetCurLoop, rowOffsetThisSubBlock,
                    rowNumCurLoop, rowNumCurLoopRound,
                    columnNum, columnNumRound, columnNumPad,
                    pingpongFlag);
            }
        }
    }

private:
    float scaleValue;
    float softcapValue;
    AscendC::LocalTensor<float> lsUbTensor;
    AscendC::LocalTensor<ElementP> lpUbTensor;
    AscendC::LocalTensor<float> tvUbTensor;
    AscendC::LocalTensor<float> lmUbTensor;
    AscendC::LocalTensor<float> llUbTensor;
    AscendC::LocalTensor<float> softcapUbTensor;
};

} // namespace SplitB

#endif // SPLITB_SOFTMAX_HPP

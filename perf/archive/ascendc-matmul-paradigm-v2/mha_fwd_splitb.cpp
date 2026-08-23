/**
 * Copyright (c) 2026, perf-shortSeqLargeBatch project.
 *
 * SplitB 前向 kernel（大 Batch 小 SeqLen 场景）——骨架版（P3 步 1）。
 *
 * 照搬自 ops-transformer flash_attention_score_bn2gs1s2_b.h
 * （FlashAttentionScoreBn2gs1s2B，配套 tiling 类 FlashAttentionScoreTilingB）。
 * 本文件当前是【编译链路验证骨架】：
 *   - matmul 类型 / 成员 / REGIST_MATMUL_OBJ 注册链路已按参考建立（R1 验证目标）
 *   - Init 仅完成 GM 指针与基础常量绑定；Process 为空实现（TODO(P3-步2)：照搬参考
 *     四段计算 + 3 槽 boIdx 流水）
 * 设计文档：perf/design/splitb_integration.md（v2）
 * 深度解读：perf/analysis/reference_splitb_deep_dive.md
 */

#ifndef MHA_FWD_SPLITB_CPP_
#define MHA_FWD_SPLITB_CPP_

#include "kernel_operator.h"
// R1 验证核心：AscendC matmul 高阶 API（matmul::Matmul / IterateBatch /
// BATCH_LESS_THAN_L1 / REGIST_MATMUL_OBJ）。参考实现原生 matmul 栈，不走 catlass
// （catlass 无 batch 驻留 L1 语义，见 perf/analysis/our_fa_extension_points.md §4.1）。
#include "lib/matmul_intf.h"
#include "splitb_tilingdata.h"

// 照搬参考 flash_attention_score.cpp:35：MatmulType/TPosition/CubeFormat/LayoutMode/
// matmul::Matmul 等高阶 API 类型位于 AscendC 命名空间。
using namespace AscendC;

namespace SplitB {

// 每个在飞 boIdx 的分块参数（照搬参考 SplitBExtraInfo）
struct SplitBExtraInfo {
    int64_t boIdx;
    int64_t biN2GoIdx;
    int64_t s1oIdx;
    int64_t taskId;
    int64_t vecS1BaseSize;    // S1 基本块
    int64_t vecS1TailSize;    // S1 尾块
    int64_t s2AlignSize;      // S2 基本块大小（16 对齐）
    int64_t s2AlignBlockSize; // S2 基本块 32 对齐大小
    int64_t s1Vec2BaseSize;
    int64_t s1Vec2BaseTailSize;
    int64_t s1Vec2OuterSize;
    int64_t qCoreOffset;
    int64_t softmaxCopyOutLimit;
    int64_t softmaxCopyOutSize;
    int64_t softmaxOutOffset = 0;
};

// INPUT_T：输入 dtype（half / bfloat16_t）
// T：计算 dtype（v1 恒 float，照搬参考 B 模板实例化 "half, float"）
// HAS_SOFTCAP：我方特性（参考无），Vec1 的 Muls(scale) 后、mask/softmax 前施加 tanh 缩放
// HAS_ATTEN：mask 编译期门控（照搬参考 hasAtten；causal/SWA 均为 true，差异在
//            tiling 的 compress mode / 偏移推导，运行时区分）
template <typename INPUT_T, typename T = float, bool HAS_ATTEN = false, bool HAS_SOFTCAP = false>
class FlashAttentionScoreSplitB {
public:
    __aicore__ inline FlashAttentionScoreSplitB() {};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                const SplitBTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // ---------------- matmul 定义（照搬参考 bn2gs1s2_b.h:107-138） ----------------
    // v1 布局固定 BSND（= 参考 LayoutMode::BSNGD）；s1/s2/dTemplate 用动态 tiling
    // 分支（参考 S_TEMPLATE_UNKNOW → matmul::Matmul 四参形态，tiling 由 host 的
    // MatmulApiTiling 生成、GM 传入、REGIST_MATMUL_OBJ 注册）。
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, LayoutMode::BSNGD>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true, LayoutMode::BSNGD>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, float, false, LayoutMode::BNGS1S2>;
    using c1Type = MatmulType<TPosition::GM, CubeFormat::ND, T, false, LayoutMode::BNGS1S2>;
    matmul::Matmul<a1Type, b1Type, c1Type, bias1Type> bmm1;

    using a2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, LayoutMode::BNGS1S2>;
    using b2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, LayoutMode::BSNGD>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, float, false, LayoutMode::BSNGD>;
    using c2Type = MatmulType<TPosition::GM, CubeFormat::ND, T, false, LayoutMode::BNGS1S2>;
    matmul::Matmul<a2Type, b2Type, c2Type, bias2Type> bmm2;

    // D 非 16 对齐时 BMM2 的 NZ 输出形态（照搬参考 bmm2Nz）
    using c2NzType = MatmulType<TPosition::GM, CubeFormat::NZ, T, false, LayoutMode::BNGS1S2>;
    matmul::Matmul<a2Type, b2Type, c2NzType, bias2Type> bmm2Nz;

protected:
    __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                     __gm__ uint8_t *attenMask, __gm__ uint8_t *attentionOut,
                                     __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                     const SplitBTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void RefreshConstexpr();
    __aicore__ inline void InitBuffer();

    // ---------------- 资源与状态（照搬参考成员，裁剪 pse/dropout） ----------------
    int32_t blockIdx;
    const SplitBTilingData *__restrict tilingData;
    int64_t boIdx;
    int64_t currentN1Idx;
    TPipe *pipe;

    AscendC::GlobalTensor<INPUT_T> queryGm;
    AscendC::GlobalTensor<INPUT_T> keyGm;
    AscendC::GlobalTensor<INPUT_T> valueGm;
    AscendC::GlobalTensor<uint8_t> attenMaskGmInt;
    AscendC::GlobalTensor<INPUT_T> attentionOutGm;

    // 基本块（host tiling 算好，kernel 只读）
    uint32_t s1BaseSize;
    uint32_t s1BaseTailSize;
    uint32_t s2BaseSize;
    uint32_t dSize;
    uint32_t dBaseSize;
    uint32_t s1Size;
    uint32_t s2Size;

    // 轴乘积（照搬参考 ComputeConstexpr 的产物）
    int64_t s1S2;
    int64_t n2GS1D;
    int64_t n2S2D;
    int64_t gS1;
    int64_t n2D;
    int64_t n2G;
    int64_t n2GD;
    int64_t bN2G;
    int64_t n2GS1;
    int64_t bBaseSize;
    int64_t s1OuterSize;
    int64_t s1Vec2BaseSize;
    int64_t s1Vec2BaseTailSize;
    int64_t s1Vec2OuterSize;
    int64_t dSizeAlign16;
    int64_t biN2G;
    int64_t biN2GS1D;
    int64_t biN2GD;
    int64_t biN2S2D;
    int64_t biN2GS1;
    int64_t tensorABatchSize;   // = bBaseSize × N2 × G：BMM 的 A 侧 batch 数
    int64_t tensorBBatchSize;   // = bBaseSize × N2：B 侧 batch 数（GQA 广播）
};

// ============================ InitInput ============================
// TODO(P3-步2)：照搬参考 InitInput（bn2gs1s2_b.h:331-424）补齐 GM workspace 布局
// （mm1Res/mm2Res ping-pong、stage1Res、512B 对齐公式）。当前仅绑定输入指针。
template <typename INPUT_T, typename T, bool HAS_ATTEN, bool HAS_SOFTCAP>
__aicore__ inline void
FlashAttentionScoreSplitB<INPUT_T, T, HAS_ATTEN, HAS_SOFTCAP>::InitInput(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *attenMask,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
    const SplitBTilingData *__restrict tiling, TPipe *tPipe)
{
    this->blockIdx = AscendC::GetBlockIdx();
    this->pipe = tPipe;
    this->tilingData = tiling;

    this->queryGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
    this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
    this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value);
    this->attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
    this->attentionOutGm.SetGlobalBuffer((__gm__ INPUT_T *)attentionOut);
    (void)softmaxLse;
    (void)workspace;
}

// ============================ ComputeConstexpr ============================
// 照搬参考 bn2gs1s2_b.h:427-477（裁剪 pse/dropout 分支）
template <typename INPUT_T, typename T, bool HAS_ATTEN, bool HAS_SOFTCAP>
__aicore__ inline void FlashAttentionScoreSplitB<INPUT_T, T, HAS_ATTEN, HAS_SOFTCAP>::ComputeConstexpr()
{
    const auto &in = this->tilingData->inputParams;
    const auto &core = this->tilingData->coreParams;
    this->s1S2 = in.get_s1Size() * in.get_s2Size();
    this->s1Size = static_cast<uint32_t>(in.get_s1Size());
    this->s2Size = static_cast<uint32_t>(in.get_s2Size());
    this->dSize = static_cast<uint32_t>(in.get_dSize());
    this->dSizeAlign16 = (this->dSize + 15) / 16 * 16;
    this->dBaseSize = static_cast<uint32_t>(core.get_dBaseSize());
    this->bBaseSize = core.get_bBaseSize();
    int64_t gS1D = in.get_gSize() * in.get_s1Size() * in.get_dSize();
    int64_t gD = in.get_gSize() * in.get_dSize();
    this->n2D = in.get_n2Size() * in.get_dSize();
    this->n2G = in.get_n2Size() * in.get_gSize();
    this->n2GD = in.get_n2Size() * gD;
    this->bN2G = in.get_bSize() * in.get_n2Size() * in.get_gSize();
    this->n2GS1D = in.get_n2Size() * gS1D;
    this->n2S2D = in.get_n2Size() * in.get_s2Size() * in.get_dSize();
    this->gS1 = in.get_gSize() * in.get_s1Size();
    this->n2GS1 = in.get_n2Size() * this->gS1;
    this->s1BaseSize = static_cast<uint32_t>(core.get_s1BaseSize());
    this->s1BaseTailSize = static_cast<uint32_t>(core.get_s1BaseTailSize());
    this->s2BaseSize = static_cast<uint32_t>(core.get_s2BaseSize());
}

// ============================ RefreshConstexpr ============================
// 照搬参考 bn2gs1s2_b.h:479-512（尾核尾批的 bBaseSize 刷新 + batch 数重算）
template <typename INPUT_T, typename T, bool HAS_ATTEN, bool HAS_SOFTCAP>
__aicore__ inline void FlashAttentionScoreSplitB<INPUT_T, T, HAS_ATTEN, HAS_SOFTCAP>::RefreshConstexpr()
{
    const auto &in = this->tilingData->inputParams;
    const auto &core = this->tilingData->coreParams;
    if (this->blockIdx == this->tilingData->multiCoreParams.get_coreNum() - 1 &&
        this->boIdx == core.get_bOuterSize() - 1) {
        this->bBaseSize = core.get_bBaseTailSize();
    }
    this->biN2G = this->bBaseSize * in.get_n2Size() * in.get_gSize();
    this->biN2GS1D = this->bBaseSize * this->n2GS1D;
    this->biN2GD = this->bBaseSize * this->n2GD;
    this->biN2S2D = this->bBaseSize * this->n2S2D;
    this->biN2GS1 = this->bBaseSize * this->n2GS1;
    this->s1OuterSize = core.get_s1OuterSize();
    this->s1Vec2BaseSize = core.get_s1Vec2BaseSize();
    this->s1Vec2BaseTailSize = core.get_s1Vec2BaseTailSize();
    this->s1Vec2OuterSize = core.get_s1Vec2OuterSize();
    this->tensorABatchSize = this->bBaseSize * in.get_n2Size() * in.get_gSize();
    this->tensorBBatchSize = this->bBaseSize * in.get_n2Size();
}

// ============================ InitBuffer ============================
// TODO(P3-步2)：照搬参考 InitBuffer（bn2gs1s2_b.h:523-541）——stage1Ping/Pong 32K、
// commonTBuf 32K、mask 11K/16K、softmaxSum/Max 8K×4、vecOut 16K（含 pseTBuf/vecOut 复用）。
template <typename INPUT_T, typename T, bool HAS_ATTEN, bool HAS_SOFTCAP>
__aicore__ inline void FlashAttentionScoreSplitB<INPUT_T, T, HAS_ATTEN, HAS_SOFTCAP>::InitBuffer()
{
}

// ============================ Init / Process ============================
template <typename INPUT_T, typename T, bool HAS_ATTEN, bool HAS_SOFTCAP>
__aicore__ inline void
FlashAttentionScoreSplitB<INPUT_T, T, HAS_ATTEN, HAS_SOFTCAP>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *attenMask,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
    const SplitBTilingData *__restrict tiling, TPipe *tPipe)
{
    this->InitInput(query, key, value, attenMask, attentionOut, softmaxLse, workspace, tiling, tPipe);
    this->ComputeConstexpr();
    this->InitBuffer();
}

// TODO(P3-步2)：照搬参考 Process（bn2gs1s2_b.h:544-605）——3 槽 boIdx 软件流水：
//   WaitBmm1Result → IterateBmm1(boIdx+2) → ProcessVec1(boIdx+1) → WaitBmm2Result
//   → IterateBmm2(boIdx+1) → ProcessVec2(boIdx)
// 以及 IterateBmm1/2（SetTensorA/B + IterateBatch(tensorABatchSize, tensorBBatchSize)）、
// ProcessVec1（scale/[softcap]/mask/SoftmaxFlashV2/cast→P）、ProcessVec2（Div/搬出）。
template <typename INPUT_T, typename T, bool HAS_ATTEN, bool HAS_SOFTCAP>
__aicore__ inline void FlashAttentionScoreSplitB<INPUT_T, T, HAS_ATTEN, HAS_SOFTCAP>::Process()
{
    // 步 1 骨架：仅验证启动链路（blockIdx / batch 区间解析），无计算。
    const auto &mc = this->tilingData->multiCoreParams;
    int64_t multiCoreInnerOffset = static_cast<int64_t>(this->blockIdx) * mc.get_splitFactorSize();
    int64_t multiCoreInnerLimit = multiCoreInnerOffset + mc.get_splitFactorSize();
    if (mc.get_totalSize() < multiCoreInnerLimit) {
        multiCoreInnerLimit = mc.get_totalSize();
    }
    (void)multiCoreInnerOffset;  // TODO(P3-步2)：boIdx 循环起点
    (void)multiCoreInnerLimit;
}

} // namespace SplitB

// ============================ kernel 入口 ============================
// 照搬参考 flash_attention_score.cpp 的 INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ 展开：
// COPY_TILING_DATA + REGIST_MATMUL_OBJ + op.Init + op.Process。
// 差异：我们无 GE tiling 框架，tiling 结构由 host 直接 reinterpret 自 GM_ADDR。
//
// 混合核类型声明（照搬参考 :379）：matmul::Matmul 高阶 API 依赖 MIX_AIC_1_2
// （1 cube + 2 vector）的 KFC 基础设施（GetSysWorkSpacePtr）——缺它上板即 aicore 异常。
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
template <typename INPUT_T, bool HAS_ATTEN, bool HAS_SOFTCAP>
__global__ __aicore__ void FAInferSplitB(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attenMask,
                                         GM_ADDR attentionOut, GM_ADDR softmaxLse, GM_ADDR workspace,
                                         GM_ADDR tiling)
{
    TPipe tPipe;
    SplitB::FlashAttentionScoreSplitB<INPUT_T, float, HAS_ATTEN, HAS_SOFTCAP> op;
    // tiling 结构 GM→栈拷贝（参考由 GE 框架的 GET_TILING_DATA_WITH_STRUCT 完成；我们无框架，
    // 逐字节拷到局部结构后按普通指针使用——与 mha_fwd_kvcache 读 tiling 的 __gm__ 访问同机制）
    SplitB::SplitBTilingData tilingLocal;
    {
        const __gm__ uint8_t *src = reinterpret_cast<const __gm__ uint8_t *>(tiling);
        uint8_t *dst = reinterpret_cast<uint8_t *>(&tilingLocal);
        for (uint32_t i = 0; i < sizeof(SplitB::SplitBTilingData); ++i) {
            dst[i] = src[i];
        }
    }
    const SplitB::SplitBTilingData *__restrict tilingData = &tilingLocal;
    const AscendC::tiling::TCubeTiling *__restrict bmm1tiling = &tilingData->bmm1TilingData;
    const AscendC::tiling::TCubeTiling *__restrict bmm2tiling = &tilingData->bmm2TilingData;
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling,
                      op.bmm2Nz, bmm2tiling);
    op.Init(query, key, value, attenMask, attentionOut, softmaxLse, workspace, tilingData, &tPipe);
    op.Process();
}

#endif // MHA_FWD_SPLITB_CPP_

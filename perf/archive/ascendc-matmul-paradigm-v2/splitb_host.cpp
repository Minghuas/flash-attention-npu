/**
 * Copyright (c) 2026, perf-shortSeqLargeBatch project.
 *
 * SplitB host 侧：tiling 填充（照搬 CANN FlashAttentionScoreTilingB 公式）+
 * workspace 分配 + launch。仿 fag_general_host.cpp 的组织模式（可 include kernel 侧头）。
 *
 * 照搬来源：ops-transformer/attention/flash_attention_score/op_host/arch22/
 * flash_attention_score_tiling_general.cpp 的 FlashAttentionScoreTilingB 类（2932 行起）。
 * 公式对照与决策记录见 perf/design/splitb_integration.md（v2）与
 * perf/analysis/reference_splitb_deep_dive.md。
 */

#include "splitb_host.hpp"

#include <cmath>

#include "acl/acl.h"
#include "kernel_operator.h"
#include "tiling/platform/platform_ascendc.h"
// host 侧生成 bmm1/bmm2 的 cube tiling（对应参考 TilingB::SetBmm1/2TilingInput）
#include "lib/matmul/matmul_tiling.h"
#include "splitb_tilingdata.h"
#include "fwd_splitb_dispatch.hpp"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace SplitB {

namespace {

// ---------------- 照搬参考的常量（tiling_general.cpp 头部） ----------------
constexpr int64_t GM_ALIGN = 512;              // workspace 512B 对齐
constexpr int64_t BLOCK_B_UB_SIZE_LIMIT = 8 * 1024;  // blockBUBSizeLimit_（元素数）
constexpr int64_t FRACTAL_NUM = 16;
constexpr int64_t MAX_S1_BASE_SIZE = 256;      // maxS1BaseSize_

int64_t CeilDivI(int64_t a, int64_t b) { return (a + b - 1) / b; }
int64_t AlignUpI(int64_t a, int64_t b) { return CeilDivI(a, b) * b; }
int64_t AlignDownI(int64_t a, int64_t b) { return a / b * b; }

// 照搬 tiling_base.h CalcTschBlockDim：AIV 单位切片数 → AI Core block 数
uint32_t CalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum)
{
    if (aicCoreNum == 0 || aivCoreNum == 0 || aicCoreNum > aivCoreNum) {
        return sliceNum;
    }
    uint32_t ration = aivCoreNum / aicCoreNum;
    return (sliceNum + (ration - 1)) / ration;
}

// 照搬参考 flash_attention_score_common.h IsBasicBlockInSoftMax
bool IsBasicBlockInSoftMax(int32_t srcM, int32_t srcK)
{
    constexpr int32_t SOFTMAX_M_ALIGNED_SIZE = 8;
    constexpr int32_t SOFTMAX_K_ALIGNED_SIZE = 64;
    return srcM % SOFTMAX_M_ALIGNED_SIZE == 0 && srcK % SOFTMAX_K_ALIGNED_SIZE == 0;
}

}  // namespace

void mha_fwd_splitb(at::Tensor &q, const at::Tensor &k, const at::Tensor &v, at::Tensor &out,
                    at::Tensor &softmaxlse, const at::Tensor &mask, float softmax_scale,
                    bool is_causal, bool is_local, int64_t window_size_left,
                    int64_t window_size_right, float softcap)
{
    const auto sizes = q.sizes();
    const int64_t B = sizes[0];
    const int64_t Sq = sizes[1];
    const int64_t H = sizes[2];
    const int64_t D = sizes[3];
    const int64_t Sk = k.size(1);
    const int64_t Hkv = k.size(2);
    const int64_t N2 = Hkv;         // kv 头数
    const int64_t G = H / Hkv;      // GQA 组数
    const bool isBf16 = q.dtype() == at::kBFloat16;
    const int64_t dtypeSize = isBf16 ? 2 : 2;  // fp16/bf16 均 2 字节

    // ---------------- 平台参数 ----------------
    // 注意：PlatformAscendCManager::GetInstance() 返回 PlatformAscendC*（管理器单例即平台对象）
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t aicNum = platform->GetCoreNumAic();
    const uint32_t aivNum = platform->GetCoreNumAiv();
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);

    // ---------------- 基本块（照搬 TilingB::CalcS1S2BasicBlock / SetCoreParams） ----------------
    const int64_t alignedS1 = AlignUpI(Sq, FRACTAL_NUM);
    const int64_t alignedS2 = AlignUpI(Sk, FRACTAL_NUM);
    const int64_t alignedD = AlignUpI(D, FRACTAL_NUM);

    const int64_t s2BasicBlock = alignedS2;  // S2 不切分（核心）
    int64_t s1BasicBlock = AlignDownI(BLOCK_B_UB_SIZE_LIMIT / s2BasicBlock, FRACTAL_NUM);
    s1BasicBlock = std::min(s1BasicBlock, alignedS1);
    s1BasicBlock = std::min(s1BasicBlock, MAX_S1_BASE_SIZE);
    // s1Vec2 基本块（照搬：dVec2 ≤ 上限走 16 对齐分支；fp16/bf16 的 2/inputDtypeBytes = 1）
    int64_t s1Vec2BasicBlock = AlignDownI(BLOCK_B_UB_SIZE_LIMIT / alignedD, FRACTAL_NUM) * 2 / dtypeSize;
    s1Vec2BasicBlock = std::min(s1Vec2BasicBlock, alignedS1);

    const int64_t bBaseSize = 1;             // 每个 boIdx = 1 个 batch 的全部头
    const int64_t bOuterSize = B;

    // ---------------- tiling 结构填充 ----------------
    SplitBTilingData tiling;
    std::memset(&tiling, 0, sizeof(tiling));

    auto &in = tiling.inputParams;
    in.set_bSize(B);
    in.set_n2Size(N2);
    in.set_gSize(G);
    in.set_s1Size(Sq);
    in.set_s2Size(Sk);
    in.set_alignedS2(alignedS2);
    in.set_dSize(D);
    in.set_scaleValue(softmax_scale);
    in.set_softcapValue(softcap);
    in.set_attenMaskS2Size(static_cast<uint32_t>(alignedS2));
    in.set_attenMaskShapeType(2);   // (1,1,1,S1,S2)：跨 batch/head 共享一个 [S1,S2] mask
    in.set_attenMaskCompressMode(0); // NO_COMPRESS
    in.set_isCausalFlag(is_causal ? 1 : 0);
    in.set_layoutType(0);           // v1 固定 BSND

    auto &core = tiling.coreParams;
    core.set_s1BaseSize(static_cast<int32_t>(s1BasicBlock));
    core.set_s1BaseTailSize(static_cast<int32_t>(Sq - (CeilDivI(Sq, s1BasicBlock) - 1) * s1BasicBlock));
    core.set_s1OuterSize(CeilDivI(Sq, s1BasicBlock));
    core.set_s1Vec2BaseSize(static_cast<int32_t>(s1Vec2BasicBlock));
    core.set_s1Vec2BaseTailSize(static_cast<int32_t>(Sq - (CeilDivI(Sq, s1Vec2BasicBlock) - 1) * s1Vec2BasicBlock));
    core.set_s1Vec2OuterSize(CeilDivI(Sq, s1Vec2BasicBlock));
    core.set_s2BaseSize(static_cast<int32_t>(s2BasicBlock));
    core.set_s2BaseTailSize(static_cast<int32_t>(Sk));
    core.set_s2OuterSize(1);
    core.set_dBaseSize(static_cast<int32_t>(alignedD));
    core.set_dBaseTailSize(static_cast<int32_t>(D));
    core.set_dOuterSize(1);
    core.set_bBaseSize(static_cast<int32_t>(bBaseSize));
    core.set_bBaseTailSize(static_cast<int32_t>(bBaseSize));
    core.set_bOuterSize(bOuterSize);

    // ---------------- bmm1/bmm2 cube tiling（照搬 TilingB::SetBmm1/2TilingInput） ----------------
    // batch 数 = bBaseSize × N2 × G（A 侧 batch，B 侧由 Layout 5 元组广播）
    const int64_t batchNum = bBaseSize * N2 * G;
    const matmul_tiling::DataType bmmDtype =
        isBf16 ? matmul_tiling::DataType::DT_BF16 : matmul_tiling::DataType::DT_FLOAT16;

    matmul_tiling::MatmulApiTiling bmm1(*platform);
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
    bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_FLOAT);  // T 恒 float（照搬 B 模板 "half, float"）
    bmm1.SetShape(static_cast<int32_t>(Sq), static_cast<int32_t>(Sk), static_cast<int32_t>(D));
    bmm1.SetOrgShape(static_cast<int32_t>(Sq), static_cast<int32_t>(Sk), static_cast<int32_t>(D),
                     static_cast<int32_t>(D));  // [需验证] 参考传 s1StrideSize/s2StrideSize，待步2核对语义
    bmm1.SetALayout(static_cast<int32_t>(bBaseSize), static_cast<int32_t>(Sq), static_cast<int32_t>(N2),
                    static_cast<int32_t>(G), static_cast<int32_t>(D));
    bmm1.SetBLayout(static_cast<int32_t>(bBaseSize), static_cast<int32_t>(Sk), static_cast<int32_t>(N2),
                    1, static_cast<int32_t>(D));
    bmm1.SetCLayout(static_cast<int32_t>(bBaseSize), static_cast<int32_t>(Sq), static_cast<int32_t>(N2),
                    static_cast<int32_t>(G), static_cast<int32_t>(Sk));
    bmm1.SetBatchNum(static_cast<int32_t>(batchNum));
    bmm1.SetBias(false);
    bmm1.SetBufferSpace(static_cast<int32_t>(l1Size), static_cast<int32_t>(l0cSize));
    bmm1.SetFixSplit(static_cast<int32_t>(s1BasicBlock), static_cast<int32_t>(s2BasicBlock));
    TORCH_CHECK(bmm1.GetTiling(tiling.bmm1TilingData) != -1, "splitb: bmm1 GetTiling failed");

    matmul_tiling::MatmulApiTiling bmm2(*platform);
    // [需验证] 参考此处 A=VECCALC/NZ（P 由 vector 侧写入 workspace）；与 kernel 侧 a2Type(GM/ND)
    // 的对应关系待步 2 运行期核对，先照抄参考 host 侧设置。
    bmm2.SetAType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::NZ, bmmDtype, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
    bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_FLOAT);
    bmm2.SetShape(static_cast<int32_t>(Sq), static_cast<int32_t>(alignedD), static_cast<int32_t>(Sk));
    bmm2.SetOrgShape(static_cast<int32_t>(Sq), static_cast<int32_t>(alignedD), static_cast<int32_t>(Sk),
                     static_cast<int32_t>(Sk));
    bmm2.SetALayout(static_cast<int32_t>(bBaseSize), static_cast<int32_t>(Sq), static_cast<int32_t>(N2),
                    static_cast<int32_t>(G), static_cast<int32_t>(Sk));
    bmm2.SetBLayout(static_cast<int32_t>(bBaseSize), static_cast<int32_t>(Sk), static_cast<int32_t>(N2),
                    1, static_cast<int32_t>(D));
    bmm2.SetCLayout(static_cast<int32_t>(bBaseSize), static_cast<int32_t>(Sq), static_cast<int32_t>(N2),
                    static_cast<int32_t>(G), static_cast<int32_t>(D));
    bmm2.SetBatchNum(static_cast<int32_t>(batchNum));
    bmm2.SetBias(false);
    bmm2.SetBufferSpace(static_cast<int32_t>(l1Size), static_cast<int32_t>(l0cSize));
    const int64_t maxDBasicBlock = AlignDownI(static_cast<int64_t>(l0cSize) / (s1BasicBlock * 4), 16);
    bmm2.SetFixSplit(static_cast<int32_t>(s1BasicBlock),
                     static_cast<int32_t>(std::min(maxDBasicBlock, alignedD)));
    TORCH_CHECK(bmm2.GetTiling(tiling.bmm2TilingData) != -1, "splitb: bmm2 GetTiling failed");

    // TODO(P3-步2)：softmaxFlashTilingData —— 参考用 SoftMaxFlashV2TilingFunc（依赖 ge::Shape，
    // torch 扩展无 GE 框架）。步 2 从 asc/impl/adv_api/detail/activation/softmax/
    // softmax_flashv2_base_impl.h 的可见实现手工翻译。当前 memset 0（kernel 骨架不使用）。

    // ---------------- 核间切分（照搬 TilingB::SetMultiCoreParams） ----------------
    const int64_t totalSize = bOuterSize;
    const int64_t usedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
    const int64_t splitFactorSize = CeilDivI(totalSize, usedAivNum);
    const int64_t coreNum = CeilDivI(totalSize, splitFactorSize);
    const uint32_t blockDim = CalcTschBlockDim(static_cast<uint32_t>(coreNum), aicNum, aivNum);

    auto &mc = tiling.multiCoreParams;
    mc.set_coreNum(static_cast<int32_t>(coreNum));
    mc.set_totalSize(totalSize);
    mc.set_splitFactorSize(splitFactorSize);
    mc.set_splitFactorTailSize(totalSize - (coreNum - 1) * splitFactorSize);

    // ---------------- workspace（照搬 TilingB::GetWorkspaceSize 公式，T=float） ----------------
    const int64_t n2g = N2 * G;
    const int64_t calcTypeBytes = 4;  // T = float
    const int64_t mm1PerBytes = AlignUpI(bBaseSize * n2g * Sq * s2BasicBlock * calcTypeBytes, GM_ALIGN);
    const int64_t mm2PerBytes = AlignUpI(bBaseSize * n2g * Sq * alignedD * calcTypeBytes, GM_ALIGN);
    const int64_t stage1PerBytes = AlignUpI(bBaseSize * n2g * Sq * s2BasicBlock * dtypeSize, GM_ALIGN);
    const int64_t perCoreBytes = mm1PerBytes * 2 + stage1PerBytes * 2 + mm2PerBytes * 2;
    const int64_t workSpaceSize = perCoreBytes * coreNum;
    at::Tensor workspace_tensor =
        at::empty({workSpaceSize}, at::device(at::kPrivateUse1).dtype(at::kByte));

    // ---------------- tiling 拷贝到 device ----------------
    at::Tensor tiling_cpu_tensor =
        at::empty({static_cast<int64_t>(sizeof(SplitBTilingData))}, at::device(at::kCPU).dtype(at::kByte));
    std::memcpy(tiling_cpu_tensor.data_ptr<uint8_t>(), &tiling, sizeof(SplitBTilingData));
    at::Tensor tiling_gpu_tensor = tiling_cpu_tensor.to(at::Device(at::kPrivateUse1));

    // ---------------- launch ----------------
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    FwdLaunchArgs fwd_args;
    fwd_args.blockDim = blockDim;
    fwd_args.aclStream = aclStream;
    fwd_args.fftsAddr = 0;  // SplitB 不使用 CrossCoreFlag/ffts 同步（matmul API 内建同步）
    fwd_args.is_bf16 = isBf16;
    fwd_args.paged_KV = false;
    fwd_args.is_causal = is_causal;
    fwd_args.is_local = is_local;
    fwd_args.flashDecodeFlag = false;
    fwd_args.has_softcap = (softcap > 0.0f);
    fwd_args.qDevice = static_cast<uint8_t *>(q.data_ptr());
    fwd_args.kDevice = static_cast<uint8_t *>(k.data_ptr());
    fwd_args.vDevice = static_cast<uint8_t *>(v.data_ptr());
    fwd_args.maskDevice = mask.defined() ? static_cast<uint8_t *>(mask.data_ptr()) : nullptr;
    fwd_args.blockTableDevice = nullptr;
    fwd_args.oDevice = static_cast<uint8_t *>(out.data_ptr());
    fwd_args.softmaxLseDevice = static_cast<uint8_t *>(softmaxlse.data_ptr());
    fwd_args.qSeqDevice = nullptr;
    fwd_args.kvSeqDevice = nullptr;
    fwd_args.workspaceDevice = static_cast<uint8_t *>(workspace_tensor.data_ptr());
    fwd_args.tilingDevice = static_cast<uint8_t *>(tiling_gpu_tensor.data_ptr());
    (void)window_size_left;
    (void)window_size_right;
    launch_fwd_splitb(fwd_args);
}

}  // namespace SplitB

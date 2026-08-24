/**
 * Copyright (c) 2026, perf-shortSeqLargeBatch project.
 *
 * SplitB host 侧（v3 catlass 路线）：tiling 填充（照搬 CANN FlashAttentionScoreTilingB
 * 公式，机制层适配 FAInfer 范式）+ workspace 分配 + launch。
 *
 * v3 相对范式 A 归档版（perf/archive/ascendc-matmul-paradigm-v2/）的三处变更：
 *   1. 删 MatmulApiTiling 生成段——catlass BlockMmad 自管 L1/L0，无需 host cube tiling
 *   2. 核间切分 aic 基数（FAInfer 同款），非参考的 aiv 基数
 *   3. 补 fftsAddr（CrossCoreFlag 依赖，照抄 flash_api.cpp 的 rtGetC2cCtrlAddr）
 * 设计规范：perf/design/splitb_integration.md（v3 §4）
 */

#include "splitb_host.hpp"

#include <cmath>

#include "acl/acl.h"
#include "kernel_operator.h"
#include "tiling/platform/platform_ascendc.h"
#include "splitb_tilingdata.h"
#include "fwd_splitb_dispatch.hpp"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "runtime/rt_ffts.h"  // rtGetC2cCtrlAddr（CrossCoreFlag 的 ffts 同步基址）

namespace SplitB {

// 构建标识（每次结构改动更新；运行时打印以确认二进制版本——devlog #22 溯源教训）
constexpr const char *SPLITB_BUILD_TAG = "printf-v3 (devlog #42)";

namespace {

// ---------------- 照搬参考的常量（tiling_general.cpp 头部） ----------------
constexpr int64_t GM_ALIGN = 512;                    // workspace 512B 对齐
constexpr int64_t BLOCK_B_UB_SIZE_LIMIT = 8 * 1024;  // blockBUBSizeLimit_（元素数）
constexpr int64_t FRACTAL_NUM = 16;
constexpr int64_t MAX_S1_BASE_SIZE = 128;            // 参考为 256；catlass L1Tile M=Q_TILE_CEIL=128 硬约束（因地制宜）
constexpr int64_t DEFAULT_BLOCK_SIZE = 128;          // paged KV 页大小占位（非 paged 穿透用，FAInfer tiling.blockSize 同源）

int64_t CeilDivI(int64_t a, int64_t b) { return (a + b - 1) / b; }
int64_t AlignUpI(int64_t a, int64_t b) { return CeilDivI(a, b) * b; }
int64_t AlignDownI(int64_t a, int64_t b) { return a / b * b; }

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
    const int64_t dtypeSize = 2;    // fp16/bf16
    printf("222 [splitb] ENTER mha_fwd_splitb (build=%s)\n", SPLITB_BUILD_TAG); fflush(stdout);

    // ---------------- 平台参数（aic 基数） ----------------
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t aicNum = platform->GetCoreNumAic();
    printf("333 [splitb] platform ok aicNum=%u\n", aicNum); fflush(stdout);

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
    printf("444 [splitb] blocks ok s1Base=%lld\n", (long long)s1BasicBlock); fflush(stdout);

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
    in.set_blockSize(static_cast<uint32_t>(DEFAULT_BLOCK_SIZE));
    in.set_scaleValue(softmax_scale);
    in.set_softcapValue(softcap);
    in.set_windowSizeLeft(window_size_left);
    in.set_windowSizeRight(window_size_right);
    in.set_isCausalFlag(is_causal ? 1 : 0);
    const bool dbgEnv = getenv("FLASH_ATTN_SPLITB_DEBUG") != nullptr;
    in.set_debugFlag(dbgEnv ? 1 : 0);
    const bool smOnlyEnv = getenv("FLASH_ATTN_SPLITB_SOFTMAX_ONLY") != nullptr;
    in.set_softmaxOnly(smOnlyEnv ? 1 : 0);
    const bool dumpEnv = getenv("FLASH_ATTN_SPLITB_DUMP") != nullptr;
    in.set_dumpFlag(dumpEnv ? 1 : 0);


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

    // ---------------- 核间切分（aic 基数；参考为 aiv 基数——D7 因地制宜项） ----------------
    const int64_t totalSize = bOuterSize;
    // debug/dump 均强制单核：避免多核 printf/DumpTensor 输出串扰（用户要求）
    const int64_t usedCoreNum = (dbgEnv || dumpEnv || smOnlyEnv) ? 1 : std::min(totalSize, static_cast<int64_t>(aicNum));
    const int64_t splitFactorSize = CeilDivI(totalSize, usedCoreNum);
    const int64_t coreNum = CeilDivI(totalSize, splitFactorSize);

    printf("555 [splitb] tiling filled\n"); fflush(stdout);
    auto &mc = tiling.multiCoreParams;
    mc.set_coreNum(static_cast<int32_t>(coreNum));
    mc.set_totalSize(totalSize);
    mc.set_splitFactorSize(splitFactorSize);
    mc.set_splitFactorTailSize(totalSize - (coreNum - 1) * splitFactorSize);

    // ---------------- workspace（与 kernel perTileF/perBatchF 布局严格一致，见 mha_fwd_splitb.cpp） ----------------
    // 每核 [batchBuf0: nTilePerBatch × tile 块 + P-scratch | batchBuf1: 同构]；每 tile 块（float 计）：
    //   S 区: ROW_NUM_MAX(=Q_TILE_CEIL=128) × colsPad（fp32）
    //   OTmp 区: ROW_NUM_MAX × dPad
    //   stats:   2 × ROW_NUM_MAX（max+sum 各 128 行距）
    // P（fp16）独立区（devlog #44.35 临时调试布局，修 bug 后回归 #44.23 链式）：
    //   批尾连续 T 个独立 P 槽，P 与 S 完全不复用（S 自 QK 写出后永无人覆写）
    // tile 数 = CeilDiv(G, qNBlockTile) × N2 × CeilDiv(Sq, 128)（与 kernel GetQNBlockTile/
    //   GetQSBlockTile 公式严格一致——两者独立计算，改动必须同步，devlog #34）
    const int64_t rowNumMax = 128;                            // = Q_TILE_CEIL（kernel_common.hpp）
    const int64_t colsPad = alignedS2;                        // align16(Sk)
    const int64_t dPad = AlignUpI(D, FRACTAL_NUM);            // align16(D)
    const int64_t s1AreaF = rowNumMax * colsPad;              // floats
    const int64_t pScratchF = s1AreaF / 2;                     // 单 P 槽（128×colsPad half；#44.35 起 T 槽全独立）
    const int64_t oAreaF = rowNumMax * dPad;
    const int64_t statsPerTask = 2 * rowNumMax;
    const int64_t qNBlockTile = std::min(std::max((rowNumMax / Sq) / 2 * 2, (int64_t)1), G);
    const int64_t nTilePerBatch = CeilDivI(G, qNBlockTile) * N2 * CeilDivI(Sq, rowNumMax);
    const int64_t perTileF = s1AreaF + oAreaF + statsPerTask;
    const int64_t perBatchF = nTilePerBatch * (perTileF + pScratchF); // T×(tile 块+独立 P 槽)（#44.35 调试；回归链式改 + pScratchF）
    const int64_t perCoreF = 2 * perBatchF;
    const int64_t perCoreBytes = AlignUpI(perCoreF * 4, GM_ALIGN);
    const int64_t workSpaceSize = perCoreBytes * coreNum;
    (void)dtypeSize;
    if (dbgEnv) {
        printf("666 [splitb host] B=%lld Sq=%lld Sk=%lld H=%lld Hkv=%lld D=%lld | qNBlockTile=%lld "
               "nTilePerBatch=%lld colsPad=%lld coreNum=%lld splitF=%lld wsBytes=%lld\n",
               (long long)B, (long long)Sq, (long long)Sk, (long long)H, (long long)Hkv, (long long)D,
               (long long)qNBlockTile, (long long)nTilePerBatch, (long long)alignedS2,
               (long long)coreNum, (long long)splitFactorSize, (long long)workSpaceSize);
        fflush(stdout);
    }
    at::Tensor workspace_tensor =
        at::empty({workSpaceSize}, at::device(at::kPrivateUse1).dtype(at::kByte));
    printf("777 [splitb] ws alloc ok %lld bytes\n", (long long)workSpaceSize); fflush(stdout);

    // ---------------- tiling 拷贝到 device ----------------
    at::Tensor tiling_cpu_tensor =
        at::empty({static_cast<int64_t>(sizeof(SplitBTilingData))}, at::device(at::kCPU).dtype(at::kByte));
    std::memcpy(tiling_cpu_tensor.data_ptr<uint8_t>(), &tiling, sizeof(SplitBTilingData));
    at::Tensor tiling_gpu_tensor = tiling_cpu_tensor.to(at::Device(at::kPrivateUse1));
    printf("888 [splitb] tiling H2D ok\n"); fflush(stdout);

    // ---------------- ffts 同步基址（CrossCoreFlag 依赖，照抄 flash_api.cpp:734-736） ----------------
    uint64_t fftsAddr = 0;
    uint32_t fftsLen = 0;
    rtError_t rtErr = rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    TORCH_CHECK(rtErr == 0, "splitb: rtGetC2cCtrlAddr failed, ret=", rtErr);
    printf("999 [splitb] ffts ok addr=%llx len=%u\n", (unsigned long long)fftsAddr, fftsLen); fflush(stdout);

    // ---------------- launch ----------------
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    FwdLaunchArgs fwd_args;
    fwd_args.blockDim = static_cast<uint32_t>(coreNum);
    fwd_args.aclStream = aclStream;
    fwd_args.fftsAddr = fftsAddr;
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
    printf("1000 [splitb] pre-launch blockDim=%u dtype=%s mask(c=%d,l=%d) sc=%d\n",
           fwd_args.blockDim, isBf16 ? "bf16" : "fp16", (int)fwd_args.is_causal,
           (int)fwd_args.is_local, (int)fwd_args.has_softcap); fflush(stdout);
    launch_fwd_splitb(fwd_args);
    printf("9999 [splitb] launch ENQUEUED (async)\n"); fflush(stdout);
}

}  // namespace SplitB

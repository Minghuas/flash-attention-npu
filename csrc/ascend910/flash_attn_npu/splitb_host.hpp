/**
 * Copyright (c) 2026, perf-shortSeqLargeBatch project.
 *
 * SplitB 路径的 host 侧入口（轻量声明头，flash_api.cpp include，不拖入重依赖）。
 * 实现在 splitb_host.cpp（仿 fag_general_host.cpp 模式：可 include kernel 侧头）。
 *
 * 触发条件照搬 CANN TilingB::IsCapable（用户拍板严格照搬，不放宽）：
 *   alignedS2 ≤ 128  且  N2×G×alignedS1×alignedS2×dtypeBytes ≤ 128KB
 * v1 功能约束（未适配特性回退旧路径，非优化选择）：dropout=0、不要求 return_softmax。
 * 小 B 不回落（照搬参考，SplitB 是 fallback 模板）。
 */

#ifndef SPLITB_HOST_HPP
#define SPLITB_HOST_HPP

#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstdint>

#include <torch/extension.h>  // at::Tensor（与 flash_api.cpp 的 include 约定一致）

namespace SplitB {

// 触发条件判定（纯函数，无重依赖）
inline bool should_use(int64_t seqlen_q, int64_t seqlen_k, int64_t num_heads,
                       size_t dtype_size, float p_dropout, bool return_softmax)
{
    // v1 功能约束：未适配特性回退旧路径
    if (p_dropout != 0.0f || return_softmax) {
        return false;
    }
    // 照搬 TilingB::IsCapable：
    //   HIGH_PERF_SUPPORT_S2_BASIC = 128；blockBSizeLimit_ = 64K 元素 × 2B
    int64_t alignedS1 = (seqlen_q + 15) / 16 * 16;
    int64_t alignedS2 = (seqlen_k + 15) / 16 * 16;
    if (alignedS2 > 128) {
        return false;
    }
    int64_t n2g = num_heads;  // N2×G = q 头数（N2=kv头 × G=组数）
    if (n2g * alignedS1 * alignedS2 * static_cast<int64_t>(dtype_size) > 128 * 1024) {
        return false;
    }
    return true;
}

// 环境变量开关（调试/灰度用）：
//   FLASH_ATTN_DISABLE_SPLITB=1 —— 强制关闭（A/B 对比）
//   FLASH_ATTN_FORCE_SPLITB=1   —— 无视条件强制开启（测试非触发形状）
// P3 步 1：默认关闭（kernel 为骨架），步 2 完成后翻为默认开启。
inline bool env_enabled()
{
    static const bool disabled = getenv("FLASH_ATTN_DISABLE_SPLITB") != nullptr;
    static const bool forced = getenv("FLASH_ATTN_FORCE_SPLITB") != nullptr;
    if (disabled) {
        return false;
    }
    return forced;  // 步 1 默认 false；步 2 改为 return true;
}

// SplitB 路径执行：tiling 填充（照搬 TilingB 公式）+ workspace 分配 + launch。
// mask 形参：causal/SWA 时 flash_api.cpp 已生成的 2048×2048 triu 表（可传空 tensor 表示无 mask）。
void mha_fwd_splitb(at::Tensor &q, const at::Tensor &k, const at::Tensor &v, at::Tensor &out,
                    at::Tensor &softmaxlse, const at::Tensor &mask, float softmax_scale,
                    bool is_causal, bool is_local, int64_t window_size_left,
                    int64_t window_size_right, float softcap);

}  // namespace SplitB

#endif  // SPLITB_HOST_HPP

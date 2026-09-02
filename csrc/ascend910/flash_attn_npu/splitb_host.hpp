/**
 * Copyright (c) 2026, perf-shortSeqLargeBatch project.
 *
 * SplitB 路径的 host 侧入口（轻量声明头，flash_api.cpp include，不拖入重依赖）。
 * 实现在 splitb_host.cpp（仿 fag_general_host.cpp 模式：可 include kernel 侧头）。
 *
 * 触发条件 = 功能支持面（硬约束）∩ 形状闸门（照搬 CANN TilingB::IsCapable 两条件
 * + 本 kernel 验证边界）。小 B 不回落（照搬参考，SplitB 是 fallback 模板）。
 * S3 完成 + 多核全绿（devlog #44.47）后默认启用；未支持特性回退旧路径。
 */

#ifndef SPLITB_HOST_HPP
#define SPLITB_HOST_HPP

#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstdint>

#include <torch/extension.h>  // at::Tensor（与 flash_api.cpp 的 include 约定一致）

namespace SplitB {

// 功能支持面（硬约束：kernel 未实现的特性绝不路由，FORCE 也不绕过——未实现分支
// 进入即静默错数，而非 merely 未优化）。S4 完成后逐项放开：
//   - causal/SWA：kernel 内仅 NO_MASK 分支有计算体（MASK_CAUSAL/MASK_SWA 实例化
//     但为空壳），mask 穿透 softmax 属 S4；
//   - softcap：softmax 的 ApplySoftcap 已照抄（splitb_softmax.hpp:252）但未经
//     上板验证，验证前不路由；
//   - dropout/return_softmax：参考实现中 dropout 由更高优先级模板（DropMask=90）
//     接管、本模板天然不触达；return_softmax 本 kernel 无输出通道。
inline bool features_supported(bool is_causal, bool is_local, float softcap,
                               float p_dropout, bool return_softmax)
{
    return !is_causal && !is_local && softcap <= 0.0f &&
           p_dropout == 0.0f && !return_softmax;
}

// 形状闸门（照搬 TilingB::IsCapable 两条件 + 本 kernel 验证边界）。
inline bool shape_supported(int64_t seqlen_q, int64_t seqlen_k, int64_t num_heads,
                            int64_t head_size, size_t dtype_size)
{
    // 退化形状（空 Q/KV）回退旧路径，不进入本 kernel
    if (seqlen_q <= 0 || seqlen_k <= 0) {
        return false;
    }
    // 主条件：max(Sq, Sk) ≤ 128。参考条件 alignedS2≤128 的收紧版——同时覆盖
    // alignedS1（本 kernel Q 单 tile，Q_TILE_CEIL=128；Sq>128 跨 tile 未验证）
    if (seqlen_q > 128 || seqlen_k > 128) {
        return false;
    }
    // D ≤ 128：workspace OTmp 区 128×align16(D) fp32 的设计上限（D>128 未验证）
    if (head_size > 128) {
        return false;
    }
    // [v3.4] 放宽：去掉了参考实现的 N2×G×alignedS1×alignedS2×dtype ≤ 128KB 闸门。
    // 该条是 TilingB::IsCapable 为 BATCH_LESS_THAN_L1（batch 数据驻留 L1）设的预算——
    // 我们的 Pingpong 引擎逐 tile 装载（L1 用量与 S 无关，每 tile 就是 128×128 级），
    // 不吃这个约束。保留的硬约束仅为 max(Sq,Sk)≤128（引擎 M/N 维 L1 装载上限）+ D≤128。
    // 效果：H24/s64（192KB）、H8/s128（256KB）、H24/s128（768KB）等形状进入 SplitB 覆盖。
    return true;
}

// 触发条件判定（纯函数，无重依赖；功能面 ∩ 形状闸门）
inline bool should_use(int64_t seqlen_q, int64_t seqlen_k, int64_t num_heads,
                       int64_t head_size, size_t dtype_size, float p_dropout,
                       bool return_softmax, bool is_causal, bool is_local, float softcap)
{
    return features_supported(is_causal, is_local, softcap, p_dropout, return_softmax) &&
           shape_supported(seqlen_q, seqlen_k, num_heads, head_size, dtype_size);
}

// 路由决策（flash_api 各前向接口调用；env 开关在此读一次静态缓存）：
//   FLASH_ATTN_DISABLE_SPLITB —— 强制关闭（A/B 对比）
//   FLASH_ATTN_FORCE_SPLITB   —— 绕过形状闸门（测试非触发形状；功能面不绕过）
// 默认启用（S3 完成 + 多核全绿，devlog #44.47；骨架期曾默认关）。
inline bool route_splitb(int64_t seqlen_q, int64_t seqlen_k, int64_t num_heads,
                         int64_t head_size, size_t dtype_size, float p_dropout,
                         bool return_softmax, bool is_causal, bool is_local, float softcap)
{
    if (!features_supported(is_causal, is_local, softcap, p_dropout, return_softmax)) {
        return false;
    }
    static const bool disabled = getenv("FLASH_ATTN_DISABLE_SPLITB") != nullptr;
    if (disabled) {
        return false;
    }
    static const bool forced = getenv("FLASH_ATTN_FORCE_SPLITB") != nullptr;
    if (forced) {
        return true;
    }
    return shape_supported(seqlen_q, seqlen_k, num_heads, head_size, dtype_size);
}

// SplitB 路径执行：tiling 填充（照搬 TilingB 公式）+ workspace 分配 + launch。
// mask 形参：causal/SWA 时 flash_api.cpp 已生成的 2048×2048 triu 表（可传空 tensor 表示无 mask）。
void mha_fwd_splitb(at::Tensor &q, const at::Tensor &k, const at::Tensor &v, at::Tensor &out,
                    at::Tensor &softmaxlse, const at::Tensor &mask, float softmax_scale,
                    bool is_causal, bool is_local, int64_t window_size_left,
                    int64_t window_size_right, float softcap);

}  // namespace SplitB

#endif  // SPLITB_HOST_HPP

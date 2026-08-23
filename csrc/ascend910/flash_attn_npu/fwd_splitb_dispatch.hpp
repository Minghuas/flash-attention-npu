/**
 * Copyright (c) 2026, perf-shortSeqLargeBatch project.
 *
 * SplitB 前向 dispatch：轻量声明头（flash_api.cpp / splitb_host.cpp 可 include，
 * 不拖入重 kernel 模板）。复用 fwd_dispatch.hpp 的 FwdLaunchArgs 结构体；
 * 模板实例化在 autogen/fwd_dispatch_{fp16,bf16}_splitb.cpp（generate_kernels.py 生成），
 * impl 体在 fwd_splitb_dispatch_impl.hpp（由 autogen TU include mha_fwd_splitb.cpp 后展开）。
 */

#ifndef FWD_SPLITB_DISPATCH_HPP
#define FWD_SPLITB_DISPATCH_HPP

#pragma once

#include "fwd_dispatch.hpp"  // FwdLaunchArgs（复用，不新增实体）

// 运行时入口：按 dtype 分派到 autogen TU 的模板实例。
// 模板树轴：HAS_ATTEN = (is_causal || is_local) × HAS_SOFTCAP，T 恒 float。
// 定义 inline 在本轻量头（与 fwd_dispatch.hpp 的 launch_fwd 同模式）：调用方 TU
// （splitb_host.o）发射符号，launch_fwd_splitb_impl<DType> 在链接期从 autogen TU 解析。
template <typename DType>
void launch_fwd_splitb_impl(const FwdLaunchArgs &a);

inline void launch_fwd_splitb(const FwdLaunchArgs &a) {
    if (a.is_bf16) {
        launch_fwd_splitb_impl<bfloat16_t>(a);
    } else {
        launch_fwd_splitb_impl<half>(a);
    }
}

#endif  // FWD_SPLITB_DISPATCH_HPP

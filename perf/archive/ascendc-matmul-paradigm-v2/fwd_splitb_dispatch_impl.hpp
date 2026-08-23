/**
 * Copyright (c) 2026, perf-shortSeqLargeBatch project.
 *
 * SplitB 前向 dispatch 的共享 impl 体。由 autogen/fwd_dispatch_{fp16,bf16}_splitb.cpp
 * include（每个 TU 显式实例化一个 launch_fwd_splitb_impl<DType>），kernel 模板
 * （FAInferSplitB）落在独立 TU 并行编译——与 fwd_dispatch_impl.hpp 的组织方式一致。
 */

#pragma once

#include "fwd_splitb_dispatch.hpp"

// 标准头（kernel 文件假定可见，与 fwd_dispatch_impl.hpp 同款注释）
#include <algorithm>
#include <cstring>
#include <limits>

// FAInferSplitB kernel 模板与本目录的 SplitBTilingData
#include "mha_fwd_splitb.cpp"

template <typename DType>
void launch_fwd_splitb_impl(const FwdLaunchArgs &a) {
    const uint32_t blockDim = a.blockDim;
    const aclrtStream aclStream = a.aclStream;
    const bool has_atten = a.is_causal || a.is_local;
    const bool has_softcap = a.has_softcap;
    // SplitB v1：无 paged / FD / TND 轴（触发条件天然排除；dropout 在 host 侧拦截）
    (void)a.paged_KV;
    (void)a.flashDecodeFlag;

    if (has_atten) {
        if (has_softcap) {
            FAInferSplitB<DType, true, true>
                <<<blockDim, nullptr, aclStream>>>(
                    a.qDevice, a.kDevice, a.vDevice, a.maskDevice, a.oDevice, a.softmaxLseDevice,
                    a.workspaceDevice, a.tilingDevice);
        } else {
            FAInferSplitB<DType, true, false>
                <<<blockDim, nullptr, aclStream>>>(
                    a.qDevice, a.kDevice, a.vDevice, a.maskDevice, a.oDevice, a.softmaxLseDevice,
                    a.workspaceDevice, a.tilingDevice);
        }
    } else {
        if (has_softcap) {
            FAInferSplitB<DType, false, true>
                <<<blockDim, nullptr, aclStream>>>(
                    a.qDevice, a.kDevice, a.vDevice, a.maskDevice, a.oDevice, a.softmaxLseDevice,
                    a.workspaceDevice, a.tilingDevice);
        } else {
            FAInferSplitB<DType, false, false>
                <<<blockDim, nullptr, aclStream>>>(
                    a.qDevice, a.kDevice, a.vDevice, a.maskDevice, a.oDevice, a.softmaxLseDevice,
                    a.workspaceDevice, a.tilingDevice);
        }
    }
}

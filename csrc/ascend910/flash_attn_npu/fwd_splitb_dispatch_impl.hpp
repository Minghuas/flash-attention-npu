/**
 * Copyright (c) 2026, perf-shortSeqLargeBatch project.
 *
 * SplitB 前向 dispatch 的共享 impl 体（v3 catlass 路线）。由 autogen/
 * fwd_dispatch_{fp16,bf16}_splitb.cpp include（每个 TU 显式实例化一个
 * launch_fwd_splitb_impl<DType>），kernel 模板（SplitB::FAInferSplitB）落在独立 TU 并行编译。
 * 模板树轴：MaskType(NO/CAUSAL/SWA) × HAS_SOFTCAP，与 FAInfer launch 树风格一致。
 */

#pragma once

#include "fwd_splitb_dispatch.hpp"

// 标准头（kernel 文件假定可见，与 fwd_dispatch_impl.hpp 同款注释）
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <limits>

// SplitB::FAInferSplitB kernel 模板 + FaiKenel::MaskType + KernelCommon 常量
#include "mha_fwd_splitb.cpp"
#include <cstdio>

template <typename DType>
void launch_fwd_splitb_impl(const FwdLaunchArgs &a) {
    const uint32_t blockDim = a.blockDim;
    const aclrtStream aclStream = a.aclStream;
    const bool dbgEnv = getenv("FLASH_ATTN_SPLITB_DEBUG") != nullptr;  // #44.45 同款总开关
    if (dbgEnv) {
        printf("3001 [splitb-dispatch] impl entered, blockDim=%u\n", blockDim); fflush(stdout);
    }
    const bool has_softcap = a.has_softcap;
    // SplitB v1：无 paged / FD / TND 轴（触发条件天然排除；dropout 在 host 侧拦截）
    (void)a.paged_KV;
    (void)a.flashDecodeFlag;

    if (a.is_local) {
        if (has_softcap) {
            SplitB::FAInferSplitB<DType, FaiKenel::MaskType::MASK_SWA, true>
                <<<blockDim, nullptr, aclStream>>>(
                    a.fftsAddr, a.qDevice, a.kDevice, a.vDevice, a.maskDevice, a.oDevice,
                    a.softmaxLseDevice, a.workspaceDevice, a.tilingDevice);
        } else {
            SplitB::FAInferSplitB<DType, FaiKenel::MaskType::MASK_SWA, false>
                <<<blockDim, nullptr, aclStream>>>(
                    a.fftsAddr, a.qDevice, a.kDevice, a.vDevice, a.maskDevice, a.oDevice,
                    a.softmaxLseDevice, a.workspaceDevice, a.tilingDevice);
        }
    } else if (a.is_causal) {
        if (has_softcap) {
            SplitB::FAInferSplitB<DType, FaiKenel::MaskType::MASK_CAUSAL, true>
                <<<blockDim, nullptr, aclStream>>>(
                    a.fftsAddr, a.qDevice, a.kDevice, a.vDevice, a.maskDevice, a.oDevice,
                    a.softmaxLseDevice, a.workspaceDevice, a.tilingDevice);
        } else {
            SplitB::FAInferSplitB<DType, FaiKenel::MaskType::MASK_CAUSAL, false>
                <<<blockDim, nullptr, aclStream>>>(
                    a.fftsAddr, a.qDevice, a.kDevice, a.vDevice, a.maskDevice, a.oDevice,
                    a.softmaxLseDevice, a.workspaceDevice, a.tilingDevice);
        }
    } else {
        if (has_softcap) {
            SplitB::FAInferSplitB<DType, FaiKenel::MaskType::NO_MASK, true>
                <<<blockDim, nullptr, aclStream>>>(
                    a.fftsAddr, a.qDevice, a.kDevice, a.vDevice, a.maskDevice, a.oDevice,
                    a.softmaxLseDevice, a.workspaceDevice, a.tilingDevice);
        } else {
            if (dbgEnv) {
                printf("3002 [splitb-dispatch] branch NO_MASK, launching <<<%u>>>...\n", blockDim); fflush(stdout);
            }
            SplitB::FAInferSplitB<DType, FaiKenel::MaskType::NO_MASK, false>
                <<<blockDim, nullptr, aclStream>>>(
                    a.fftsAddr, a.qDevice, a.kDevice, a.vDevice, a.maskDevice, a.oDevice,
                    a.softmaxLseDevice, a.workspaceDevice, a.tilingDevice);
            if (dbgEnv) {
                printf("3003 [splitb-dispatch] launch returned\n"); fflush(stdout);
            }
        }
    }
}

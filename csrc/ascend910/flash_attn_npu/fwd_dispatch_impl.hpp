/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

//
// Shared implementation of the v2 forward FAInfer dispatch. Included by the
// generated autogen/fwd_dispatch_<dtype>_<layout>.cpp stubs, each of which
// explicitly instantiates one launch_fwd_impl<DType, IS_TND>, so the FAInfer
// template instantiations land in separate (parallel-compiled) object files.
//
// The launch tree below reproduces the exact dtype x paged x mask x layout
// combinations of the three original host functions (mha_fwd_kvcache: BSND
// with FD; mha_fwd: BSND non-paged; mha_varlen_fwd: TND). Flash-decode is NOT
// a template axis: it is a runtime tiling flag (FAInferTilingData
// .flashDecodeFlag) so the AICPU scheduler-metadata path can decide it on
// device; idle cores past needCoreNum are skipped inside the kernel.

#pragma once

#include "fwd_dispatch.hpp"

// Standard headers that the CATLASS/FAG headers (reached via mha_fwd_kvcache.cpp)
// assume to be already visible.
#include <algorithm>
#include <cstring>
#include <limits>

// mha_fwd_kvcache.cpp provides the SplitFuse::FAInfer kernel template, the
// FAInferKernel class, FAIKernelParams, and the FaiKenel enum namespace.
#include "mha_fwd_kvcache.cpp"

// 8-param FAInfer (no IS_FD template arg — flash-decode moved to tiling).
#define FWD_LAUNCH(DTYPE, PAGED, MASK, LAYOUT_ENUM, SOFTCAP)                       \
    SplitFuse::FAInfer<DTYPE, DTYPE, float, PAGED,                                 \
                       FaiKenel::MaskType::MASK, LAYOUT_ENUM,                      \
                       Catlass::Epilogue::LseModeT::OUT_ONLY, SOFTCAP>             \
        <<<blockDim, nullptr, aclStream>>>(                                        \
            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice,     \
            oDevice, softmaxLseDevice, qSeqDevice, kvSeqDevice,                    \
            workspaceDevice, tilingDevice)

template <typename DType, bool IS_TND>
void launch_fwd_impl(const FwdLaunchArgs &a) {
    constexpr auto LAYOUT = IS_TND ? FaiKenel::inputLayout::TND : FaiKenel::inputLayout::BSND;

    const uint32_t blockDim = a.blockDim;
    const aclrtStream aclStream = a.aclStream;
    const uint64_t fftsAddr = a.fftsAddr;
    const bool paged_KV = a.paged_KV;
    const bool is_causal = a.is_causal;
    const bool is_local = a.is_local;
    const bool flashDecodeFlag = a.flashDecodeFlag;
    const bool has_softcap = a.has_softcap;
    uint8_t *qDevice = a.qDevice;
    uint8_t *kDevice = a.kDevice;
    uint8_t *vDevice = a.vDevice;
    uint8_t *maskDevice = a.maskDevice;
    uint8_t *blockTableDevice = a.blockTableDevice;
    uint8_t *oDevice = a.oDevice;
    uint8_t *softmaxLseDevice = a.softmaxLseDevice;
    uint8_t *qSeqDevice = a.qSeqDevice;
    uint8_t *kvSeqDevice = a.kvSeqDevice;
    uint8_t *workspaceDevice = a.workspaceDevice;
    uint8_t *tilingDevice = a.tilingDevice;
    (void)flashDecodeFlag;

    if (paged_KV) {
        if (is_local) {
            if (has_softcap) {
                FWD_LAUNCH(DType, true, MASK_SWA, LAYOUT, true);
            } else {
                FWD_LAUNCH(DType, true, MASK_SWA, LAYOUT, false);
            }
        } else if (is_causal) {
            if (has_softcap) {
                FWD_LAUNCH(DType, true, MASK_CAUSAL, LAYOUT, true);
            } else {
                FWD_LAUNCH(DType, true, MASK_CAUSAL, LAYOUT, false);
            }
        } else {
            if (has_softcap) {
                FWD_LAUNCH(DType, true, NO_MASK, LAYOUT, true);
            } else {
                FWD_LAUNCH(DType, true, NO_MASK, LAYOUT, false);
            }
        }
    } else {
        if (is_local) {
            if (has_softcap) {
                FWD_LAUNCH(DType, false, MASK_SWA, LAYOUT, true);
            } else {
                FWD_LAUNCH(DType, false, MASK_SWA, LAYOUT, false);
            }
        } else if (is_causal) {
            if (has_softcap) {
                FWD_LAUNCH(DType, false, MASK_CAUSAL, LAYOUT, true);
            } else {
                FWD_LAUNCH(DType, false, MASK_CAUSAL, LAYOUT, false);
            }
        } else {
            if (has_softcap) {
                FWD_LAUNCH(DType, false, NO_MASK, LAYOUT, true);
            } else {
                FWD_LAUNCH(DType, false, NO_MASK, LAYOUT, false);
            }
        }
    }
}

#undef FWD_LAUNCH

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
// v3 (950) forward FAInfer dispatch, fp16 x TND variant.
// One explicit instantiation per translation unit so the FAInfer / FAInferDn
// kernel templates compile in parallel across cores; head_dim is a runtime
// tiling axis (not a template parameter), so it is not a generation axis.

#include "../fai_host_api_impl.hpp"

template void launch_fai_dispatch<half, true>(
    const FwdLaunchArgs &a);

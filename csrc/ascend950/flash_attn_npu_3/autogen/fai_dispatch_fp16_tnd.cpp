/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
#include "../fai_host_api_impl.hpp"

namespace fai_host {
template aclError launch_fai_dispatch<half, true>(
    uint32_t kernelKey, bool enableDN, bool lseMode,
    uint32_t blockDim, aclrtStream stream,
    uint8_t *qDevice, uint8_t *kDevice, uint8_t *vDevice,
    uint8_t *maskDevice, uint8_t *blockTableDevice,
    uint8_t *oDevice, uint8_t *lseDevice,
    uint8_t *qSeqDevice, uint8_t *kvSeqDevice,
    uint8_t *workspaceDevice, uint8_t *tilingDevice);
}  // namespace fai_host

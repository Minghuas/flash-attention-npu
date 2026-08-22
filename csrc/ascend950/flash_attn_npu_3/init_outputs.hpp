#ifndef FAI950_INIT_OUTPUTS_HPP
#define FAI950_INIT_OUTPUTS_HPP

#include <cstdint>
#include <limits>

#include "catlass/arch/resource.hpp"
#include "kernel_operator.h"
#include "kernel_common.hpp"

namespace Catlass::Epilogue::Block {

template <class ArchTag_, class ElementO_>
class InitOutputs950 {
public:
    using ArchTag = ArchTag_;
    using ElementO = ElementO_;

    static constexpr uint32_t OUTPUT_UB_OFFSET = 6U * 32768U;
    static constexpr uint32_t LSE_UB_OFFSET = 7U * 32768U + 2048U;
    static constexpr uint32_t LSE_ELEMS_PER_ROW = 8U;

    __aicore__ inline
    explicit InitOutputs950(Arch::Resource<ArchTag> &resource)
    {
        outputUbTensor = resource.ubBuf.template GetBufferByByte<ElementO>(OUTPUT_UB_OFFSET);
        lseUbTensor = resource.ubBuf.template GetBufferByByte<float>(LSE_UB_OFFSET);
    }

    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementO> gOutput,
                    AscendC::GlobalTensor<float> gLse,
                    uint32_t qSBlockSize,
                    uint32_t qNBlockSize,
                    uint32_t qHeads,
                    uint32_t embedV,
                    uint32_t outputStride)
    {
        auto partition = GetFAIGroupedRowPartition(qSBlockSize, qNBlockSize, 8U);
        uint32_t rowCount = partition.validRows;
        uint32_t logicalRowStart = partition.logicalRowStart;
        if (rowCount == 0U) {
            return;
        }
        uint32_t embedRound = (embedV + 15U) / 16U * 16U;
        uint32_t outputElems = rowCount * embedRound;
        uint32_t lseElems = rowCount * LSE_ELEMS_PER_ROW;

        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        AscendC::Duplicate(outputUbTensor, static_cast<ElementO>(0), outputElems);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
        if (qNBlockSize == 1U) {
            AscendC::DataCopyPad(
                gOutput[logicalRowStart * outputStride], outputUbTensor,
                AscendC::DataCopyExtParams(rowCount, embedV * sizeof(ElementO), 0,
                    (outputStride - embedV) * sizeof(ElementO), 0));
        } else {
            uint32_t firstHead = logicalRowStart / qSBlockSize;
            uint32_t headCount = rowCount / qSBlockSize;
            for (uint32_t headLocal = 0; headLocal < headCount; ++headLocal) {
                AscendC::DataCopyPad(
                    gOutput[(firstHead + headLocal) * embedV], outputUbTensor[headLocal * qSBlockSize * embedRound],
                    AscendC::DataCopyExtParams(qSBlockSize, embedV * sizeof(ElementO), 0,
                        (outputStride - embedV) * sizeof(ElementO), 0));
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
        AscendC::Duplicate(
            lseUbTensor,
            std::numeric_limits<float>::infinity(),
            lseElems);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
        if (qNBlockSize == 1U) {
            AscendC::DataCopyPad(
                gLse[logicalRowStart * qHeads], lseUbTensor,
                AscendC::DataCopyExtParams(rowCount, sizeof(float), 0,
                    (qHeads - 1U) * sizeof(float), 0));
        } else {
            uint32_t firstHead = logicalRowStart / qSBlockSize;
            uint32_t headCount = rowCount / qSBlockSize;
            for (uint32_t headLocal = 0; headLocal < headCount; ++headLocal) {
                AscendC::DataCopyPad(
                    gLse[firstHead + headLocal], lseUbTensor[headLocal * qSBlockSize],
                    AscendC::DataCopyExtParams(qSBlockSize, sizeof(float), 0,
                        (qHeads - 1U) * sizeof(float), 0));
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    AscendC::LocalTensor<ElementO> outputUbTensor;
    AscendC::LocalTensor<float> lseUbTensor;
};

}

#endif

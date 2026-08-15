#ifndef FAI950_INIT_OUTPUTS_HPP
#define FAI950_INIT_OUTPUTS_HPP

#include <cstdint>
#include <limits>
#include "catlass/arch/resource.hpp"
#include "kernel_operator.h"

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
                    uint32_t rowCount,
                    uint32_t qHeads,
                    uint32_t embedV,
                    uint32_t outputStride)
    {
        uint32_t embedRound = (embedV + 15U) / 16U * 16U;
        uint32_t outputElems = rowCount * embedRound;
        uint32_t lseElems = rowCount * LSE_ELEMS_PER_ROW;

        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        AscendC::Duplicate(outputUbTensor, static_cast<ElementO>(0), outputElems);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
        AscendC::DataCopyPad(
            gOutput,
            outputUbTensor,
            AscendC::DataCopyExtParams(
                rowCount,
                embedV * sizeof(ElementO),
                0,
                (outputStride - embedV) * sizeof(ElementO),
                0));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
        AscendC::Duplicate(
            lseUbTensor,
            std::numeric_limits<float>::infinity(),
            lseElems);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
        AscendC::DataCopyPad(
            gLse,
            lseUbTensor,
            AscendC::DataCopyExtParams(
                rowCount,
                sizeof(float),
                0,
                (qHeads - 1U) * sizeof(float),
                0));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    AscendC::LocalTensor<ElementO> outputUbTensor;
    AscendC::LocalTensor<float> lseUbTensor;
};

}
#endif

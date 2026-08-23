// debug/test_alibi.cpp
// Minimal NPU test of csrc/arch22/flash_attn_npu_v2/alibi.hpp's ApplyAlibiRows, modeled on
// debug/playground.cpp. One AIV kernel fills a zero-initialized score tile with the ALiBi
// bias (slopes from GM), copies it out; host compares against a plain-C++ reference.
//
// Build (like playground.cpp):
//   bisheng -O2 -x asc --npu-arch=dav-2201 debug/test_alibi.cpp -o debug/test_alibi
// Run:  ./debug/test_alibi

#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "kernel_operator.h"

using namespace AscendC;

#include "../csrc/arch22/flash_attn_npu_v2/alibi.hpp"

const bool print_result = true;
// const bool print_result = false;

// ---- fixed test config (edit + recompile to try other cases) ----
constexpr int H = 2;            // num heads
constexpr int QS_BLOCK = 8;     // rows per head (qSBlockSize)
constexpr int COLS = 8;        // tile width (columnNumRound)
constexpr int ROW_NUM = H * QS_BLOCK;
constexpr int SCORE_N = ROW_NUM * COLS;
constexpr int ABS_ROW_START = 0;
constexpr int Q_POS_BASE = 30;   // i_q = Q_POS_BASE + token
constexpr int KV_S_START = 10;   // j  = KV_S_START + col

// constexpr int UB_MAX_ELEMENT_NUM = 8192;  // 避免UB溢出！
// constexpr int H = 2;            // num heads
// // constexpr int QS_BLOCK = 128;     // rows per head (qSBlockSize)
// constexpr int QS_BLOCK = 8;     // rows per head (qSBlockSize)
// constexpr int COLS = 512;        // tile width (columnNumRound)
// // constexpr int COLS = 256;        // tile width (columnNumRound)  256是边界
// constexpr int ROW_NUM = H * QS_BLOCK;
// constexpr int SCORE_N = ROW_NUM * COLS;
// constexpr int ABS_ROW_START = 0;
// constexpr int Q_POS_BASE = 1300;   // i_q = Q_POS_BASE + token
// constexpr int KV_S_START = 1024;   // j  = KV_S_START + col

// TODO: 研究 Q_POS_BASE 和 KV_S_START的三种情况

template <Alibi::AlibiMaskType MODE>
class KernelAlibi {
public:
    __aicore__ inline KernelAlibi() = default;

    __aicore__ inline void Init(GM_ADDR out, GM_ADDR slopes) {
        gSlopes.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(slopes), H);
        gOut.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out), SCORE_N);
        pipe.InitBuffer(scoreQ, 1, SCORE_N * sizeof(float));
        pipe.InitBuffer(workBuf, COLS * sizeof(float));
    }

    __aicore__ inline void Main() {
        LocalTensor<float> score = scoreQ.AllocTensor<float>();
        LocalTensor<float> work = workBuf.Get<float>();
        Duplicate<float>(score, 0.0f, SCORE_N);          // init 0 -> after ApplyAlibiRows, score == bias
        PipeBarrier<PIPE_V>();
        Alibi::ApplyAlibiRows<MODE>(score, /*scoreOffset=*/0, /*rowStride=*/COLS, COLS,
                                    ABS_ROW_START, ROW_NUM, QS_BLOCK, Q_POS_BASE,
                                    gSlopes, /*slopesGmOffset=*/0, work, KV_S_START);
        scoreQ.EnQue(score);
        LocalTensor<float> out = scoreQ.DeQue<float>();
        DataCopy(gOut, out, SCORE_N);
        scoreQ.FreeTensor(out);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECOUT, 1> scoreQ;
    TBuf<TPosition::VECIN> workBuf;
    GlobalTensor<float> gOut;
    GlobalTensor<float> gSlopes;
};

__global__ __aicore__ void kernel_nomask(GM_ADDR out, GM_ADDR slopes) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelAlibi<Alibi::AlibiMaskType::NO_MASK> op;
    op.Init(out, slopes);
    op.Main();
}

__global__ __aicore__ void kernel_causal(GM_ADDR out, GM_ADDR slopes) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelAlibi<Alibi::AlibiMaskType::MASK_CAUSAL> op;
    op.Init(out, slopes);
    op.Main();
}

static std::vector<float> launch(const std::vector<float>& slopes, bool causal) {
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    float* hOut = nullptr;
    aclrtMallocHost(reinterpret_cast<void**>(&hOut), SCORE_N * sizeof(float));
    uint8_t* dOut = nullptr;
    uint8_t* dSlopes = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&dOut), SCORE_N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&dSlopes), H * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dSlopes, H * sizeof(float), slopes.data(), H * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    if (causal) {
        kernel_causal<<<1, nullptr, stream>>>(dOut, dSlopes);
    } else {
        kernel_nomask<<<1, nullptr, stream>>>(dOut, dSlopes);
    }
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(hOut, SCORE_N * sizeof(float), dOut, SCORE_N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<float> out(hOut, hOut + SCORE_N);
    aclrtFreeHost(hOut);
    aclrtFree(dOut);
    aclrtFree(dSlopes);
    aclrtDestroyStream(stream);
    return out;
}

static float compare(const std::vector<float>& out, const std::vector<float>& slopes, bool causal) {
    float maxdiff = 0.0f;
    for (int ri = 0; ri < ROW_NUM; ++ri) {
        int head = (ABS_ROW_START + ri) / QS_BLOCK;
        int token = (ABS_ROW_START + ri) % QS_BLOCK;
        float slope = slopes[head];
        int i_q = Q_POS_BASE + token;
        for (int c = 0; c < COLS; ++c) {
            if(print_result) std::cout << out[ri * COLS + c] << " ";
            int j = KV_S_START + c;
            float expected = causal ? slope * static_cast<float>(j)
                                    : -slope * std::fabs(static_cast<float>(i_q - j));
            float d = std::fabs(out[ri * COLS + c] - expected);
            if (d > maxdiff) maxdiff = d;
        }
        if(print_result) std::cout << std::endl;
    }
    if(print_result) std::cout << std::endl;
    return maxdiff;
}

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    // 打印超参数
    std::cout << "H=" << H << "\n QS_BLOCK=" << QS_BLOCK << "\n COLS=" << COLS << "\n ROW_NUM=" << ROW_NUM << "\n SCORE_N=" << SCORE_N << "\n ABS_ROW_START=" << ABS_ROW_START << "\n Q_POS_BASE=" << Q_POS_BASE << "\n KV_S_START=" << KV_S_START << std::endl;

    // std::vector<float> slopes = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> slopes = {1.0f, 2.0f};

    std::vector<float> out = launch(slopes, /*causal=*/false);
    float md = compare(out, slopes, false);
    std::cout << "[NO_MASK ] maxdiff=" << md << "  " << (md < 5e-3f ? "PASS" : "FAIL") << std::endl;
    out = launch(slopes, /*causal=*/true);
    md = compare(out, slopes, true);
    std::cout << "[CAUSAL  ] maxdiff=" << md << "  " << (md < 5e-3f ? "PASS" : "FAIL") << std::endl;

    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}

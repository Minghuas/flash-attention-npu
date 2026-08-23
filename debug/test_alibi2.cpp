// debug/test_alibi.cpp
// Standalone CPU test of csrc/arch22/flash_attn_npu_v2/alibi.hpp's ApplyAlibiRows.
//
// Goal (divide & conquer): confirm alibi.hpp's ALiBi bias MATH is correct IN ISOLATION,
// independent of the FlashAttention kernel's parameter threading. We mock the AscendC
// vector intrinsics (LocalTensor / GlobalTensor / CreateVecIndex / Abs / Muls / Adds /
// Add / PipeBarrier) with plain CPU loops so the REAL alibi.hpp compiles and runs on host,
// then compare its output against a plain-C++ reference for NO_MASK / MASK_CAUSAL /
// MASK_SWA across many parameter combos (single/multi head, row/col offsets, head-
// boundary tiles, negative baseColIdx).
//
// Build:  g++ -std=c++17 -O2 debug/test_alibi.cpp -o debug/test_alibi && ./debug/test_alibi

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "../csrc/arch22/flash_attn_npu_v2/alibi.hpp"

// ===================== Reference bias (plain C++) =====================
// Authoritative spec mirrored here:
//   NO_MASK:      bias[ri][c] = -slope[head] * |i_q - j_abs|
//   MASK_CAUSAL:  bias[ri][c] = +slope[head] * j_abs        (drops row-constant -slope*i)
//   MASK_SWA:     alibi.hpp currently delegates to NO_MASK.
// head = (absRowStart+ri)/qSBlockSize, token = (absRowStart+ri)%qSBlockSize,
// i_q   = qPosBase + token,  j_abs = kvSStartIdx + c.
struct TestParams {
    uint32_t H;             // num heads (== slopes.size())
    uint32_t qSBlockSize;   // rows per head in the S-block
    uint32_t cols;          // columnNumRound (tile width)
    uint32_t absRowStart;   // logical start row of this tile (drives head/token)
    uint32_t rowNumCurLoop; // rows in this tile
    int64_t  qPosBase;      // query position base (i_q = qPosBase + token)
    int64_t  kvSStartIdx;   // KV tile start (j_abs = kvSStartIdx + c)
    std::vector<float> slopes;
};

template <Alibi::AlibiMaskType MODE>
bool run_test(const std::string& name, const TestParams& p, bool verbose) {
    std::vector<float> score(p.rowNumCurLoop * p.cols, 0.0f);  // init 0 -> after ApplyAlibi, == bias
    std::vector<float> work(p.cols, 0.0f);
    AscendC::LocalTensor<float>  scoreUb(score.data());
    AscendC::LocalTensor<float>  workUb(work.data());
    AscendC::GlobalTensor<float> slopesGm(p.slopes.data());

    Alibi::ApplyAlibiRows<MODE>(scoreUb, /*scoreOffset=*/0, /*rowStride=*/p.cols, p.cols,
                                p.absRowStart, p.rowNumCurLoop, p.qSBlockSize, p.qPosBase,
                                slopesGm, /*slopesGmOffset=*/0, workUb, p.kvSStartIdx);

    float maxdiff = 0.0f;
    int mismatches = 0;
    for (uint32_t ri = 0; ri < p.rowNumCurLoop; ++ri) {
        uint32_t absRow = p.absRowStart + ri;
        uint32_t head   = absRow / p.qSBlockSize;
        uint32_t token  = absRow % p.qSBlockSize;
        float    slope  = p.slopes[head];
        for (uint32_t c = 0; c < p.cols; ++c) {
            int64_t j_abs = p.kvSStartIdx + c;
            float expected;
            if (MODE == Alibi::AlibiMaskType::MASK_CAUSAL) {
                expected = slope * static_cast<float>(j_abs);                       // +slope*j
            } else { // NO_MASK and MASK_SWA (SWA delegates to NO_MASK in alibi.hpp)
                int64_t i_q = p.qPosBase + token;
                expected = -slope * std::fabs(static_cast<float>(i_q - j_abs));     // -slope*|i_q-j|
            }
            float actual = score[ri * p.cols + c];
            float d = std::fabs(actual - expected);
            if (d > maxdiff) maxdiff = d;
            if (d > 1e-4f && ++mismatches <= 8 && verbose)
                printf("    MISMATCH ri=%u(head=%u,token=%u) c=%u j=%ld: actual=%.4f expected=%.4f\n",
                       ri, head, token, c, (long)j_abs, actual, expected);
        }
    }
    bool ok = maxdiff < 1e-3f;
    printf("[%s] %-16s maxdiff=%.3g  %s\n", ok ? "PASS" : "FAIL", name.c_str(), maxdiff,
           ok ? "" : (verbose ? "" : "(run verbose for details)"));
    return ok;
}

int main() {
    printf("=== alibi.hpp ApplyAlibiRows isolation test (CPU, mocked AscendC) ===\n\n");
    bool all = true;
    std::vector<float> s4 = {0.5f, 0.2f, 0.1f, 0.05f};

    // ---- NO_MASK ----
    all &= run_test<Alibi::AlibiMaskType::NO_MASK>("NO_MASK H=1",      {1, 8, 16, 0, 8,  0, 0,  {0.5f}}, true);
    all &= run_test<Alibi::AlibiMaskType::NO_MASK>("NO_MASK H=4",      {4, 8, 16, 0, 32, 0, 0,  s4}, true);  // cross-head slope
    all &= run_test<Alibi::AlibiMaskType::NO_MASK>("NO_MASK kvOff",    {4, 8, 16, 0, 32, 0, 4,  s4}, true);  // j shifted
    all &= run_test<Alibi::AlibiMaskType::NO_MASK>("NO_MASK qPos",     {4, 8, 16, 0, 32, 3, 0,  s4}, true);  // i_q shifted
    all &= run_test<Alibi::AlibiMaskType::NO_MASK>("NO_MASK midBlock", {4, 8, 16, 6, 12, 0, 0,  s4}, true);  // head boundary at 8/16/24
    all &= run_test<Alibi::AlibiMaskType::NO_MASK>("NO_MASK negBase",  {2, 8, 16, 0, 16, 0, 20, {0.5f, 0.2f}}, true); // i_q<j (baseColIdx<0)
    all &= run_test<Alibi::AlibiMaskType::NO_MASK>("NO_MASK qPos!=kv", {3, 8, 16, 0, 24, 5, 2,  {0.5f, 0.2f, 0.1f}}, true);

    // ---- MASK_CAUSAL ----
    all &= run_test<Alibi::AlibiMaskType::MASK_CAUSAL>("CAUSAL H=1",   {1, 8, 16, 0, 8,  0, 0,  {0.5f}}, true);
    all &= run_test<Alibi::AlibiMaskType::MASK_CAUSAL>("CAUSAL H=4",   {4, 8, 16, 0, 32, 0, 0,  s4}, true);  // per-head rescale
    all &= run_test<Alibi::AlibiMaskType::MASK_CAUSAL>("CAUSAL kvOff", {4, 8, 16, 0, 32, 0, 5,  s4}, true);  // ramp starts at kvSStartIdx
    all &= run_test<Alibi::AlibiMaskType::MASK_CAUSAL>("CAUSAL mid",   {4, 8, 16, 6, 12, 0, 0,  s4}, true);

    // ---- MASK_SWA (delegates to NO_MASK in alibi.hpp) ----
    all &= run_test<Alibi::AlibiMaskType::MASK_SWA>("SWA H=4",         {4, 8, 16, 0, 32, 0, 0,  s4}, true);

    printf("\n=== %s ===\n", all ? "ALL PASS (alibi.hpp logic correct)" : "SOME FAILED (alibi.hpp has a logic bug)");
    return all ? 0 : 1;
}

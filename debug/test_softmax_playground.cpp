/**
 * SplitB Softmax 独立验证 playground（devlog #44.20，用户主导剥离方案）。
 *
 * 目的：完全脱离 SplitB 四段 kernel，只验证 splitb_softmax.hpp 的计算正确性。
 *   - host 构造 S（模拟 QK 产物，可辨识数值 (b+1)(h+1)(s+1)(j+1)/den）写入 GM
 *   - 设备侧（AIV-only kernel）复刻 mha_fwd_splitb 段2 的调用形态：
 *     同款事件预置 + smEpilogue.init + tile 循环（同款 Layout/GemmCalc/gP/gS/gStats
 *     指针与 workspace 布局），并对同一批 tile 重复 ROUNDS 轮调用
 *     （复现"第 3 次 smEpilogue 调用出错"的模式——全流程 kernel 中 b1 tile0 即第 3 次）
 *   - 每轮调用后 DumpTensor 导出 P/stats（desc 见下），host 读回后与 CPU 参考
 *     （fp32 exp + fp16 量化语义）逐元素比对，打印 PASS/FAIL 与首批错位
 *
 * 事件/dump 说明：
 *   - SplitBSoftmax 内部使用 MTE3_V/V_MTE3/V_MTE2/MTE2_V 的 EVENT_ID0-1（ping-pong），
 *     kernel 入口预置 MTE3_V(0,1,2,4) + V_MTE2(0,1,2,3)（与 mha_fwd_splitb VEC 段同款）
 *   - desc 编码（每轮 R、batch b、tile t）：
 *       100 + R*10 + b*2 + t : 调用前的 S 区（float，AIV0 行）——验证输入未被破坏
 *       200 + R*10 + b*2 + t : 调用后的 P（fp16，AIV0 行）
 *       300 + R*10 + b*2 + t : 调用后的 stats（max[0..128)+sum[128..256)）
 *
 * 编译（在仓库根目录；与 setup.py 同款 flag，AIV-only 单源）：
 *   bisheng -x asc --npu-arch=dav-2201 --cce-auto-infer-kernel-type=false \
 *     -O2 -std=c++17 -fPIC \
 *     -I/usr/local/Ascend/ascend-toolkit/9.0.0/aarch64-linux/tikcpp/include \
 *     -I/usr/local/Ascend/cann-9.0.0/aarch64-linux/tikcpp/include \
 *     -Icsrc/catlass/include -Icsrc/ascend910/flash_attn_npu \
 *     -I/usr/local/Ascend/cann-9.0.0/aarch64-linux/ascendc/include \
 *     -o debug/test_softmax_playground debug/test_softmax_playground.cpp \
 *     -L/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64 -lascendcl -ltiling_api -lplatform
 *   （ ascend_toolkit/latest 若存在可用其路径替换版本号；运行：debug/test_softmax_playground ）
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "kernel_operator.h"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "splitb_softmax.hpp"

using namespace AscendC;

// ---------------- 测例常量（与 test_splitb_stage_full.py 的 B2/S32/Sk32/H2 同款） ----------------
constexpr int B = 2;      // batch
constexpr int H = 2;      // q 头数（MHA：Hkv=H）
constexpr int Sq = 32;    // q 序列长
constexpr int Sk = 32;    // kv 序列长
constexpr int ROUNDS = 2; // 重复调用轮数（dump 预算）

constexpr int ROW_NUM_MAX = 128;                    // stats 行距（= Q_TILE_CEIL）
constexpr int COLS_PAD = ((Sk + 15) / 16) * 16;     // = Sk（32 已 16 对齐）
constexpr int S1_AREA = ROW_NUM_MAX * COLS_PAD;     // S/P 区（float 计）
constexpr int STATS = 2 * ROW_NUM_MAX;              // stats 区
// 剥离布局省略 OTmp 区（softmax 不触碰）；gStats 偏移 = tile*(S1_AREA+STATS)+S1_AREA
constexpr int P_AREA = S1_AREA / 2;   // P 槽大小：128×colsPad halfs = S 区一半（#44.23 链式方案）
constexpr int PER_TILE = S1_AREA + STATS;
constexpr int N_TILE = H;                           // MHA G=1：每头 1 tile
constexpr int PER_BATCH = N_TILE * PER_TILE;
// 每轮独立 GM 区（round r 用 [r*B, r*B+B) 段）：S 恒为 host 原始值——避免
// 原地覆写使第 2+ 轮输入变成"上轮 P"（首版 harness 的 host 侧 bug，devlog #44.20）
constexpr int P_SCRATCH = P_AREA;  // t=0 的 P 落此处；t>=1 的 P[t] 链式写入 S[t-1] 死区
constexpr int WS_FLOATS = ROUNDS * B * PER_BATCH + ROUNDS * B * P_SCRATCH;
constexpr int SDEN = B * H * Sq;                    // S 归一化分母：S_raw ≤ Sk

// ---------------- 设备侧：AIV-only softmax kernel ----------------
class SoftmaxPlayground {
public:
    __aicore__ inline SoftmaxPlayground(GM_ADDR ws)
    {
        gWsF.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws), WS_FLOATS);
        gP.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(ws));
    }

    __aicore__ inline void Main()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        AscendC::printf("[SM-PG] blk=%u subIdx=%u subNum=%u\n",
                        blockIdx, AscendC::GetSubBlockIdx(), AscendC::GetSubBlockNum());
        // 事件预置（mha_fwd_splitb VEC 段同款：MTE3_V 0,1,2,4 + V_MTE2 0-3）
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);

        // DBG 串行化（devlog #44.20）：单 block 顺序处理全部 batch——分离跨 block 并发因素
        const uint32_t bCount = 2;   // DBG 恢复双批处理（复刻全流程 b1 指纹）
        for (uint32_t bo = 0; bo < bCount; ++bo)
        if (blockIdx == 0 || bo == 0) {
        const uint32_t batchIdx = bo;

        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> resource;
        SplitB::SplitBSoftmax<half, false> sm;   // NO_SOFTCAP（ScaleS 当前被注释=不乘，见类内说明）
        sm.init(resource, 1.0f, 0.0f);

        for (uint32_t r = 0; r < ROUNDS; ++r) {
            for (uint32_t t = 0; t < N_TILE; ++t) {
                const uint64_t tileBase =
                    (static_cast<uint64_t>(r) * B + batchIdx) * PER_BATCH + t * PER_TILE;
                const uint32_t rowNum = Sq;             // qNBlockSize=1：rowNum=qSBlockSize=Sq

                // 调用前 dump S 区（前 32 行完整；验证输入未被破坏）
                if (r <= 1) {
                    const uint8_t dimD = 2;
                    uint32_t shapeD[2] = {rowNum, COLS_PAD};
                    AscendC::ShapeInfo infoD(dimD, shapeD);
                    const uint32_t desc = 100 + r * 10 + batchIdx * 2 + t;
                    AscendC::printf(
                        "[SM-PG] pre  S r=%u b=%u tile=%u head=%u rows=%u cols=%u desc=%u\n",
                        r, batchIdx, t, t, rowNum, COLS_PAD, desc);
                    AscendC::DumpTensor(gWsF[tileBase], desc, rowNum * COLS_PAD, infoD);
                }

                // 零并发完全串行（devlog #44.20 实验2）：__vector__ 单子核 + half 循环
                //（每调用算 16 行：subNum=1 下 subIdx=0 → rowSplit=16、base 平移覆盖全 32 行）。
                // 判别：若此形态确定性全对 → bug 在双子核并发交互；仍错 → bug 在单调用链内。
                // 类驱动忠实调用（确定性复现形态，devlog #44.20）：__vector__ 单子核
                // + half 循环（每调用 16 行、base 平移）——零并发、逐位确定性。
                // 指纹：调用 1 全对；调用 2 起 P 拷贝丢失（stats 正确）。
                {
                    // 双 AIV + 链式 P（devlog #44.25：全流程修复后仍 b1t0 s8-15 坏，
                    // 唯一未验证组合 = 双 AIV 并发 + 链式。单调用/tile，行分摊在类内）
                    const uint64_t pSlot = (t == 0)
                        ? (static_cast<uint64_t>(r) * B + batchIdx + 1) * PER_BATCH
                        : (tileBase - static_cast<uint64_t>(PER_TILE));
                    const uint64_t pHalf = pSlot * 2;
                    SplitB::layout::RowMajor layOutP(Sq, Sk, COLS_PAD);
                    SplitB::layout::RowMajor layOutS(Sq, Sk, COLS_PAD);
                    Catlass::GemmCoord shape{Sq, Sk, 0};
                    sm(gP[pHalf], gWsF[tileBase], gWsF[tileBase + S1_AREA],
                       layOutP, layOutS, shape, Sq, 1);
                }

                // 调用后 dump P + stats（dump 预算控制：仅前两轮）
                if (r <= 1) {
                    const uint8_t dimD = 2;
                    uint32_t shapeD[2] = {rowNum, COLS_PAD};
                    AscendC::ShapeInfo infoD(dimD, shapeD);
                    const uint32_t desc = 200 + r * 10 + batchIdx * 2 + t;
                    AscendC::printf(
                        "[SM-PG] post P r=%u b=%u tile=%u head=%u rows=%u cols=%u desc=%u\n",
                        r, batchIdx, t, t, rowNum, COLS_PAD, desc);
                    AscendC::DumpTensor(gP[tileBase * 2], desc, rowNum * COLS_PAD, infoD);
                    const uint8_t dimS = 2;
                    uint32_t shapeS[2] = {STATS, 1};
                    AscendC::ShapeInfo infoS(dimS, shapeS);
                    const uint32_t descS = 300 + r * 10 + batchIdx * 2 + t;
                    AscendC::printf(
                        "[SM-PG] post stats r=%u b=%u tile=%u head=%u layout=max[0..128)+sum[128..256) desc=%u\n",
                        r, blockIdx, t, t, descS);
                    AscendC::DumpTensor(gWsF[tileBase + S1_AREA], descS, STATS, infoS);
                }
            }
        }

        }
        // 收尾 drain（softmax 用到的 event 全量排空）
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    AscendC::GlobalTensor<float> gWsF;
    AscendC::GlobalTensor<half> gP;
};

// KERNEL_TYPE_MIX_AIV_1_0（枚举值 2，编译器头 KernelMetaType）：每 block = 1 AIC +
// 2 AIV 子核——SplitBSoftmax 行分摊公式按 GetSubBlockNum()==2 设计（qSBlockSize/2
// 硬编码）。__vector__ 单子核型下 AIV1 半区无人计算（devlog #44.20 实证：s16-31 全
// 垃圾）。本剥离测试不依赖 auto-infer=false（那是扩展构建的范式冲突需求），故用
// KERNEL_TASK_TYPE_DEFAULT 显式 MIX。
__global__ __aicore__ void softmax_pg_kernel(GM_ADDR ws)
{
#ifdef __DAV_C220_CUBE__
    (void)ws;
    return;   // 本剥离测试无 CUBE 段
#else
    SoftmaxPlayground op(ws);
    op.Main();
#endif
}

// ---------------- host 侧：构造 S、launch、读回、参考比对 ----------------
static float sRaw(int b, int h, int s, int j)
{
    return static_cast<float>((b + 1) * (h + 1) * (s + 1) * (j + 1)) / static_cast<float>(SDEN);
}

static float toFp16(float v)
{
    // host 侧 fp16 量化模拟（round-to-nearest-even，与设备 CAST_NONE/fp16 一致近似）
    __fp16 h = static_cast<__fp16>(v);
    return static_cast<float>(h);
}

int main()
{
    const int wsBytes = WS_FLOATS * sizeof(float);
    std::vector<float> hS(WS_FLOATS, 0.0f);
    // S 写入每 tile 的 S 区（行距 COLS_PAD，前 Sq 行有效）
    for (int r = 0; r < ROUNDS; ++r) {
      for (int b = 0; b < B; ++b) {
        for (int t = 0; t < N_TILE; ++t) {
            const int64_t base = (static_cast<int64_t>(r) * B + b) * PER_BATCH + t * PER_TILE;
            for (int s = 0; s < Sq; ++s) {
                for (int j = 0; j < Sk; ++j) {
                    hS[base + s * COLS_PAD + j] = sRaw(b, t, s, j);
                }
            }
        }
      }
    }

    aclInit(nullptr);
    // 卡号可配置（默认 1，与仓库测试脚本的空闲卡惯例一致）：SM_PG_DEVICE=0/1
    const int devId = getenv("SM_PG_DEVICE") ? atoi(getenv("SM_PG_DEVICE")) : 1;
    aclrtSetDevice(devId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    void* dWs = nullptr;
    aclrtMalloc(&dWs, wsBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dWs, wsBytes, hS.data(), wsBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    softmax_pg_kernel<<<1, nullptr, stream>>>(static_cast<uint8_t*>(dWs));  // DBG 串行
    aclrtSynchronizeStream(stream);

    std::vector<float> hOut(WS_FLOATS, 0.0f);
    aclrtMemcpy(hOut.data(), wsBytes, dWs, wsBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    // ---------------- 参考与逐轮比对 ----------------
    // 注意：P 是 fp16 原地覆写（读回按 uint16 解释）；stats 为 float
    const auto* p16 = reinterpret_cast<const __fp16*>(hOut.data());
    int totalBad = 0;
    for (uint32_t r = 0; r < ROUNDS; ++r) {
        int roundBad = 0;
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < N_TILE; ++t) {
                // 参考计算（与 kernel 语义一致：S 不乘 scale（ScaleS 注释中）、
                // P=exp(S-max) fp16 量化、sum 为 fp32 exp 之和）
                float mx[Sq], sm[Sq];
                for (int s = 0; s < Sq; ++s) {
                    mx[s] = -1e30f;
                    for (int j = 0; j < Sk; ++j) {
                        mx[s] = fmaxf(mx[s], sRaw(b, t, s, j));
                    }
                    sm[s] = 0.0f;
                    for (int j = 0; j < Sk; ++j) {
                        sm[s] += expf(sRaw(b, t, s, j) - mx[s]);
                    }
                }
                const int64_t base = (static_cast<int64_t>(r) * B + b) * PER_BATCH + t * PER_TILE;
                // 逐行段统计（h0=行0-15、h1=行16-31，与 half 调用一一对应）
                int badP[2] = {0, 0}, badMx[2] = {0, 0}, badSm[2] = {0, 0};
                for (int s = 0; s < Sq; ++s) {
                    const int h = s / (Sq / 2);
                    for (int j = 0; j < Sk; ++j) {
                        const float ref = toFp16(expf(sRaw(b, t, s, j) - mx[s]));
                        // P 半元素布局：P(s,j) 在 half 索引 = base*2 + s*COLS_PAD + j
                        const float got = static_cast<float>(p16[(t == 0 ? (static_cast<int64_t>(r)*B + b + 1) * PER_BATCH : base - PER_TILE) * 2 + s * COLS_PAD + j]);
                        if (fabsf(got - ref) > 1e-3 + 1e-3 * fabsf(ref)) {
                            ++badP[h];
                            ++roundBad;
                        }
                    }
                }
                for (int s = 0; s < Sq; ++s) {
                    const int h = s / (Sq / 2);
                    const float gotMx = hOut[base + S1_AREA + s];
                    const float gotSm = hOut[base + S1_AREA + 128 + s];
                    if (fabsf(gotMx - mx[s]) > 1e-4) { ++badMx[h]; ++roundBad; }
                    if (fabsf(gotSm - sm[s]) > 2e-3) { ++badSm[h]; ++roundBad; }
                }
                printf("[TILE] r=%u b=%d t=%d | P: h0=%d h1=%d | max: h0=%d h1=%d | "
                       "sum: h0=%d h1=%d\n",
                       r, b, t, badP[0], badP[1], badMx[0], badMx[1], badSm[0], badSm[1]);
            }
        }
        printf("[ROUND %u] %s（错 %d 处）\n", r, roundBad == 0 ? "PASS" : "FAIL", roundBad);
        totalBad += roundBad;
    }
    printf("==== 总结：%s（共 %d 处不符；第 3 次 smEpilogue 调用 = r1 b1 tile0）====\n",
           totalBad == 0 ? "全 PASS" : "存在错误", totalBad);

    aclrtFree(dWs);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return totalBad == 0 ? 0 : 1;
}

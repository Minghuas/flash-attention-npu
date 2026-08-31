/**
 * Copyright (c) 2026, perf-shortSeqLargeBatch project.
 *
 * SplitB 前向 kernel（大 Batch 小 SeqLen 场景）——v3：批 matmul 引擎 + 批间错位流水。
 *
 * 【架构 v3（devlog #46，用户拍板路线）】QK/PV 弃用 tile 级 FAI 引擎（qk/pv_matmul.hpp），
 * 改用 catlass 通用 BatchedMatmul 引擎（BlockMmad<MmadAtlasA2Pingpong<true>>，example 01
 * 已验证组装）：batch 维 = dim1 = qHeads（参考实现 tensorABatchSize 同语义），每头一次
 * [Sq,D]×[D,Sk]→S / [Sq,Sk]×[Sk,D]→OTmp 的引擎调用。引擎 A/B 面 L1 双缓冲 + 跨调用
 * 槽位轮转，t19 定罪的 l1A 竞态类（devlog #45.3）结构性关闭。设计全文：
 * perf/design/splitb_batchmatmul.md。
 *
 * 【架构 v2（devlog #45，保留）】批间错位流水，CUBE/VEC 双程序 + 4 个 mode2 flag：
 *
 *   CUBE 迭代 t：QK(bo_t) → wait softmaxReady[bo_{t-1}] → wait doReady[bo_{t-3}]
 *                → PV(bo_{t-1}) → set pvReady
 *   VEC  迭代 t：wait qkReady[bo_{t-1}] → softmax(bo_{t-1}) → set softmaxReady
 *                → wait pvReady[bo_{t-2}] → divout(bo_{t-2}) → set doReady
 *
 *   稳态重叠：QK(bo_t) ∥ softmax(bo_{t-1})；PV(bo_{t-1}) ∥ divout(bo_{t-2})。
 *   所有跨核等待只指向对方核的上一个迭代 ⇒ 依赖严格递减，无环（死锁自由）。
 *   每批每 flag 收支 1:1；哨兵迭代（CUBE +3 / VEC +2）排空在飞批并补齐收支。
 *   槽位保护（2 槽 ping-pong，boIdx%2）：S 槽由 CUBE 迭代序（sm 消费）保护、
 *   P 槽由 AIV 程序序（pv 消费）保护、OTmp 槽由 doReady 保护。
 *
 * 算法层（devlog #34，承袭）：核间只切 B 轴（aic 基数）；S2 不切分（单遍 softmax、
 *   无 rescale 状态机）；workspace ping/pong 按 boIdx 奇偶，布局见 operator() 内注。
 * softmax/divout 用封装 epilogue（splitb_softmax.hpp / splitb_divout.hpp）：operator()
 * 内部按行分摊双 AIV，行 max/sum 走 GM stats。
 *
 * S3 范围：NO_MASK / fp16 / bf16 / D≤128 / MHA+GQA 数据通路（causal/SWA 为 S4；
 *   softcap 随 HAS_SOFTCAP 模板穿透 softmax epilogue）。
 */

#ifndef MHA_FWD_SPLITB_CPP_
#define MHA_FWD_SPLITB_CPP_

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "fa_block.h"
#include "kernel_common.hpp"
#include "kernel_operator.h"
#include "splitb_bm_pingpong.hpp"
#include "splitb_tilingdata.h"
#include "splitb_softmax.hpp"
#include "splitb_divout.hpp"

using namespace Catlass;
using namespace KernelCommon;

namespace SplitB {

    template <
        class BlockMmadQK,
        class BlockMmadPV,
        class EpilogueSplitBSoftmax,
        class EpilogueSplitBDivOut,
        FaiKenel::MaskType MASK_TYPE = FaiKenel::MaskType::NO_MASK>
    class SplitBKernel {
    public:
        using ArchTag = typename BlockMmadQK::ArchTag;
        using ElementQ = typename BlockMmadQK::ElementA;
        using LayoutQ = typename BlockMmadQK::LayoutA;
        using ElementK = typename BlockMmadQK::ElementB;
        using LayoutK = typename BlockMmadQK::LayoutB;
        using ElementS = typename BlockMmadQK::ElementC;
        using LayoutS = typename BlockMmadQK::LayoutC;

        using ElementP = typename BlockMmadPV::ElementA;
        using LayoutP = typename BlockMmadPV::LayoutA;
        using ElementV = typename BlockMmadPV::ElementB;
        using LayoutV = typename BlockMmadPV::LayoutB;
        using ElementOTmp = typename BlockMmadPV::ElementC;
        using LayoutOTmp = typename BlockMmadPV::LayoutC;

        using ElementO = typename EpilogueSplitBDivOut::ElementO;
        using LayoutO = typename EpilogueSplitBDivOut::LayoutO;
        using ElementLse = typename EpilogueSplitBDivOut::ElementLse;
        using LayoutLse = typename EpilogueSplitBDivOut::LayoutLse;

        // tile 几何（v3.3：qN 打包——MHA 每头一 tile，GQA 共 kv 头的头打包）
        struct TileGeom {
            uint32_t qSBlockSize;   // 本 tile 每头行数（≤128，末块取尾）
            uint32_t qNBlockSize;   // 本 tile 打包头数（MHA 恒 1；GQA ≤ qNBlockTile）
            uint32_t rowNum;        // = qSBlockSize × qNBlockSize（引擎调用的 M 维）
            uint32_t kvNIdx;        // GQA：kv 头索引
            uint32_t qNStartIdx;    // 首 q 头索引（打包组在 Q GM 上行连续）
            uint64_t tileIdx;       // 批内扁平 tile 序号（workspace tile 块索引）
        };

        struct GlobalTensorBundle {
            AscendC::GlobalTensor<ElementQ>& gQ;
            AscendC::GlobalTensor<ElementK>& gK;
            AscendC::GlobalTensor<ElementV>& gV;
            AscendC::GlobalTensor<ElementS>& gS;
            AscendC::GlobalTensor<ElementP>& gP;
            AscendC::GlobalTensor<ElementOTmp>& gOTmp;
            AscendC::GlobalTensor<float>& gStats;
            AscendC::GlobalTensor<ElementO>& gO;
            AscendC::GlobalTensor<float>& gLse;
        };

        __aicore__ inline
        SplitBKernel() {}

        __aicore__ inline
        void operator()(FAIKernelParams const &params)
        {
            // ---- tiling 读取（GM→栈；设备侧直访公有字段） ----
            SplitBTilingData tilingLocal;
            const __gm__ uint8_t *src = reinterpret_cast<const __gm__ uint8_t *>(params.tiling);
            uint8_t *dst = reinterpret_cast<uint8_t *>(&tilingLocal);
            for (uint32_t i = 0; i < sizeof(SplitBTilingData); ++i) {
                dst[i] = src[i];
            }

            const auto &in = tilingLocal.inputParams;
            batchSize = in.bSize;
            kvHeads = in.n2Size;
            groupSize = in.gSize;
            qHeads = kvHeads * groupSize;
            qSeqlen = in.s1Size;
            kvSeqlen = in.s2Size;
            embed = in.dSize;
            scaleValue = in.scaleValue;
            softcapValue = in.softcapValue;
            windowSizeLeft = in.windowSizeLeft;   // S4：SWA 窗口（-1 = 无界）
            windowSizeRight = in.windowSizeRight;
            debugFlag = (in.debugFlag != 0);   // 设备 printf 探针开关（env FLASH_ATTN_SPLITB_DEBUG）
            // dumpFlag = (in.dumpFlag != 0);     // 设备 Dump Tensor 探针开关（env FLASH_ATTN_SPLITB_DUMP，devlog #44）
            splitFactorSize = tilingLocal.multiCoreParams.splitFactorSize;

            AscendC::GlobalTensor<ElementQ> gQ;
            gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
            AscendC::GlobalTensor<ElementK> gK;
            gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
            AscendC::GlobalTensor<ElementV> gV;
            gV.SetGlobalBuffer((__gm__ ElementV *)params.v);
            AscendC::GlobalTensor<ElementO> gO;
            gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
            AscendC::GlobalTensor<float> gLse;
            gLse.SetGlobalBuffer((__gm__ float *)params.lse);

            uint32_t coreIdx = AscendC::GetBlockIdx();
#ifdef __DAV_C220_CUBE__
            // ① 硬件事件预置：引擎首次 Wait 依赖"已释放"初态（S3 实测教训）。
            //    [#47 清理收敛] CUBE 侧唯一事件消费者 = Pingpong 引擎（presetEvents=false，
            //    宿主接管），在用 ID：MTE1_MTE2{0..3}（l1A/l1B 槽）+ M_MTE1{0..3}（l0A/l0B 槽）；
            //    其 MTE2_MTE1/MTE1_M 为调用内自配对（免预置）；FIX_M 在 unit-flag 模式不使用。
            //    预置与末尾 drain 严格镜像（事件收支 launch 内闭合，#44.53g）。
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            // ② [#46 v3] 通用 Pingpong 引擎（example 01 同款组装）：A/B 面 L1 双缓冲
            //    （l1A/BTensorList[STAGES=2]）+ 两向事件全保护 + 跨调用槽位轮转——
            //    t19 定罪的 l1A 竞态类结构性关闭。局部构造（引擎无默认构造器）；
            //    QK 占 L1 前 [l1A×2 | l1B×2]，PV 顺延（L1 静态分区，共 256KB ≤ 512KB）。
            //    presetEvents=false：双实例 ctor 预置会叠加成 Set-on-set（#44.53g），
            //    沿用上方 ① 的 kernel 级预置——事件会计与 FAI 引擎时代同构。
            constexpr uint32_t l1QkFootprint =
                BlockMmadQK::STAGES * (BlockMmadQK::L1A_SIZE + BlockMmadQK::L1B_SIZE);
            BlockMmadQK blockMmadQKEngine(resource, 0, false);
            BlockMmadPV blockMmadPvEngine(resource, l1QkFootprint, false);
#endif
#ifdef __DAV_C220_VEC__
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);   // 子核1 的 softmax 链（#44.24）
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            smEpilogue.init(resource, scaleValue, softcapValue);
            divoutEpilogue.init(resource);
            coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif

            // ---- 步长（BSND，照抄 FAInfer :237-240） ----
            strideQ = static_cast<uint64_t>(qHeads * embed);
            strideO = strideQ;
            strideK = static_cast<uint64_t>(kvHeads * embed);
            strideV = strideK;
            colsPad = RoundUp(static_cast<uint32_t>(kvSeqlen), FaiKenel::BLOCK_SIZE);   // = alignedS2
            dPad = RoundUp(static_cast<uint32_t>(embed), FaiKenel::BLOCK_SIZE);

            // ---- workspace 布局（单位 = float 元素个数；与 splitb_host.cpp 公式严格一致）----
            // 每核两段连续：coreWsOffset → [tile 区: 2 批槽 × T tile 块] + [P 区: 2 批槽 × T P 槽]
            // 每 tile 块 = [S 区(128×colsPad) | OTmp 区(128×dPad) | stats 区(256)]；
            // P 槽 = 128×colsPad half。S/P/OTmp 各自独立（#44.44），批流水下无 in-place。
            sTileElems = static_cast<uint64_t>(Q_TILE_CEIL) * colsPad;
            pSlotElems = sTileElems / 2;                                  // half 计（fp32 元素折半）
            oTmpTileElems = static_cast<uint64_t>(Q_TILE_CEIL) * dPad;
            statsPerTask = 2 * static_cast<uint64_t>(Q_TILE_CEIL);        // max 128 + sum 128
            perTileElems = sTileElems + oTmpTileElems + statsPerTask;

            // ---- tile 几何（v3.3：恢复 GQA qN 打包） ----
            // 打包条件（host 侧同款守卫，须一致）：①qS 单块（Sq ≤ qSBlockTile，闸门下
            // 恒真）；②Sq 16 对齐（引擎 BSHD A 面装载的行区 fractal 对齐要求）。
            // MHA（G=1）恒 1（单 GEMM 单 B，数学 forced）。
            curQSBlockTile = GetQSBlockTile(static_cast<uint32_t>(kvSeqlen));
            curQSBlockNum = CeilDiv(static_cast<uint32_t>(qSeqlen), curQSBlockTile);
            curQNBlockTile = (curQSBlockNum == 1U && (qSeqlen % 16 == 0)) ?
                GetQNBlockTile(static_cast<uint32_t>(qSeqlen), static_cast<uint32_t>(groupSize)) : 1U;
            qNBlockNumPerGroup = CeilDiv(static_cast<uint32_t>(groupSize), curQNBlockTile);
            curQNBlockNum = qNBlockNumPerGroup * static_cast<uint32_t>(kvHeads);
            tileNumPerBatch = static_cast<uint64_t>(curQNBlockNum) *
                static_cast<uint64_t>(curQSBlockNum);                     // 批内 tile 数 T
            perBatchTileElems = tileNumPerBatch * perTileElems;
            tileAreaElems = 2 * perBatchTileElems;                        // 本核 tile 区（两批槽）
            perCoreElems = tileAreaElems + 2 * tileNumPerBatch * pSlotElems;

            // ---- 核间 B 切分（aic 基数；CUBE 用原始 blockIdx、VEC 除以子核数） ----
            const uint64_t coreWsOffset = static_cast<uint64_t>(coreIdx) * perCoreElems;

            // ---- ws 四视图（基址 = 本核对应区首，寻址不掺 coreWsOffset，#44.44） ----
            __gm__ uint8_t *ws = reinterpret_cast<__gm__ uint8_t *>(params.workSpace);
            AscendC::GlobalTensor<ElementS> gS;
            AscendC::GlobalTensor<ElementP> gP;
            AscendC::GlobalTensor<ElementOTmp> gOTmp;
            AscendC::GlobalTensor<float> gStats;
            gS.SetGlobalBuffer(reinterpret_cast<__gm__ ElementS *>(ws + coreWsOffset * sizeof(float)));
            gOTmp.SetGlobalBuffer(reinterpret_cast<__gm__ ElementOTmp *>(ws + coreWsOffset * sizeof(float)));
            gStats.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + coreWsOffset * sizeof(float)));
            gP.SetGlobalBuffer(reinterpret_cast<__gm__ ElementP *>(
                ws + (coreWsOffset + tileAreaElems) * sizeof(float)));

            GlobalTensorBundle globalTensors{
                gQ, gK, gV, gS, gP, gOTmp, gStats, gO, gLse
            };
            int64_t batchStart = static_cast<int64_t>(coreIdx) * splitFactorSize;
            int64_t batchEnd = batchStart + splitFactorSize;
            if (batchSize < batchEnd) {
                batchEnd = batchSize;                                 // 末核尾裁剪
            }
            const int64_t nBatches = (batchEnd > batchStart) ? (batchEnd - batchStart) : 0;

            if (debugFlag) {
                AscendC::printf("[SB] c%u v%u enter: Batch range:(%u, %u) tile_nums=%u BBBBBBBBBBBBBBBBBBBBB\n",
                                coreIdx, AscendC::GetSubBlockIdx(), (uint32_t)batchStart, (uint32_t)batchEnd,
                                curQNBlockNum * curQSBlockNum);
            }

            // ==================== 批间错位流水（v2，devlog #45） ====================
            // 同步拓扑（mode 2 双 AIV 计数器，每批每 flag 收支 1:1）：
            //   qkReady      CUBE set(PIPE_FIX) → 双 AIV wait
            //   softmaxReady 双 AIV set(PIPE_MTE3) → CUBE wait
            //   pvReady      CUBE set(PIPE_FIX) → 双 AIV wait
            //   doReady      双 AIV set(PIPE_MTE3) → CUBE wait（OTmp 槽回收，#45 新增）
#ifdef __DAV_C220_CUBE__
            for (int64_t t = 0; t < nBatches + 3; ++t) {
                const int64_t boQK = batchStart + t;        // 本迭代 QK 目标批
                const int64_t boPV = batchStart + t - 1;    // 本迭代 PV 目标批
                // ① 发射新 QK（先发射后等待：QK(bo_t) ∥ softmax(bo_{t-1}) 是核心重叠；
                //    S 槽(t%2) 覆写安全 = CUBE 迭代序已消费 softmaxReady(bo_{t-2})）
                if (t < nBatches) {
                    if (debugFlag) {
                        AscendC::printf("[SB] c%u | pipe t=%u S1-QK 1111111111 bo=%u\n",
                                        coreIdx, (uint32_t)t, (uint32_t)boQK);
                    }
                    StageQK(coreIdx, boQK, globalTensors, blockMmadQKEngine);
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);   // QK 全 tile 落 GM 后
                    if (debugFlag) {
                        AscendC::printf("[SB] c%u | pipe t=%u S1-QK 11111EEEEEEE bo=%u\n",
                                        coreIdx, (uint32_t)t, (uint32_t)boQK);
                    }
                }
                // ② 消费 softmax(bo_{t-1})（PV 数据依赖；t∈[1,n] 共 n 次）
                if (t >= 1 && t <= nBatches) {
                    if (debugFlag) {
                        AscendC::printf("[SB] c%u | pipe t=%u S3-PV 333331111 before wait softmaxReady (sm of bo=%u)\n",
                                        coreIdx, (uint32_t)t, (uint32_t)boPV);
                    }
                    Arch::CrossCoreWaitFlag(softmaxReady);
                    if (debugFlag) {
                        AscendC::printf("[SB] c%u | pipe t=%u S3-PV 333331111 after wait softmaxReady\n",
                                        coreIdx, (uint32_t)t);
                    }
                }
                // ③ 消费 divout(bo_{t-3})（OTmp 槽覆写保护；t∈[3,n+2] 共 n 次，
                //    末 2 个哨兵迭代补齐尾部 doReady 收支，防跨 launch 计数泄漏）
                if (t >= 3) {
                    if (debugFlag) {
                        AscendC::printf("[SB] c%u | pipe t=%u S3-PV 333332222 before wait doReady (do of bo=%u)\n",
                                        coreIdx, (uint32_t)t, (uint32_t)(boPV - 2));
                    }
                    Arch::CrossCoreWaitFlag(doReady);
                    if (debugFlag) {
                        AscendC::printf("[SB] c%u | pipe t=%u S3-PV 333332222 after wait doReady\n",
                                        coreIdx, (uint32_t)t);
                    }
                }
                // ④ 发射 PV(bo_{t-1})（t∈[1,n]）
                if (t >= 1 && t <= nBatches) {
                    if (debugFlag) {
                        AscendC::printf("[SB] c%u | pipe t=%u S3-PV 3333333333 bo=%u\n",
                                        coreIdx, (uint32_t)t, (uint32_t)boPV);
                    }
                    StagePV(coreIdx, boPV, globalTensors, blockMmadPvEngine);
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);   // PV 全 tile 落 GM 后
                    if (debugFlag) {
                        AscendC::printf("[SB] c%u | pipe t=%u S3-PV 33333EEEEEEEEEEEE bo=%u\n",
                                        coreIdx, (uint32_t)t, (uint32_t)boPV);
                    }
                }
            }
#endif
#ifdef __DAV_C220_VEC__
            for (int64_t t = 0; t < nBatches + 2; ++t) {
                const int64_t boSM = batchStart + t - 1;    // 本迭代 softmax 目标批
                const int64_t boDO = batchStart + t - 2;    // 本迭代 divout 目标批
                // ① softmax(bo_{t-1})（t∈[1,n]）
                if (t >= 1 && t <= nBatches) {
                    if (debugFlag) {
                        AscendC::printf("[SB] c%u v%u | pipe t=%u S2-SM 222221111 before wait qkReady (qk of bo=%u)\n",
                                        coreIdx, AscendC::GetSubBlockIdx(), (uint32_t)t, (uint32_t)boSM);
                    }
                    Arch::CrossCoreWaitFlag(qkReady);
                    if (debugFlag) {
                        AscendC::printf("[SB] c%u v%u | pipe t=%u S2-SM 222221111 after wait qkReady\n",
                                        coreIdx, AscendC::GetSubBlockIdx(), (uint32_t)t);
                    }
                    StageSoftmax(coreIdx, boSM, globalTensors);
                    // PIPE_MTE3：全部 P/stats 拷贝落 GM 后才置位
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
                    if (debugFlag) {
                        AscendC::printf("[SB] c%u v%u | pipe t=%u S2-SM END 22222EEEEEEE bo=%u\n",
                                        coreIdx, AscendC::GetSubBlockIdx(), (uint32_t)t, (uint32_t)boSM);
                    }
                }
                // ② divout(bo_{t-2})（t∈[2,n+1]；doReady 由 CUBE 哨兵迭代消费）
                if (t >= 2) {
                    if (debugFlag) {
                        AscendC::printf("[SB] c%u v%u | pipe t=%u S4-DO 444441111 before wait pvReady (pv of bo=%u)\n",
                                        coreIdx, AscendC::GetSubBlockIdx(), (uint32_t)t, (uint32_t)boDO);
                    }
                    Arch::CrossCoreWaitFlag(pvReady);
                    if (debugFlag) {
                        AscendC::printf("[SB] c%u v%u | pipe t=%u S4-DO 444441111 after wait pvReady\n",
                                        coreIdx, AscendC::GetSubBlockIdx(), (uint32_t)t);
                    }
                    StageDivOut(coreIdx, boDO, globalTensors);
                    // OTmp 槽回收信号：O/LSE 落 GM 后才置位
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(doReady);
                    if (debugFlag) {
                        AscendC::printf("[SB] c%u v%u | pipe t=%u S4-DO 44444EEEEEEEEEEE bo=%u\n",
                                        coreIdx, AscendC::GetSubBlockIdx(), (uint32_t)t, (uint32_t)boDO);
                    }
                }
            }
#endif

            if (debugFlag) {
                AscendC::printf("[SB] c%u v%u exit: Batch range:(%u, %u) tile_nums=%u EEEEEEEEEEEEEEEEEEEEEE\n",
                                coreIdx, AscendC::GetSubBlockIdx(), (uint32_t)batchStart, (uint32_t)batchEnd,
                                curQNBlockNum * curQSBlockNum);
            }

            // ---- 收尾：事件全量 drain（保证异步拷贝全部落盘；跨核 flag 已每批闭合） ----
#ifdef __DAV_C220_CUBE__
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
#endif
#ifdef __DAV_C220_VEC__
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);   // softmax AIV1 链（#44.24/#44.45）
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
#endif
            AscendC::PipeBarrier<PIPE_ALL>();

        }

        // ==================== 段1：QK 批 matmul（CUBE） ====================
        // [#46 v3] 每 tile 一次引擎调用 [rowNum,D]×[D,Sk]→S[rowNum,Sk]（MHA 每头
        // 一 tile；GQA 打包共享 kv 头的头）。纯计算段：跨核 flag
        //（qkReady set）在流水驱动循环内。
        __aicore__ inline void StageQK(
            uint32_t coreIdx,
            int64_t boIdx,
            GlobalTensorBundle& globalTensors,
            BlockMmadQK& blockMmadQKEngine
        ) {
#ifdef __DAV_C220_CUBE__
            (void)coreIdx;
            auto& gQ = globalTensors.gQ;
            auto& gK = globalTensors.gK;
            auto& gS = globalTensors.gS;

            const uint64_t batchBuf = static_cast<uint64_t>(boIdx) % 2;   // ping/pong 槽
            const uint64_t batchBase = batchBuf * perBatchTileElems;

            // 循环不变量外提：B 面布局（K^T 转置视图 ColumnMajor(strideK, Sk)，v2 同款
            // ——ldm=strideK=Hkv·D 即 BSHD [B,S,H,D] 输入的 n 行跨步，元素 (k,n)→k+n·Hkv·D）
            // 与 strideQ 的类型转换。A/C 面描述依赖尾块（qS 末块行数、qN 末组头数
            // 如 G=6/qN=4 → 4,2），非全循环常量，保留在循环内。
            LayoutK layoutB(static_cast<uint64_t>(strideK), static_cast<uint32_t>(kvSeqlen));
            for (uint32_t qSBlockIdx = 0; qSBlockIdx < curQSBlockNum; ++qSBlockIdx) {
                for (uint32_t qNBlockIdx = 0; qNBlockIdx < curQNBlockNum; ++qNBlockIdx) {
                    const TileGeom tg = GetTileGeom(qSBlockIdx, qNBlockIdx);
                    const uint64_t gmQ = static_cast<uint64_t>(boIdx) * qSeqlen * strideQ +
                        static_cast<uint64_t>(qSBlockIdx) * curQSBlockTile * strideQ +
                        static_cast<uint64_t>(tg.qNStartIdx) * embed;
                    const uint64_t gmK = static_cast<uint64_t>(boIdx) * kvSeqlen * strideK +
                        static_cast<uint64_t>(tg.kvNIdx) * embed;
                    const uint64_t sOff = batchBase + tg.tileIdx * perTileElems;

                    // A：BSHD 操作数描述（[B,S,H,D] 头内行距 = H·D；引擎内自装，
                    // GQA 打包/MHA 单头统一形态）；C：S 头块（行步长 colsPad）
                    typename BlockMmadQK::AOperandBSHD aOperand{
                        tg.qSBlockSize, tg.qNBlockSize, static_cast<int64_t>(strideQ)};
                    LayoutS layoutC(tg.rowNum, static_cast<uint32_t>(kvSeqlen), colsPad);
                    GemmCoord actualShapeQK{tg.rowNum, static_cast<uint32_t>(kvSeqlen),
                                            static_cast<uint32_t>(embed)};
                    blockMmadQKEngine(gQ[gmQ], aOperand, gK[gmK], layoutB, gS[sOff], layoutC,
                                      actualShapeQK);
                }
            }
#endif
        }

        // ==================== 段2：softmax 全部 tile（VEC；每 tile 双 AIV 拆行） ====================
        // 纯计算段：跨核 flag（qkReady wait / softmaxReady set）在流水驱动循环内。
        __aicore__ inline void StageSoftmax(
            uint32_t coreIdx,
            int64_t boIdx,
            GlobalTensorBundle& globalTensors
        ) {
#ifdef __DAV_C220_VEC__
            (void)coreIdx;
            auto& gS = globalTensors.gS;
            auto& gP = globalTensors.gP;
            auto& gStats = globalTensors.gStats;

            const uint64_t batchBuf = static_cast<uint64_t>(boIdx) % 2;
            const uint64_t batchBase = batchBuf * perBatchTileElems;

            if constexpr (MASK_TYPE == FaiKenel::MaskType::NO_MASK) {
                for (uint32_t qSBlockIdx = 0; qSBlockIdx < curQSBlockNum; ++qSBlockIdx) {
                    for (uint32_t qNBlockIdx = 0; qNBlockIdx < curQNBlockNum; ++qNBlockIdx) {
                        const TileGeom tg = GetTileGeom(qSBlockIdx, qNBlockIdx);
                        const uint64_t sOff = batchBase + tg.tileIdx * perTileElems;
                        const uint64_t statOff = sOff + sTileElems + oTmpTileElems;
                        // P 槽 half 偏移：槽号 = 批内槽基 + tileIdx（pSlotElems float 计 ×2）
                        const uint64_t pOff = (batchBuf * tileNumPerBatch + tg.tileIdx) * pSlotElems * 2;
                        LayoutP layOutP(tg.rowNum, static_cast<uint32_t>(kvSeqlen), colsPad);
                        LayoutS layOutS(tg.rowNum, static_cast<uint32_t>(kvSeqlen), colsPad);
                        GemmCoord actualBlockShapeQK{tg.rowNum, static_cast<uint32_t>(kvSeqlen),
                                                    static_cast<uint32_t>(embed)};
                        smEpilogue(gP[pOff], gS[sOff], gStats[statOff],
                                layOutP, layOutS, actualBlockShapeQK,
                                tg.qSBlockSize, tg.qNBlockSize);
                    }
                }
            } else {
                // TODO(S4)：causal / SWA mask（softmax mask 重载）——实现前 mask 型模板
                // 不可用（dispatch 不应路由到此）
            }
#endif
        }

        // ==================== 段3：PV 批 matmul（CUBE） ====================
        // [#46 v3] per-head：[Sq,Sk]×[Sk,D]→OTmp[Sq,D]。A = P 头槽（行步长 colsPad）、
        // B = V[b,kvN] 行主序、C = OTmp 头块（行步长 dPad）。
        __aicore__ inline void StagePV(
            uint32_t coreIdx,
            int64_t boIdx,
            GlobalTensorBundle& globalTensors,
            BlockMmadPV& blockMmadPvEngine
        ) {
#ifdef __DAV_C220_CUBE__
            (void)coreIdx;
            auto& gV = globalTensors.gV;
            auto& gP = globalTensors.gP;
            auto& gOTmp = globalTensors.gOTmp;

            const uint64_t batchBuf = static_cast<uint64_t>(boIdx) % 2;
            const uint64_t batchBase = batchBuf * perBatchTileElems;

            // 循环不变量外提：B 面布局（V [Sk,D] RowMajor(Sk, strideV)——ldm=Hkv·D
            // 即 BSHD 的 n 行跨步）。A/C 面描述依赖尾块（同段1注），保留在循环内。
            LayoutV layoutB(static_cast<uint32_t>(kvSeqlen), static_cast<uint64_t>(strideV));
            for (uint32_t qSBlockIdx = 0; qSBlockIdx < curQSBlockNum; ++qSBlockIdx) {
                for (uint32_t qNBlockIdx = 0; qNBlockIdx < curQNBlockNum; ++qNBlockIdx) {
                    const TileGeom tg = GetTileGeom(qSBlockIdx, qNBlockIdx);
                    const uint64_t sOff = batchBase + tg.tileIdx * perTileElems;
                    const uint64_t oOff = sOff + sTileElems;
                    const uint64_t pOff = (batchBuf * tileNumPerBatch + tg.tileIdx) * pSlotElems * 2;
                    const uint64_t gmV = static_cast<uint64_t>(boIdx) * kvSeqlen * strideV +
                        static_cast<uint64_t>(tg.kvNIdx) * embed;

                    // A = P 头槽 [rowNum,Sk]（workspace 布局，行步长 colsPad）；
                    // C = OTmp（行步长 dPad）
                    LayoutP layoutA(tg.rowNum, static_cast<uint32_t>(kvSeqlen), colsPad);
                    LayoutOTmp layoutC(tg.rowNum, static_cast<uint32_t>(embed), dPad);
                    GemmCoord actualShapePV{tg.rowNum, static_cast<uint32_t>(embed),
                                            static_cast<uint32_t>(kvSeqlen)};
                    blockMmadPvEngine(gP[pOff], layoutA, gV[gmV], layoutB, gOTmp[oOff], layoutC,
                                      actualShapePV);
                }
            }
#endif
        }

        // ==================== 段4：divout 全部 tile（VEC；每 tile 双 AIV 拆行） ====================
        // 纯计算段：跨核 flag（pvReady wait / doReady set）在流水驱动循环内。
        __aicore__ inline void StageDivOut(
            uint32_t coreIdx,
            int64_t boIdx,
            GlobalTensorBundle& globalTensors
        ) {
#ifdef __DAV_C220_VEC__
            (void)coreIdx;
            auto& gS = globalTensors.gS;
            auto& gOTmp = globalTensors.gOTmp;
            auto& gStats = globalTensors.gStats;
            auto& gO = globalTensors.gO;
            auto& gLse = globalTensors.gLse;

            const uint64_t batchBuf = static_cast<uint64_t>(boIdx) % 2;
            const uint64_t batchBase = batchBuf * perBatchTileElems;

            if constexpr (MASK_TYPE == FaiKenel::MaskType::NO_MASK) {
                for (uint32_t qSBlockIdx = 0; qSBlockIdx < curQSBlockNum; ++qSBlockIdx) {
                    for (uint32_t qNBlockIdx = 0; qNBlockIdx < curQNBlockNum; ++qNBlockIdx) {
                        const TileGeom tg = GetTileGeom(qSBlockIdx, qNBlockIdx);
                        const uint64_t sOff = batchBase + tg.tileIdx * perTileElems;
                        const uint64_t oOff = sOff + sTileElems;
                        const uint64_t statOff = sOff + sTileElems + oTmpTileElems;
                        const uint64_t gmO = static_cast<uint64_t>(boIdx) * qSeqlen * strideO +
                            static_cast<uint64_t>(qSBlockIdx) * curQSBlockTile * strideO +
                            static_cast<uint64_t>(tg.qNStartIdx) * embed;
                        const uint64_t gmLse = static_cast<uint64_t>(boIdx) * qHeads * qSeqlen +
                            static_cast<uint64_t>(tg.qNStartIdx) * qSeqlen +
                            static_cast<uint64_t>(qSBlockIdx) * curQSBlockTile;
                        LayoutO layoutO(static_cast<uint32_t>(qSeqlen),
                                        static_cast<uint32_t>(embed * qHeads));
                        LayoutOTmp layoutOTmpT(tg.rowNum, static_cast<uint32_t>(embed), dPad);
                        LayoutLse layoutLse(static_cast<uint32_t>(qHeads),
                                            static_cast<uint32_t>(qSeqlen));
                        GemmCoord actualBlockShapePV{tg.rowNum, static_cast<uint32_t>(embed),
                                                    static_cast<uint32_t>(kvSeqlen)};
                        divoutEpilogue(
                            gO[gmO], gOTmp[oOff], gLse[gmLse], gStats[statOff],
                            layoutO, layoutOTmpT, layoutLse, actualBlockShapePV,
                            tg.qSBlockSize, tg.qNBlockSize);
                    }
                }
            }
#endif
        }

    private:
        // tile 几何（v3.3：qN 打包；MHA 退化为此前的每头一 tile）
        __aicore__ inline
        TileGeom GetTileGeom(uint32_t qSBlockIdx, uint32_t qNBlockIdx) const
        {
            TileGeom g;
            const uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;
            g.kvNIdx = qNBlockIdx / qNBlockNumPerGroup;
            g.qNStartIdx = g.kvNIdx * static_cast<uint32_t>(groupSize) +
                qNBlockIdxCurGroup * curQNBlockTile;
            g.qSBlockSize = (qSBlockIdx == (curQSBlockNum - 1U)) ?
                (static_cast<uint32_t>(qSeqlen) - qSBlockIdx * curQSBlockTile) : curQSBlockTile;
            g.qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1U)) ?
                (static_cast<uint32_t>(groupSize) - qNBlockIdxCurGroup * curQNBlockTile) : curQNBlockTile;
            g.rowNum = g.qSBlockSize * g.qNBlockSize;
            g.tileIdx = static_cast<uint64_t>(qSBlockIdx) * curQNBlockNum + qNBlockIdx;
            return g;
        }

        // tiling 派生参数
        int64_t batchSize;
        int64_t kvHeads;
        int64_t groupSize;
        int64_t qHeads;
        int64_t qSeqlen;
        int64_t kvSeqlen;
        int64_t embed;
        float scaleValue;
        float softcapValue;
        int64_t windowSizeLeft;   // S4：SWA 窗口（-1 = 无界）
        int64_t windowSizeRight;
        bool debugFlag;
        int64_t splitFactorSize;

        uint64_t strideQ;
        uint64_t strideO;
        uint64_t strideK;
        uint64_t strideV;
        uint32_t colsPad;
        uint32_t dPad;

        // workspace 布局（与 splitb_host.cpp 公式严格一致，改动必须同步）
        uint64_t sTileElems;
        uint64_t pSlotElems;
        uint64_t oTmpTileElems;
        uint64_t statsPerTask;
        uint64_t perTileElems;
        uint64_t tileNumPerBatch;          // 批内 tile 数 T
        uint64_t perBatchTileElems;        // 单批 tile 区 = T × perTileElems
        uint64_t tileAreaElems;            // 本核 tile 区 = 2 × perBatchTileElems
        uint64_t perCoreElems;

        // tile 几何
        uint32_t curQNBlockTile;           // v3 恒 1（保留字段对齐 host 打印）
        uint32_t qNBlockNumPerGroup;       // = groupSize
        uint32_t curQNBlockNum;            // = qHeads
        uint32_t curQSBlockTile;
        uint32_t curQSBlockNum;

        Arch::Resource<ArchTag> resource;
        Arch::CrossCoreFlag qkReady{QK_READY_ID};
        Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
        Arch::CrossCoreFlag pvReady{PV_READY_ID};
        Arch::CrossCoreFlag doReady{DO_READY_ID};   // divout 完成 → CUBE 的 OTmp 槽回收

        EpilogueSplitBSoftmax smEpilogue;
        EpilogueSplitBDivOut divoutEpilogue;
    };
}

namespace SplitB {
    template <
        typename InputDtypeQ = half,
        FaiKenel::MaskType maskCategory = FaiKenel::MaskType::NO_MASK,
        bool HAS_SOFTCAP = false>
    __global__ __aicore__ void FAInferSplitB(
        uint64_t fftsAddr,
        GM_ADDR q,
        GM_ADDR k,
        GM_ADDR v,
        GM_ADDR mask,
        GM_ADDR o,
        GM_ADDR lse,
        GM_ADDR workspace,
        GM_ADDR tiling
    )
    {
        AscendC::SetSyncBaseAddr(fftsAddr);
        (void)mask;   // S4：causal/SWA mask 经 softmax 的 mask 重载

        using ArchTag = Arch::AtlasA2;
        using ElementQ = InputDtypeQ;
        using LayoutQ = layout::RowMajor;
        using ElementK = InputDtypeQ;
        using LayoutK = layout::ColumnMajor;   // K^T 视图：B = [D,Sk] 列主序
        using ElementV = InputDtypeQ;
        using LayoutV = layout::RowMajor;
        using ElementS = float;
        using LayoutS = layout::RowMajor;
        using ElementP = InputDtypeQ;
        using LayoutP = layout::RowMajor;
        using ElementOTmp = float;
        using LayoutOTmp = layout::RowMajor;

        // [#46 v3/v3.2] Pingpong 引擎 = 我方 fork（splitb_bm_pingpong.hpp，自 catlass
        // submodule v1.6.1 逐行拷贝 + 生命周期定制；依赖库零改动）。A/B 面 L1 双缓冲，
        // ENABLE_UNIT_FLAG=true（unit flag copyout，example 01 实证组合）
        using DispatchPolicyBM = Gemm::MmadAtlasA2Pingpong<true>;
        using QType = Gemm::GemmType<ElementQ, LayoutQ>;
        using KType = Gemm::GemmType<ElementK, LayoutK>;
        using SType = Gemm::GemmType<ElementS, LayoutS>;
        using L1TileShapeQK = GemmShape<128, 128, 128>;   // M≥rowNum、N≥Sk、K≥D（闸门均 ≤128）
        using L0TileShapeQK = GemmShape<128, 128, 64>;
        using BlockMmadQK = SplitBBlockMmad<DispatchPolicyBM, L1TileShapeQK, L0TileShapeQK,
                                            QType, KType, SType>;

        using PType = Gemm::GemmType<ElementP, LayoutP>;
        using VType = Gemm::GemmType<ElementV, LayoutV>;
        using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
        using L1TileShapePV = GemmShape<128, 128, 128>;   // M≥rowNum、N≥D、K≥Sk
        using L0TileShapePV = GemmShape<128, 128, 64>;
        using BlockMmadPV = SplitBBlockMmad<DispatchPolicyBM, L1TileShapePV, L0TileShapePV,
                                            PType, VType, OTmpType>;

        using SplitBSoftmaxEpilogue = SplitBSoftmax<ElementQ, HAS_SOFTCAP>;
        using SplitBDivOutEpilogue = SplitBDivOut<ElementQ>;

        using SplitBKernelType = SplitBKernel<
            BlockMmadQK, BlockMmadPV, SplitBSoftmaxEpilogue, SplitBDivOutEpilogue, maskCategory>;

        FAIKernelParams params{q, k, v, mask, nullptr, nullptr, nullptr, o, lse, workspace, tiling};
        SplitBKernelType splitBKernel;
        splitBKernel(params);
    }
}

#endif // MHA_FWD_SPLITB_CPP_

/**
 * Copyright (c) 2026, perf-shortSeqLargeBatch project.
 *
 * SplitB 前向 kernel（大 Batch 小 SeqLen 场景）——方案 B 重构版（devlog #34）。
 *
 * 文件形态照抄 FAInfer mha_fwd_kvcache.cpp（用户要求，便于长期扩展）：
 *   namespace SplitB 内定义 SplitBKernel 模板类（空构造 + operator()(FAIKernelParams const&)
 *   + runMainLoop + 成员变量），文件尾部模板入口函数 FAInferSplitB 组装类型并调用。
 * 算法层照搬 ops-transformer flash_attention_score_bn2gs1s2_b.h（TilingB）：
 *   核间只切 B 轴；S2 不切分（单遍 softmax、无 rescale 状态机）；每 batch 四段流水：
 *   QK(整批) → softmax(整批) → PV(整批) → divout(整批)；workspace ping/pong 按 boIdx 奇偶。
 * 机制层用 FAInfer 已验证范式（D7）：catlass BlockMmadQK/PV + __DAV_C220_* 显式双段 +
 *   CrossCoreFlag×3 + fftsAddr；核间 aic 基数。
 *
 * tile 模型照抄 FAInfer runMainLoop（mha_fwd_kvcache.cpp:513-541，用户 FIXME #2/#4）：
 *   每 batch 的任务 = (qSBlockIdx × qNBlockIdx) tile；rowNum = qSBlockSize × qNBlockSize
 *   一次打包多头（如 S1=64 时 2 头×64 行 = 128 行满 L1Tile M 维），blockMmadQK 内部
 *   处理打包行——不再有逐头双层循环。
 * softmax/divout 用 FAInfer 式封装 epilogue（splitb_softmax.hpp / splitb_divout.hpp，
 *   FIXME #5/#6）：operator() 内部按 qNBlockSize/subBlockNum 把行分摊给两个 AIV，
 *   SubCoreCompute 逐行块计算；行 max/sum 走 GM stats（四段批所需，不同于 FAInfer 的 UB 驻留）。
 *
 * 与 FAInfer 主体的映射差异（仅一处结构性）：FAInfer 的 runMainLoop 为单 tile 粒度
 * （任务 = batch×qNBlock×qSBlock 跨核轮转），SplitB 为单 batch 粒度（核间 B 轴切分 +
 * 四段批，devlog #32/#34）；tile 几何因此提为私有成员 GetTileGeom（四段各需一次）。
 *
 * S3 范围：NO_MASK / fp16 / bf16 / D≤128 / MHA+GQA 数据通路（causal/SWA 为 S4；
 *   softcap 已随 HAS_SOFTCAP 模板穿透 softmax epilogue）。
 * 设计规范：perf/design/splitb_integration.md（v3 §3）；决策记录 perf/devlog.md #34。
 */

#ifndef MHA_FWD_SPLITB_CPP_
#define MHA_FWD_SPLITB_CPP_

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "pv_matmul.hpp"
#include "qk_matmul.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "fa_block.h"
#include "kernel_common.hpp"
#include "kernel_operator.h"
#include "splitb_tilingdata.h"
#include "splitb_softmax.hpp"
#include "splitb_divout.hpp"

using namespace Catlass;
using namespace KernelCommon;

namespace SplitB {

    // SplitB 的 KV 栈长：S2 不切分，单 KV 栈即整段 S2（触发闸门 alignedS2 ≤ 128）。
    // 区别于 FAInfer 的 MAX_KV_STACK_LEN=512（常规 kernel 按 512 行逐栈迭代 KV，devlog #36）。
    // 与 Q_TILE_CEIL（Q 张量分块上限）概念不同，只是数值恰好相等——不混用。
    // 消费点：L1 预算公式（给 V tile 预留 D×栈长×dtype）与 BlockMmad init 的 KVStackLen
    //（后者仅在 nIdx>0 时进入地址计算，我方恒 nIdx=0，传此值只为语义自洽）。
    constexpr uint32_t S2_STACK_LEN = 128;

    template <
        class BlockMmadQK,
        class BlockMmadPV,
        class EpilogueSplitBSoftmax,
        class EpilogueSplitBDivOut,
        FaiKenel::MaskType MASK_TYPE = FaiKenel::MaskType::NO_MASK>
    class SplitBKernel {
    public:
        using ArchTag = typename BlockMmadQK::ArchTag;
        using L1TileShape = typename BlockMmadQK::L1TileShape;
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

        // tile 几何（FAInfer :513-541 的 qNBlockTile/qSBlockTile 段；四段各需一次故提为嵌套类型）
        struct TileGeom {
            uint32_t qSBlockSize;   // 本 tile 每头行数（≤128，末块取尾）
            uint32_t qNBlockSize;   // 本 tile 头数（末组取尾；恒偶，双 AIV 对半）
            uint32_t rowNum;        // = qSBlockSize × qNBlockSize（打包行数）
            uint32_t kvNIdx;        // GQA：kv 头索引
            uint32_t qNStartIdx;    // 首 q 头索引
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
            AscendC::GlobalTensor<int32_t>& gBlockTable;
        };

        __aicore__ inline
        SplitBKernel() {}

        __aicore__ inline
        void operator()(FAIKernelParams const &params)
        {
            // ---- tiling 读取（GM→栈；getter 为 [host]，设备侧直访公有字段） ----
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
            pagedBlockSize = in.blockSize;
            scaleValue = in.scaleValue;
            softcapValue = in.softcapValue;
            windowSizeLeft = in.windowSizeLeft;   // S4：SWA 窗口（FAInfer 语义，-1 = 无界）
            windowSizeRight = in.windowSizeRight;
            debugFlag = (in.debugFlag != 0);   // 设备 printf 探针开关（env FLASH_ATTN_SPLITB_DEBUG）
            dumpFlag = (in.dumpFlag != 0);     // 设备 Dump Tensor 探针开关（env FLASH_ATTN_SPLITB_DUMP，devlog #44）
            softmaxOnly = (in.softmaxOnly != 0); // 只跑段1+2（env FLASH_ATTN_SPLITB_SOFTMAX_ONLY，devlog #44.25）
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
            // ws 四视图（gS/gP/gOTmp/gStats）依赖布局参数，移至几何段之后设置（devlog #44.44）

            uint32_t coreIdx = AscendC::GetBlockIdx();
#ifdef __DAV_C220_CUBE__
            // ① 硬件事件预置：内部 ping-pong 事件的首次 Wait 依赖预置的"已释放"初态（S3 实测教训：
            //    漏预置 + 漏 init 会挂死在 loadQGM/QK 内部）
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
            // ② BlockMmad 的 L1/L0 buffer 绑定（init 参数公式照抄 FAInfer :200-213）
            const uint32_t embedU = static_cast<uint32_t>(embed);
            uint32_t kDynNum = RoundUp(embedU, NUM_128);
            kDynNum = kDynNum < NUM_256 ? NUM_256 : kDynNum;
            uint32_t maxQKPL1Size = L1_MAX_SIZE - embedU * S2_STACK_LEN * sizeof(ElementQ);   // V tile L1 预留：D×128（SplitB 栈长，#36）
            uint32_t maxQL1Size = Q_TILE_CEIL * kDynNum * sizeof(ElementQ);
            uint32_t maxNDynNum =
                ((maxQKPL1Size - maxQL1Size) / kDynNum / sizeof(ElementQ) / DOUBLE_BUFFER) / NUM_32 * NUM_32;
            uint32_t nDynNum = maxNDynNum < L1_MAX_N_NUM ? maxNDynNum : L1_MAX_N_NUM;
            nDynNum = L1_MAX_N_NUM % nDynNum != 0 ? RoundDown((nDynNum - 1), NUM_32) : nDynNum;
            uint32_t l1QkSize = BlockMmadQK::L1TileShape::M * kDynNum * sizeof(ElementQ);
            blockMmadQK.init(resource, nDynNum, kDynNum, S2_STACK_LEN);
            uint32_t kPVDynNum = nDynNum * kDynNum / BlockMmadPV::L1TileShape::M;
            blockMmadPV.init(resource, nDynNum, kPVDynNum, S2_STACK_LEN, l1QkSize);
            blockMmadQK.resetBlockStart(0, pagedBlockSize);   // 非 paged：kvStart=0 恒（FAInfer 每 runMainLoop 前同式）
            blockMmadPV.resetBlockStart(0, pagedBlockSize);
#endif
#ifdef __DAV_C220_VEC__
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);   // 子核1 的 softmax 链（#44.24）
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            smEpilogue.init(resource, scaleValue, softcapValue);
            divoutEpilogue.init(resource);
            coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif

            // ---- 步长（照抄 FAInfer :237-240 BSND） ----
            strideQ = static_cast<uint64_t>(qHeads * embed);
            strideO = static_cast<uint64_t>(qHeads * embed);
            strideK = static_cast<uint64_t>(kvHeads * embed);
            strideV = static_cast<uint64_t>(kvHeads * embed);
            colsPad = RoundUp(static_cast<uint32_t>(kvSeqlen), FaiKenel::BLOCK_SIZE);   // = alignedS2（对齐 16）
            dPad = RoundUp(static_cast<uint32_t>(embed), FaiKenel::BLOCK_SIZE);
            blockStackNum = CeilDiv(S2_STACK_LEN, pagedBlockSize);   // 非 paged 且 PV 体内未使用（残留参数，#36）；=1

            // ---- workspace 布局（单位一律为 float 元素个数；与 splitb_host.cpp 公式严格一致）----
            // devlog #44.44 规范化（FAInfer 哲学）：每核两段连续，P 与 tile 区完全独立——
            //   coreWsOffset → [tile 区: batchBuf0 的 T tile 块 | batchBuf1 的 T tile 块]
            //             [P 区:   batchBuf0 的 T P 槽 | batchBuf1 的 T P 槽]
            //   每 tile 块 = [S 区 | OTmp 区 | stats 区]；gP 基址 = 本核 P 区首（独立指针）。
            // 历史注（#44.23/#44.35/#44.37）：P 曾复用 S 区（in-place→链式死区），批流水下
            //   跨 AIV 写读竞争致 P(b1,t0) s8-15 坏 → 全解耦。链式回归仅 S5 实测 GM 成瓶颈
            //   才考虑（stride-2 方案 #44.38 留档；softmax 的 MTE2_V 事件链未动）。
            sTileElems = static_cast<uint64_t>(Q_TILE_CEIL) * colsPad;   // 单 tile 的 S 区大小（fp32 元素计）
            pSlotElems = sTileElems / 2;                                  // 单 P 槽大小（128×colsPad 个 half = S 区一半，fp32 元素计）
            oTmpTileElems = static_cast<uint64_t>(Q_TILE_CEIL) * dPad;       // 单 tile 的 OTmp 区大小（PV 未归一 O，fp32）
            statsPerTask = 2 * static_cast<uint64_t>(Q_TILE_CEIL);    // 单 tile 的行统计区：max 128 + sum 128
            perTileElems = sTileElems + oTmpTileElems + statsPerTask;               // 单 tile 块总大小（三区之和；P 不占 tile 块）

            // ---- tile 几何（FAInfer :513-517，device 侧公式） ----
            curQNBlockTile = GetQNBlockTile(static_cast<uint32_t>(qSeqlen), static_cast<uint32_t>(groupSize));
            qNBlockNumPerGroup = CeilDiv(static_cast<uint32_t>(groupSize), curQNBlockTile);
            curQNBlockNum = qNBlockNumPerGroup * static_cast<uint32_t>(kvHeads);
            curQSBlockTile = GetQSBlockTile(static_cast<uint32_t>(kvSeqlen));
            curQSBlockNum = CeilDiv(static_cast<uint32_t>(qSeqlen), curQSBlockTile);
            tileNumPerBatch = static_cast<uint64_t>(curQNBlockNum) *
                static_cast<uint64_t>(curQSBlockNum);                 // 批内 tile 数 T
            perBatchTileElems = tileNumPerBatch * perTileElems;               // 单批 tile 区（ping/pong 槽内 tile 块连续）
            tileAreaElems = 2 * perBatchTileElems;                        // 本核 tile 区总大小（两批）
            perCoreElems = tileAreaElems + 2 * tileNumPerBatch * pSlotElems;
                                                                      // 单核配额 = tile 区 + P 区（每批 T 个独立 P 槽；批内
                                                                      //   tile 块不复用：批级 flag 下全批中间结果须并存）

            // ---- 核间 B 切分（aic 基数；CUBE 用原始 blockIdx、VEC 除以子核数，照 FAInfer :178/:235） ----
            const uint64_t coreWsOffset = static_cast<uint64_t>(coreIdx) * perCoreElems;  // 本核 workspace 基址偏移（float 元素计，= 前面所有核配额之和）

            // ---- ws 四视图（devlog #44.44 规范化：各视图基址 = 本核对应区首，寻址不再掺 coreWsOffset）----
            // gS/gOTmp/gStats → 本核 tile 区首；gP → 本核 P 区首（独立指针，FAInfer 哲学）
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
            AscendC::GlobalTensor<int32_t> gBlockTable;   // 非 paged 占位（FAInfer 非 paged 分支不触碰）

            GlobalTensorBundle globalTensors{
                gQ, gK, gV, gS, gP, gOTmp, gStats, gO, gLse, gBlockTable
            };
            int64_t batchStart = static_cast<int64_t>(coreIdx) * splitFactorSize;   // 本核负责的批区间起点（host 按核均分 B）
            int64_t batchEnd = batchStart + splitFactorSize;
            if (batchSize < batchEnd) {
                batchEnd = batchSize;                                 // 末核尾裁剪（B 不整除核数时）
            }

            // ============================ 批次循环 ============================
            // 照搬参考 Process() 的四段批结构（devlog #32/#34）：
            //   每次迭代内：CUBE 段1 QK 该 batch 全部 tile（catlass 逐 tile，rowNum 打包多头）
            //              VEC  段2 softmax 全部 tile（epilogue 内部双 AIV 拆行，stats→GM）
            //              CUBE 段3 PV 全部 tile（批次级等 softmaxReady 一次；自设满足内部 wait）
            //              VEC  段4 divout 全部 tile（epilogue 内部双 AIV 拆行，O/LSE 散射）
            // flag 每 batch 每 stage 一次；ping/pong 按 boIdx 奇偶。
            if (debugFlag) {
                AscendC::printf("[SB] c%u enter bs=%u be=%u tiles=%u\n", coreIdx,
                                (uint32_t)batchStart, (uint32_t)batchEnd,
                                curQNBlockNum * curQSBlockNum);
            }
            for (int64_t boIdx = batchStart; boIdx < batchEnd; ++boIdx) {
                runMainLoop(coreIdx, boIdx, globalTensors);
            }

            // ---- 收尾：事件全量 drain（照抄 FAInfer :372-410，保证异步拷贝全部落盘） ----
#ifdef __DAV_C220_CUBE__
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
#endif
#ifdef __DAV_C220_VEC__
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
#endif
            AscendC::PipeBarrier<PIPE_ALL>();

        }

        // 单个 batch 的四段计算（QK → Softmax → PV → DivO）。
        // 段序即文本序、计算序（FAInfer 的段序形态）：CUBE 核依序执行 段1→段3，VEC 核
        // 依序执行 段2→段4；跨核由 flag 链衔接（全批粒度，参考同款，devlog #39/#40）：
        //   段1 set qkReady（每批 1 次，广播双 AIV）→ 段2 各 AIV wait；
        //   段2 全部 tile 完成后 set softmaxReady → 段3 批级 wait 一次（PV 内部逐调用
        //   等待由 DispatchPolicyPV 的 WAIT_SOFTMAX=false 编译期关闭）；
        //   段3 set pvReady → 段4 wait。
        // 调试（devlog #42）：仅 debugFlag（env FLASH_ATTN_SPLITB_DEBUG）时的
        // AscendC::printf 探针，打印点在段首/同步 wait 之后——最后一条输出即把挂死
        // 定位到下一个 wait。
        __aicore__ inline void runMainLoop(
            uint32_t coreIdx,
            int64_t boIdx,
            GlobalTensorBundle& globalTensors
        ) {
            auto& gQ = globalTensors.gQ;
            auto& gK = globalTensors.gK;
            auto& gV = globalTensors.gV;
            auto& gS = globalTensors.gS;
            auto& gP = globalTensors.gP;
            auto& gOTmp = globalTensors.gOTmp;
            auto& gStats = globalTensors.gStats;
            auto& gO = globalTensors.gO;
            auto& gLse = globalTensors.gLse;
            auto& gBlockTable = globalTensors.gBlockTable;

            const uint64_t batchBuf = static_cast<uint64_t>(boIdx) % 2;   // ping/pong 槽（照搬参考）
            // batchBase 相对本核 tile 区首（视图已含 coreWsOffset，devlog #44.44）
            const uint64_t batchBase = batchBuf * perBatchTileElems;

            // ==================== 段1：QK 全部 tile → S 写批缓冲（CUBE） ====================
#ifdef __DAV_C220_CUBE__
            // if (debugFlag) {
            //     AscendC::printf("[SB] c%u b%u S1-QK\n", coreIdx, (uint32_t)boIdx);
            // }
            uint32_t kvSIdx = 0;            // 单 KV 栈（S2 不切分）：栈索引恒 0
            uint32_t kvSLoopNumTotal = 1;   // 栈数恒 1
            // FIXME: 理论而言，进入本kernel的QS不超过128，因此该循环次数最多为1次
            for (uint32_t qSBlockIdx = 0; qSBlockIdx < curQSBlockNum; ++qSBlockIdx) { // FIXME: 原先基于128的设计应该也考虑到UB的空间限制，但是限制QK计算相对独立，能否增大每次计算的tile大小，以减少循环次数？（128 是 L1/L0 硬件约束而非 UB 约束——L1TileShapeQK::M=Q_TILE_CEIL=128 与 L0A ping-pong 容量决定；要增大 M 需改共享 catlass tile 定义 + L1 预算公式，建议 S5 性能项评估，devlog #38 附记）
                for (uint32_t qNBlockIdx = 0; qNBlockIdx < curQNBlockNum; ++qNBlockIdx) {
                    const TileGeom tg = GetTileGeom(qSBlockIdx, qNBlockIdx);
                    const uint64_t gmQ = static_cast<uint64_t>(boIdx) * qSeqlen * strideQ +
                        static_cast<uint64_t>(qSBlockIdx) * curQSBlockTile * strideQ +
                        static_cast<uint64_t>(tg.qNStartIdx) * embed;
                    const uint64_t gmK = static_cast<uint64_t>(boIdx) * kvSeqlen * strideK +
                        static_cast<uint64_t>(tg.kvNIdx) * embed;
                    const uint64_t sOff = batchBase + tg.tileIdx * perTileElems;

                    LayoutQ layoutQTemp(tg.rowNum, static_cast<uint32_t>(embed));
                    LayoutK layoutKTemp(strideK, static_cast<uint32_t>(kvSeqlen));
                    LayoutS layOutS(tg.rowNum, static_cast<uint32_t>(kvSeqlen), colsPad);
                    uint32_t singleHead = tg.qNBlockSize;   // loadQGM 引用参数
                    uint32_t qHeadsP = static_cast<uint32_t>(qHeads);
                    blockMmadQK.loadQGM(gQ[gmQ], layoutQTemp, tg.rowNum, singleHead, qHeadsP);
                    GemmCoord actualBlockShapeQK{tg.rowNum, static_cast<uint32_t>(kvSeqlen),
                                                static_cast<uint32_t>(embed)};
                    blockMmadQK(gQ[gmQ], gK[gmK], gS[sOff], gBlockTable,
                            layoutQTemp, layoutKTemp, layOutS, actualBlockShapeQK,
                            kvSIdx, kvSLoopNumTotal, pagedBlockSize, strideK);
                }
            }
            // ---- 段1 dump（devlog #44.12/#44.15，逐 tile 有效区紧凑版）----
            // 整区方案曾超 1MB 预算（数据全丢只剩最后一条）。有效 S = 每 tile 前
            // rowNum×colsPad（本配置 colsPad=Sk 无 pad），desc = 100 + b*10 + tile。
            if (dumpFlag && coreIdx == 0) {
                AscendC::PipeBarrier<PIPE_FIX>();
                for (uint32_t qSb = 0; qSb < curQSBlockNum; ++qSb) {
                    for (uint32_t qNb = 0; qNb < curQNBlockNum; ++qNb) {
                        const TileGeom tgd = GetTileGeom(qSb, qNb);
                        const uint32_t descD = 100 + static_cast<uint32_t>(boIdx) * 10
                            + static_cast<uint32_t>(tgd.tileIdx);
                        AscendC::printf(
                            "[SB-DUMP] stage=QK(S_raw fp32) core=%u b=%u tile=%u qNStart=%u rows=%u cols=%u desc=%u\n",
                            coreIdx, (uint32_t)boIdx, (uint32_t)tgd.tileIdx,
                            tgd.qNStartIdx, tgd.rowNum, colsPad, descD);
                        const uint8_t dimD = 2;
                        uint32_t shapeD[2] = {tgd.rowNum, colsPad};
                        AscendC::ShapeInfo infoD(dimD, shapeD);   // 设备侧不能用初始化列表转指针（devlog #44.16）
                        AscendC::DumpTensor(gS[batchBase + tgd.tileIdx * perTileElems],
                                            descD, tgd.rowNum * colsPad, infoD);
                    }
                }
            }
            // AscendC::PipeBarrier<PIPE_ALL>();   // DBG 注入：阶段1 段边界二分（devlog #44.11）
            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);   // 每 batch 一次
#endif

            // ==================== 段2：softmax 全部 tile（VEC；每 tile 双 AIV 拆行） ====================
#ifdef __DAV_C220_VEC__
            Arch::CrossCoreWaitFlag(qkReady);   // 每 batch 一次
            if (debugFlag) {
                AscendC::printf("[SB] c%u v%u b%u S2-SM\n", coreIdx,
                                AscendC::GetSubBlockIdx(), (uint32_t)boIdx);
            }
            // ---- DBG 探针（devlog #44.35）：SM 消费前的 S 快照（desc=860+b*10+tile）----
            // 目的：指纹自洽（坏行 = "S≈0 输入"的正确 softmax 输出）→ 疑点在 SM 读到的
            // S 本身。P/S 解耦后 S 永不被覆写，此处即"SM 将要读到的 S"的直接观测：
            //   860 系坏（b1 t0 s8-15 ≈0/垃圾）→ S 在 [QK Fixpipe 完成, 本 dump] 窗口被写坏
            //   860 系对而 P@200 仍坏        → GM 的 S 没问题，错在 SM 的 GM→UB 搬运/事件链
            // 注意观测者效应：此处 dump 改变流水时序；若加探针后 P@200 变全对，本身即
            // "读侧时序竞争"的证据（devlog #44.34 三点一致 → 排除 PV/DO 追加写）。
            // if (dumpFlag && coreIdx == 0 && AscendC::GetSubBlockIdx() == 0) {
            //     for (uint32_t qSb = 0; qSb < curQSBlockNum; ++qSb) {
            //         for (uint32_t qNb = 0; qNb < curQNBlockNum; ++qNb) {
            //             const TileGeom tgd = GetTileGeom(qSb, qNb);
            //             const uint32_t descS = 860 + static_cast<uint32_t>(boIdx) * 10
            //                 + static_cast<uint32_t>(tgd.tileIdx);
            //             AscendC::printf(
            //                 "[SB-DUMP] stage=PRE(S_raw fp32) core=%u b=%u tile=%u qNStart=%u rows=%u cols=%u desc=%u\n",
            //                 coreIdx, (uint32_t)boIdx, (uint32_t)tgd.tileIdx,
            //                 tgd.qNStartIdx, tgd.rowNum, colsPad, descS);
            //             const uint8_t dimS = 2;
            //             uint32_t shapeS[2] = {tgd.rowNum, colsPad};
            //             AscendC::ShapeInfo infoS(dimS, shapeS);
            //             AscendC::DumpTensor(gS[batchBase + tgd.tileIdx * perTileElems],
            //                                 descS, tgd.rowNum * colsPad, infoS);
            //         }
            //     }
            // }
            if constexpr (MASK_TYPE == FaiKenel::MaskType::NO_MASK) {
                // FIXME: 理论而言，进入本kernel的QS不超过128，因此该循环次数最多为1次
                for (uint32_t qSBlockIdx = 0; qSBlockIdx < curQSBlockNum; ++qSBlockIdx) {
                    for (uint32_t qNBlockIdx = 0; qNBlockIdx < curQNBlockNum; ++qNBlockIdx) {
                        const TileGeom tg = GetTileGeom(qSBlockIdx, qNBlockIdx);
                        const uint64_t sOff = batchBase + tg.tileIdx * perTileElems;
                        const uint64_t statOff = sOff + sTileElems + oTmpTileElems;
                        // P 槽 half 偏移（与 gS 同款就地计算）：槽号 = 批内槽基 + tileIdx；
                        // pSlotElems 为 float 计，×2 换 half（gP 为 fp16 视图）
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
            // ---- 段2 dump（devlog #44.12/#44.15）：逐 tile 有效 P（half 紧凑）----
            // P 独立区（#44.35）有效数据 = 每 tile 前 rowNum×colsPad half。desc = 200 + b*10 + tile。
            if (dumpFlag && coreIdx == 0 && AscendC::GetSubBlockIdx() == 0) {
                AscendC::PipeBarrier<PIPE_MTE3>();
                for (uint32_t qSb = 0; qSb < curQSBlockNum; ++qSb) {
                    for (uint32_t qNb = 0; qNb < curQNBlockNum; ++qNb) {
                        const TileGeom tgd = GetTileGeom(qSb, qNb);
                        const uint32_t descD = 200 + static_cast<uint32_t>(boIdx) * 10
                            + static_cast<uint32_t>(tgd.tileIdx);
                        AscendC::printf(
                            "[SB-DUMP] stage=SM(P_unorm fp16) core=%u b=%u tile=%u qNStart=%u rows=%u cols=%u desc=%u\n",
                            coreIdx, (uint32_t)boIdx, (uint32_t)tgd.tileIdx,
                            tgd.qNStartIdx, tgd.rowNum, colsPad, descD);
                        const uint8_t dimD = 2;
                        uint32_t shapeD[2] = {tgd.rowNum, colsPad};
                        AscendC::ShapeInfo infoD(dimD, shapeD);
                        AscendC::DumpTensor(gP[(batchBuf * tileNumPerBatch + tgd.tileIdx) * pSlotElems * 2],
                                            descD, tgd.rowNum * colsPad, infoD);
                    }
                }
            }
            // 每 batch 一次（双 AIV 各自执行；FAInfer :849 同款段尾单点）：
            // PIPE_MTE3 保证在本分支全部 P/stats 拷贝之后才置位
            // AscendC::PipeBarrier<PIPE_ALL>();   // DBG 注入：阶段1 段边界二分（devlog #44.11）
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif

            // ==================== 段3：PV 全部 tile（CUBE） ====================
#ifdef __DAV_C220_CUBE__
            // softmaxOnly：整段跳过（含批级 wait/set——原循环条件写法漏掉循环外的
            // CrossCoreWaitFlag(softmaxReady) 与 CrossCoreSetFlag(pvReady)，会在
            // PV 不执行时仍消费/置位 flag，靠错位 set/wait 互相抵消才未挂死，#44.27）
            
            Arch::CrossCoreWaitFlag(softmaxReady);   // 批次级一次：等本批全部 softmax 完成
            //（消费 softmaxReady 后 continue——用户方案，flag 收支优于我方"不消费累积"
            // 设计：每批 set 数=wait 数，任意 B 安全，#44.28）
            if (!softmaxOnly) {
                if (debugFlag) {
                    AscendC::printf("[SB] c%u b%u S3-PV\n", coreIdx, (uint32_t)boIdx);
                }
                for (uint32_t qSBlockIdx = 0; qSBlockIdx < curQSBlockNum; ++qSBlockIdx) {
                    for (uint32_t qNBlockIdx = 0; qNBlockIdx < curQNBlockNum; ++qNBlockIdx) {
                        const TileGeom tg = GetTileGeom(qSBlockIdx, qNBlockIdx);
                        const uint64_t sOff = batchBase + tg.tileIdx * perTileElems;
                        const uint64_t oOff = sOff + sTileElems;
                        // P 槽 half 偏移（与 gS 同款就地计算；pSlotElems float 计 ×2 → half）
                        const uint64_t pOff = (batchBuf * tileNumPerBatch + tg.tileIdx) * pSlotElems * 2;
                        const uint64_t gmV = static_cast<uint64_t>(boIdx) * kvSeqlen * strideV +
                            static_cast<uint64_t>(tg.kvNIdx) * embed;

                        LayoutP layoutPTemp(tg.rowNum, static_cast<uint32_t>(kvSeqlen), colsPad);
                        LayoutV layoutVTemp(static_cast<uint32_t>(kvSeqlen), strideV);
                        LayoutOTmp layoutOTmpT(tg.rowNum, static_cast<uint32_t>(embed), dPad);
                        GemmCoord actualBlockShapePV{tg.rowNum, static_cast<uint32_t>(embed),
                                                    static_cast<uint32_t>(kvSeqlen)};
                        // softmaxFlag 的等待由 DispatchPolicyPV 的 WAIT_SOFTMAX=false 编译期
                        // 关闭（上方批级 Wait 已承担同步，#39/#40）
                        blockMmadPV(gP[pOff], gV[gmV], gOTmp[oOff], gBlockTable,
                                layoutPTemp, layoutVTemp, layoutOTmpT, actualBlockShapePV,
                                kvSIdx, kvSLoopNumTotal, pagedBlockSize,
                                static_cast<uint32_t>(kvSeqlen), strideV,
                                blockStackNum, softmaxReady);
                    }
                }
                // 段3 OTmp 不在此 dump：kernel 末尾的整区 float 视图 dump 一并覆盖
                //（OTmp/stats 此时全部就绪且不再被写，devlog #44.12）
                // AscendC::PipeBarrier<PIPE_ALL>();
                // // DBG 探针（devlog #44.35）：PV 后 S 区金丝雀（desc=810+b*10+tile）。
                // // P/S 解耦后 S 自 QK 写出后本应无人再写——此处与 PRE(860) 逐值一致 →
                // // 段3 清白；出现异物 → 段3 的写落进了 S 区（GM 混用实锤）。
                // if (dumpFlag && coreIdx == 0) {
                //     AscendC::PipeBarrier<PIPE_FIX>();
                //     for (uint32_t qSb = 0; qSb < curQSBlockNum; ++qSb) {
                //         for (uint32_t qNb = 0; qNb < curQNBlockNum; ++qNb) {
                //             const TileGeom tgd = GetTileGeom(qSb, qNb);
                //             const uint32_t descD = 810 + static_cast<uint32_t>(boIdx) * 10
                //                 + static_cast<uint32_t>(tgd.tileIdx);
                //             AscendC::printf(
                //                 "[SB-DUMP] stage=PPV(S_raw fp32) core=%u b=%u tile=%u qNStart=%u rows=%u cols=%u desc=%u\n",
                //                 coreIdx, (uint32_t)boIdx, (uint32_t)tgd.tileIdx,
                //                 tgd.qNStartIdx, tgd.rowNum, colsPad, descD);
                //             const uint8_t dimD = 2;
                //             uint32_t shapeD[2] = {tgd.rowNum, colsPad};
                //             AscendC::ShapeInfo infoD(dimD, shapeD);
                //             AscendC::DumpTensor(gS[batchBase + tgd.tileIdx * perTileElems],
                //                                 descD, tgd.rowNum * colsPad, infoD);
                //         }
                //     }
                // }
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);   // 每 batch 一次
            }
#endif

            // ==================== 段4：divout 全部 tile（VEC；每 tile 双 AIV 拆行） ====================
#ifdef __DAV_C220_VEC__
            // softmaxOnly：段4 整段跳过（含批级 WaitFlag(pvReady)——否则 VEC 会等
            // 一个 CUBE 侧（段3 被 continue 跳过）永远不会 set 的 flag 而挂死）
            if (!softmaxOnly) {
                Arch::CrossCoreWaitFlag(pvReady);   // 每 batch 一次
                if (debugFlag) {
                    AscendC::printf("[SB] c%u v%u b%u S4-DO\n", coreIdx,
                                    AscendC::GetSubBlockIdx(), (uint32_t)boIdx);
                }
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
            }
#endif   // 段4 #ifdef __DAV_C220_VEC__（devlog #44.15：dump 块移动时曾误删此 endif）
            // ---- 段4 后 dump（devlog #44.15：kernel 末尾 dump 来不及刷出，移入 batch
            // 循环内运行中时机；与段1/段2 同款）----
            // WS 终态逐 tile：OTmp=310+b*10+tile（rowNum×dPad）、stats=330+b*10+tile（2×rowNum）
            // O=400+b（本 batch 区，fp16 行主序 s*strideO+h*embed+d）
            // LSE=450+b（本 batch 区，[H,Sq] 头主序 h*Sq+s）
#ifdef __DAV_C220_VEC__
            if (dumpFlag && coreIdx == 0 && AscendC::GetSubBlockIdx() == 0) {
                AscendC::PipeBarrier<PIPE_MTE3>();
                for (uint32_t qSb = 0; qSb < curQSBlockNum; ++qSb) {
                    for (uint32_t qNb = 0; qNb < curQNBlockNum; ++qNb) {
                        const TileGeom tgd = GetTileGeom(qSb, qNb);
                        const uint64_t tileBase = batchBase + tgd.tileIdx * perTileElems;
                        // desc=600+b*10+tile（devlog #44.41：原 310+b*10 在 b≥2 时与
                        // stats 家族 330+b*10 撞号——B=4 的 OTmp"错误"实为 parser 读到
                        // stats 记录的假象，O/LSE 全对已证 kernel 无恙）
                        const uint32_t descO = 600 + static_cast<uint32_t>(boIdx) * 10
                            + static_cast<uint32_t>(tgd.tileIdx);
                        const uint32_t descS = 330 + static_cast<uint32_t>(boIdx) * 10
                            + static_cast<uint32_t>(tgd.tileIdx);
                        if (softmaxOnly) {
                            // 剥离模式（devlog #44.26）：段3/4 未运行，OTmp/O/LSE 为垃圾
                            // 不 dump（Q：用户指出）；stats 仍 dump（softmax 产物）
                            const uint8_t dimS2 = 2;
                            uint32_t shapeS2[2] = {1, static_cast<uint32_t>(statsPerTask)};
                            AscendC::ShapeInfo infoS2(dimS2, shapeS2);
                            AscendC::printf(
                                "[SB-DUMP] stage=STATS(max,sum fp32) core=%u b=%u tile=%u qNStart=%u rows=%u layout=max[0..128)+sum[128..256) desc=%u\n",
                                coreIdx, (uint32_t)boIdx, (uint32_t)tgd.tileIdx,
                                tgd.qNStartIdx, tgd.rowNum, descS);
                            AscendC::DumpTensor(gS[tileBase + sTileElems + oTmpTileElems],
                                                descS, static_cast<uint32_t>(statsPerTask), infoS2);
                            continue;
                        }
                        AscendC::printf(
                            "[SB-DUMP] stage=OTmp(fp32) core=%u b=%u tile=%u qNStart=%u rows=%u cols=%u desc=%u\n",
                            coreIdx, (uint32_t)boIdx, (uint32_t)tgd.tileIdx,
                            tgd.qNStartIdx, tgd.rowNum, dPad, descO);
                        const uint8_t dimD = 2;
                        uint32_t shapeD[2] = {tgd.rowNum, dPad};
                        AscendC::ShapeInfo infoD(dimD, shapeD);
                        AscendC::DumpTensor(gS[tileBase + sTileElems], descO,
                                            (tgd.rowNum / 2) * dPad, infoD);
                        AscendC::printf(
                            "[SB-DUMP] stage=STATS(max,sum fp32) core=%u b=%u tile=%u qNStart=%u rows=%u layout=max[0..rows)+sum[128..128+rows) desc=%u\n",
                            coreIdx, (uint32_t)boIdx, (uint32_t)tgd.tileIdx,
                            tgd.qNStartIdx, tgd.rowNum, descS);
                        // stats 整块（max [0..rowNum)、sum [128..128+rowNum)，行距
                        // ROW_NUM_MAX=128；devlog #44.15：连续 dump 2×rows 只覆盖
                        // max+未写区，须 dump 整块 statsPerTask）
                        const uint8_t dimS = 2;
                        uint32_t shapeS[2] = {1, static_cast<uint32_t>(statsPerTask)};
                        AscendC::ShapeInfo infoS(dimS, shapeS);
                        // 192 floats：max [0..128) + sum [128..192)（rowNum≤64 时 sum 全覆盖）
                        AscendC::DumpTensor(gS[tileBase + sTileElems + oTmpTileElems],  // FIXME: Stats为什么用gS
                                            descS, static_cast<uint32_t>(statsPerTask), infoS);
                    }
                }
                if (!softmaxOnly) {   // 段4 未运行不 dump O/LSE（devlog #44.26）
                {
                    const uint8_t dimD = 2;
                    uint32_t shapeD[2] = {static_cast<uint32_t>(qSeqlen),
                                          static_cast<uint32_t>(strideO)};
                    AscendC::ShapeInfo infoD(dimD, shapeD);
                    AscendC::printf(
                        "[SB-DUMP] stage=O(fp16 final) core=%u b=%u rows=%u cols=%u layout=s*%u+h*%u+d desc=%u\n",
                        coreIdx, (uint32_t)boIdx,
                        static_cast<uint32_t>(qSeqlen), static_cast<uint32_t>(embed * qHeads),
                        static_cast<uint32_t>(strideO), static_cast<uint32_t>(embed),
                        400 + static_cast<uint32_t>(boIdx));
                    AscendC::DumpTensor(gO[static_cast<uint64_t>(boIdx) * qSeqlen * strideO],
                                        400 + static_cast<uint32_t>(boIdx),
                                        static_cast<uint32_t>(qSeqlen * strideO),
                                        infoD);
                }
                {
                    const uint8_t dimD = 2;
                    uint32_t shapeD[2] = {static_cast<uint32_t>(qHeads),
                                          static_cast<uint32_t>(qSeqlen)};
                    AscendC::ShapeInfo infoD(dimD, shapeD);
                    AscendC::printf(
                        "[SB-DUMP] stage=LSE(ln(sum)+max) core=%u b=%u heads=%u tokens=%u layout=h*%u+s desc=%u\n",
                        coreIdx, (uint32_t)boIdx,
                        static_cast<uint32_t>(qHeads), static_cast<uint32_t>(qSeqlen),
                        static_cast<uint32_t>(qSeqlen),
                        450 + static_cast<uint32_t>(boIdx));
                    AscendC::DumpTensor(gLse[static_cast<uint64_t>(boIdx) * qHeads * qSeqlen],
                                        450 + static_cast<uint32_t>(boIdx),
                                        static_cast<uint32_t>(qHeads * qSeqlen),
                                        infoD);
                }
                }   // if (!softmaxOnly)——O/LSE dump 门控闭合（devlog #44.26）
            }
#endif
        }

    private:
        // tile 几何（FAInfer :513-541 的 qNBlockIdxCurGroup/kvNIdx/qNStartIdx/rowNum 段；
        // 四段各需一次故提为成员函数）
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
        int64_t windowSizeLeft;   // S4：SWA 窗口（FAInfer 语义）
        int64_t windowSizeRight;
        bool debugFlag;
        bool dumpFlag;
        bool softmaxOnly;
        int64_t splitFactorSize;

        uint64_t strideQ;
        uint64_t strideO;
        uint64_t strideK;
        uint64_t strideV;
        uint32_t pagedBlockSize;   // paged KV 页大小（tiling.blockSize；非 paged 仅占位穿透，FAInfer 同名）
        uint32_t colsPad;
        uint32_t dPad;
        uint32_t blockStackNum;

        // workspace 布局（与 splitb_host.cpp 公式严格一致，改动必须同步）。
        // 命名规约（devlog #44.44，对齐 FAInfer 元素计数语义如 MAX_UB_S_ELEM_NUM）：
        //   *Elems = float 元素个数（gS/gOTmp/gStats 为 float 视图；P 区按 half 存，
        //   gP 寻址时 ×2 换算，见各段就地计算的 pOff）。FAInfer 区段对照：S 区=mm1Out、
        //   P 区=smOnlineOut、OTmp 区=mm2Out（我方为逐 tile 交错结构，故不直接借用其区段名）。
        uint64_t sTileElems;
        uint64_t pSlotElems;       // 单 P 槽
        uint64_t oTmpTileElems;
        uint64_t statsPerTask;
        uint64_t perTileElems;
        uint64_t tileNumPerBatch; // 批内 tile 数 T
        uint64_t perBatchTileElems;   // 单批 tile 区 = T × perTileElems
        uint64_t tileAreaElems;   // 本核 tile 区 = 2 × perBatchTileElems（gP 基址在本核此偏移之后）
        uint64_t perCoreElems;

        // tile 几何（FAInfer :513-517 device 侧公式）
        uint32_t curQNBlockTile;
        uint32_t qNBlockNumPerGroup;
        uint32_t curQNBlockNum;
        uint32_t curQSBlockTile;
        uint32_t curQSBlockNum;

        Arch::Resource<ArchTag> resource;
        Arch::CrossCoreFlag qkReady{QK_READY_ID};
        Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
        Arch::CrossCoreFlag pvReady{PV_READY_ID};

        BlockMmadQK blockMmadQK;
        BlockMmadPV blockMmadPV;
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
        GM_ADDR tiling)
    {
        // 设备侧 printf：AscendC::printf（无 fflush——输出走调试通道，kernel 结束刷出；
        // bring-up 临时探针，调通后随 #42 清单移除）
        // AscendC::printf("4001 [SB] blk=%u kernel launched\n", AscendC::GetBlockIdx());

        AscendC::SetSyncBaseAddr(fftsAddr);
        (void)mask;   // S4：causal/SWA mask 经 softmax 的 mask 重载

        using ArchTag = Arch::AtlasA2;
        using ElementQ = InputDtypeQ;
        using LayoutQ = layout::RowMajor;
        using ElementK = InputDtypeQ;
        using LayoutK = layout::ColumnMajor;
        using ElementV = InputDtypeQ;
        using LayoutV = layout::RowMajor;
        using ElementS = float;
        using LayoutS = layout::RowMajor;
        using ElementP = InputDtypeQ;
        using LayoutP = layout::RowMajor;
        using ElementOTmp = float;
        using LayoutOTmp = layout::RowMajor;

        using L1TileShapeQK = GemmShape<Q_TILE_CEIL, 128, 128>;
        using L0TileShapeQK = GemmShape<128, 128, 128>;
        using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQKT<false, false>;
        using QType = Gemm::GemmType<ElementQ, LayoutQ>;
        using KType = Gemm::GemmType<ElementK, LayoutK>;
        using SType = Gemm::GemmType<ElementS, LayoutS>;
        using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK,
                                                   QType, KType, SType>;

        using L1TileShapePV = GemmShape<128, 128, 256>;
        using L0TileShapePV = GemmShape<128, 128, 128>;
        // 第三参 WAIT_SOFTMAX=false：PV 逐 tile 调用不做内部 softmaxFlag 等待——
        // SplitB 为批粒度同步（段3 批级 Wait 一次，devlog #39/#40）；FAInfer 缺省 true 不变
        using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPVT<false, false, false>;
        using PType = Gemm::GemmType<ElementP, LayoutP>;
        using VType = Gemm::GemmType<ElementV, LayoutV>;
        using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
        using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV,
                                                   PType, VType, OTmpType>;

        using SplitBSoftmaxEpilogue = SplitBSoftmax<ElementQ, HAS_SOFTCAP>;
        using SplitBDivOutEpilogue = SplitBDivOut<ElementQ>;

        using SplitBKernelType = SplitBKernel<
            BlockMmadQK, BlockMmadPV, SplitBSoftmaxEpilogue, SplitBDivOutEpilogue, maskCategory>;

        FAIKernelParams params{q, k, v, mask, nullptr, nullptr, nullptr, o, lse, workspace, tiling};
        SplitBKernelType splitBKernel;
        // AscendC::printf("4002 [SB] blk=%u kernel type created\n", AscendC::GetBlockIdx());
        splitBKernel(params);
        // AscendC::printf("4003 [SB] blk=%u kernel invoked\n", AscendC::GetBlockIdx());
    }
}

#endif // MHA_FWD_SPLITB_CPP_

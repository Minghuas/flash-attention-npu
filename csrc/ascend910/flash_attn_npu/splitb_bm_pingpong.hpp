/**
 * Copyright (c) 2026, perf-shortSeqLargeBatch project.
 *
 * SplitB 专用 Pingpong BlockMmad 引擎——自 catlass submodule（csrc/catlass，
 * gitcode.com/cann/catlass v1.6.1）include/catlass/gemm/block/block_mmad_pingpong.hpp
 * **逐行 fork**（devlog #46.4，用户指令：依赖库不可改，需定制则拷贝到本目录——
 * qk_matmul.hpp / pv_matmul.hpp 即同款先例）。
 *
 * 【与上游的差异（仅此三处，其余逐行一致）】
 * 1. 独立类 SplitBBlockMmad（非库模板 BlockMmad 的偏特化——库内已存在
 *    MmadAtlasA2Pingpong 的特化，重复特化属 ODR 冲突）；
 * 2. 构造器尾参 presetEvents（默认 true = 上游语义）：多实例共存的宿主
 *   （SplitB 的 QK+PV 双引擎共享物理事件 ID）传 false——宿主全权接管事件
 *    生命周期：①沿用宿主 kernel 级预置（避免双 ctor 对同 ID 连续 Set，
 *    Set-on-set 语义未证实，#44.53g）；②析构 drain 跳过——否则与宿主
 *    kernel drain / 第二实例析构重复 Wait 同一批 ID → 出口挂死（#46.2
 *    实测：三子核 EEEE 全打出后卡死）；
 * 3. 头文件自包含 using（上游靠外围 namespace 语境）；
 * 4. [v3.3] 新增 AOperandBSHD + operator() BSHD 重载（A 面 [B,S,H,D] 布局的
 *    可选打包头组自装——GQA 打包/MHA 单头统一形态）与私有 loadAPackedBSHD。
 *
 * 【为什么 fork 它（v3 引擎选型，devlog #46）】A/B 面 L1 双缓冲
 * （l1A/BTensorList[STAGES=2]）+ 两向事件全保护 + 跨调用槽位轮转——
 * t19 定罪的 FAI 引擎 l1A 单缓冲竞态类结构性关闭（#45.3）。
 *
 * 上游同步提示：catlass 升级时对照 block_mmad_pingpong.hpp 重新 fork，
 * 差异仅上述四点（diff 可见）。
 */

#ifndef SPLITB_BM_PINGPONG_HPP_
#define SPLITB_BM_PINGPONG_HPP_

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/matrix_coord.hpp"

namespace SplitB {

using namespace Catlass;

template <
    class DispatchPolicy_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_ = void,
    class TileCopy_ = Gemm::Tile::TileCopy<typename DispatchPolicy_::ArchTag, AType_, BType_, CType_, BiasType_>,
    class TileMmad_ = Gemm::Tile::TileMmad<typename DispatchPolicy_::ArchTag, AType_, BType_, BiasType_>
>
struct SplitBBlockMmad {
public:
    // Type Aliases
    using DispatchPolicy = DispatchPolicy_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t STAGES = 2;   // MmadAtlasA2Pingpong 固有（上游 DispatchStagesGetter）
    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;

    // Check ArchTag
    static_assert(std::is_same_v<ArchTag, Arch::AtlasA2> || std::is_same_v<ArchTag, Arch::Ascend950>,
        "ArchTag can only be AtlasA2 or Ascend950!");

    // Check LayoutC
    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    // Check L1TileShape
    static_assert((L1A_SIZE * STAGES + L1B_SIZE * STAGES) <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");

    // Check L0TileShape
    static constexpr uint32_t L0A_TILE_SIZE = L0TileShape::M * L0TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L0TileShape::M * L0TileShape::N * sizeof(ElementAccumulator);
    static_assert((L0A_TILE_SIZE * STAGES) <= L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert((L0B_TILE_SIZE * STAGES) <= L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE <= L0C_SIZE, "L0TileShape exceeding the L0C space!");

    // Check tileshape
    static_assert(L1TileShape::M == L0TileShape::M && L1TileShape::N == L0TileShape::N,
        "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");
    static_assert(L0TileShape::K <= L1TileShape::K, "L0TileShape::K cannot exceed L1TileShape::K");

    // 32B (256b) aligned
    static_assert(Gemm::helper::TileShapeAlignChecker<L1TileShape, L0TileShape, ElementA, ElementB>::_ALIGN == 256,
        "Tile shape must be 32B aligned.");

    /// Construct（差异点 2：presetEvents 尾参，语义见文件头）
    CATLASS_DEVICE
    SplitBBlockMmad(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0, bool presetEvents = true)
        : hostOwnedEvents_(presetEvents)
    {
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1A_SIZE * STAGES;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_SIZE * i);
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_SIZE * i);
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);

            // Assign event ID for each stages
            l1AEventList[i] = i;
            l1BEventList[i] = i + STAGES;
            l0AEventList[i] = i;
            l0BEventList[i] = i + STAGES;

            // The event id that needs to be set before the loop
            if (presetEvents) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
        }
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
        if (presetEvents) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }
    }

    /// Destructor（差异点 2：宿主接管模式跳过 drain，语义见文件头）
    CATLASS_DEVICE
    ~SplitBBlockMmad()
    {
        if (!hostOwnedEvents_) {
            return;
        }
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    /// [SplitB v3.3] A 面操作数描述：BSHD 布局 [B, S, H, D] 的（可选打包）Q 头组。
    /// 头内相邻 token 行距 = rowStride（= qHeads·D）；同 s 相邻头距 = K 维（D）——
    /// 引据此在 operator() BSHD 重载内自装 A 面。headNum=1（MHA）退化为标准单头装载。
    struct AOperandBSHD {
        uint32_t rowsPerHead;   // 每头行数（qS 块行数；行区 fractal 对齐要求 16 的倍数）
        uint32_t headNum;       // 打包头数（共享同一 B；MHA=1）
        int64_t rowStride;      // BSHD 头内行距（= qHeads·D）
    };

    /// BSHD 重载：A 面按 AOperandBSHD 语义装载（内部逐头 Nd2Nz 到 zN 行区），随后
    /// 走主 operator() 流程。约定：K ≤ L1TileShape::K（kTileCount==1，闸门 D≤128
    /// 内恒真）；rowsPerHead 为 16 的倍数（fractal 对齐）。
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, AOperandBSHD const &a,
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmC, LayoutC const &layoutC,
        GemmCoord const &actualShape)
    {
        uint32_t kActual = min(actualShape.k(), L1TileShape::K);
        loadAPackedBSHD(gmA, a.rowsPerHead, a.headNum, a.rowStride, kActual);
        // 主流程的首次 A 装载已由 aPrefilledA_ 跳过；此处的 layoutA 仅其 K 预取
        // 路径（本重载约定 kTileCount==1，不触达）引用。
        LayoutA layoutA(a.rowsPerHead * a.headNum, static_cast<uint32_t>(actualShape.k()),
                        a.rowStride);
        (*this)(gmA, layoutA, gmB, layoutB, gmC, layoutC, actualShape);
    }

    /// [SplitB O1] GQA 组内 B 面（K/V）L1 驻留：kv 头 tile 固定驻留 B 槽 0，跨组内
    /// q 头复用（省 G-1 次 GM→L1 拷贝；prof 实证 MTE2 为 cube 核第一忙管道）。
    /// 事件纪律（HardEvent 单 bit set 语义，收支严格平衡，#44.53g 教训）：
    ///   setBResident（每组一次）：Wait<MTE1_MTE2>(ID2) 后拷贝；不 Set——"数据有效"
    ///     旗标由各头的 re-arm 承担
    ///   operator()（每头）：起始 Set<MTE2_MTE1>(ID2) re-arm → 首 (k,n) tile Wait 消费
    ///     → 末 tile Set<MTE1_MTE2>(ID2)（供下一头/下一组的 wait 消费）
    ///   waitBResidentPrev（组内 g>0 头调用前）：Wait<MTE1_MTE2>(ID2)，消费上一头
    ///     末 tile 的释放
    ///   收支：每组 waits=1(load)+(G-1)(prev)=G，sets=G（每头末 tile）→ 组内净零；
    ///   launch 内最后一批末头的 +1 由 kernel drain Wait(ID2) 消费；槽 1（ID3）全程
    ///   不用 → kernel 预置/排水须按 residentQK 跳过 ID3（孤儿预置 = 跨 launch 死锁）。
    ///   限制：K≤L1TileShape::K（kTileCount==1，闸门内 D/Sk≤128 恒真；k 预取路径
    ///   不支持驻留）。
    CATLASS_DEVICE
    void setBResident(AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
                      uint32_t kActual, uint32_t nActual)
    {
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[0]);
        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActual, nActual));
        copyGmToL1B(l1BTensorList[0], gmB, layoutBInL1, layoutTileB);
        bResident_ = true;
    }

    /// [SplitB O1] 组内非首头的记账 Wait（见 setBResident 注释的收支表）
    CATLASS_DEVICE
    void waitBResidentPrev()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[0]);
    }

    /// [SplitB O1] 组尾退出驻留（下一组或后续走回标准装载）
    CATLASS_DEVICE
    void clearBResident()
    {
        bResident_ = false;
    }

    /// Perform a block-scoped matrix multiply-accumulate
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmC, LayoutC const &layoutC,
        GemmCoord const &actualShape)
    {
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());

        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mRound, nRound));

        uint32_t kActual = min(actualShape.k(), L1TileShape::K);

        // load first matrix A tile from GM to L1（[v3.3] loadAPackedBSHD 预装后跳过；
        // layoutTileA 声明保留在块外——下方 K 预取路径复用（上游结构））
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActual));
        if (!aPrefilledA_) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
            copyGmToL1A(l1ATensorList[l1ListId], gmA, layoutAInL1, layoutTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
        }

        // load first matrix B tile from GM to L1（[O1] 驻留模式跳过拷贝——组内 K/V 已由
        // setBResident 装入槽 0；本头 re-arm 补给"数据有效"旗标，收支见 setBResident 注释）
        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActual, actualShape.n()));
        if (bResident_) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[0]);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
            copyGmToL1B(l1BTensorList[l1ListId], gmB, layoutBInL1, layoutTileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
        }

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);  // FIXME: 为什么不用ping-pong，L0C的空间应该够用？
        }

        uint32_t mPartLoop = CeilDiv<L0TileShape::M>(mRound);
        uint32_t nPartLoop = CeilDiv<L0TileShape::N>(nRound);

        // main loop
        uint32_t kTileCount = CeilDiv<L1TileShape::K>(actualShape.k());
        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
            uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
            uint32_t kActualNext{0};
            // preload next tile from GM to L1
            if (kLoopIdx < kTileCount - 1) {
                uint32_t kLoopIdxNext = kLoopIdx + 1;
                kActualNext = (kLoopIdxNext < kTileCount - 1) ?
                    L1TileShape::K : (actualShape.k() - kLoopIdxNext * L1TileShape::K);

                // Get L1 tensor for next stage
                auto l1ATensor = l1ATensorList[l1ListIdNext];
                auto l1BTensor = l1BTensorList[l1ListIdNext];
                // Get GM tile for next stage
                MatrixCoord gmTileAOffset{0, kLoopIdxNext * L1TileShape::K};
                MatrixCoord gmTileBOffset{kLoopIdxNext * L1TileShape::K, 0};
                auto gmTileA = gmA[layoutA.GetOffset(gmTileAOffset)];
                auto gmTileB = gmB[layoutB.GetOffset(gmTileBOffset)];

                // load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActualNext));
                copyGmToL1A(l1ATensor, gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load next matrix B tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                layoutTileB = layoutB.GetTileLayout(MakeCoord(kActualNext, actualShape.n()));
                copyGmToL1B(l1BTensor, gmTileB, layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
            }

            // Get L1 tensor for current stage（[O1] B 面驻留时固定读槽 0，不随 l1ListId 轮转）
            auto l1ATensor = l1ATensorList[l1ListId];
            auto l1BTensor = bResident_ ? l1BTensorList[0] : l1BTensorList[l1ListId];

            // Get the loop nums on L0
            uint32_t kPartLoop = CeilDiv<L0TileShape::K>(kActual);

            for (int mPartIdx = 0; mPartIdx < mPartLoop; mPartIdx++) {
                uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ?
                    L0TileShape::M : (mRound - mPartIdx * L0TileShape::M);

                for (int kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
                    uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ?
                        L0TileShape::K : (kActual - kPartIdx * L0TileShape::K);

                    // Locate the current tile on L0A
                    auto l0ATile = l0ATensorList[l0AListId];
                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mPartActual, kPartActual);
                    // Locate the current tile of matrix A on L1
                    MatrixCoord l1AOffset{mPartIdx * L0TileShape::M, kPartIdx * L0TileShape::K};
                    auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1AOffset)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    if ((mPartIdx == 0) && (kPartIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                    }

                    // Load current tile from L1 to L0A
                    copyL1ToL0A(l0ATile, l1ATile, layoutAInL0, layoutAInL1);

                    if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                    }

                    for (int nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
                        uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ?
                            L0TileShape::N : (nRound - nPartIdx * L0TileShape::N);

                        // Locate the current tile on L0B
                        auto l0BTile = l0BTensorList[l0BListId];
                        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kPartActual, nPartActual);
                        // Locate the current tile of matrix B on L1
                        MatrixCoord l1BOffset{kPartIdx * L0TileShape::K, nPartIdx * L0TileShape::N};
                        auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BOffset)];

                        // Wait for mmad finished
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        // If the current tile is the first one on the k&n axis, wait for loading matrix B from GM to L1
                        if ((kPartIdx == 0) && (nPartIdx == 0)) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(
                                l1BEventList[bResident_ ? 0 : l1ListId]);
                        }

                        // Load current tile from L1 to L0B
                        copyL1ToL0B(l0BTile, l1BTile, layoutBInL0, layoutBInL1);

                        // If the current tile is the last one on the k&n axis, notify to load matrix B from GM to L1
                        if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                                l1BEventList[bResident_ ? 0 : l1ListId]);
                        }
                        // Notify to do mmad
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        // Locate the current tile on L0C
                        MatrixCoord l0COffset{mPartIdx * L0TileShape::M, nPartIdx * L0TileShape::N};
                        auto l0CTile = l0CTensor[layoutInL0C.GetOffset(l0COffset)];

                        // Compute the matrix multiplication on L0A and L0B and write the result to the accumulator
                        // Wait for loading L0B
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                        bool initC = ((kLoopIdx == 0) && (kPartIdx == 0));
                        // If the unit flag is enabled, the unit flag is set according to the calculation progress
                        uint8_t unitFlag = 0b00;
                        if constexpr (ENABLE_UNIT_FLAG) {
                            if ((kLoopIdx == kTileCount - 1) && (mPartIdx == mPartLoop - 1) &&
                                (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                                unitFlag = 0b11;
                            } else {
                                unitFlag = 0b10;
                            }
                        }
                        // Perform calculation operations
                        tileMmad(l0CTile, l0ATile, l0BTile, mPartActual, nPartActual, kPartActual, initC, unitFlag);

                        // Notify to move the next L0B tile
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        l0BListId = (l0BListId + 1 < STAGES) ? (l0BListId + 1) : 0;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    l0AListId = (l0AListId + 1 < STAGES) ? (l0AListId + 1) : 0;
                }
            }
            l1ListId = l1ListIdNext;
            kActual = kActualNext;
        }

        // copy block out
        LayoutC layoutBlock = layoutC.GetTileLayout(actualShape.GetCoordMN());

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            copyL0CToGm(gmC, l0CTensor, layoutBlock, layoutInL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        } else {
            copyL0CToGm(gmC, l0CTensor, layoutBlock, layoutInL0C, 0b11);
        }
        aPrefilledA_ = false;   // [v3.3] 预装标志按调用消费（下一次调用默认自装）
    }

private:
    /// [SplitB v3.3] BSHD 条带 A 面装载（operator() BSHD 重载的内部实现）：逐头调用
    /// 标准 4 参 Nd2Nz（语义已被 MHA 路径上板验证），头 h 的 [rowsPerHead × K] 写入
    /// L1 槽 zN 排布的行区 [h·rowsPerHead, ...)（zN GetOffset；rowsPerHead 16 对齐
    /// ⇒ fractal 对齐，由 BSHD 重载约定保证）。事件纪律与引擎内置装载一致；
    /// 置 aPrefilledA_ 令主 operator() 跳过其首次 A 装载。
    /// 不用 FAI loadQGM 的 9 参条带重载：其目的 stride 按 zN(rowNum,K) 定容推导，
    /// 与本引擎 zN(L1TileM, L1TileK) 槽寻址仅在 rowNumRound==L1TileM 时重合。
    CATLASS_DEVICE
    void loadAPackedBSHD(AscendC::GlobalTensor<ElementA> const &gmA,
                         uint32_t rowsPerHead, uint32_t headNum,
                         int64_t rowStride, uint32_t kActual)
    {
        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
        for (uint32_t h = 0; h < headNum; ++h) {
            LayoutA layoutHead(rowsPerHead, kActual, rowStride);
            int64_t dstOff = layoutAInL1.GetOffset(MatrixCoord{h * rowsPerHead, 0});
            copyGmToL1A(l1ATensorList[l1ListId][dstOff], gmA[h * kActual],
                        layoutAInL1, layoutHead);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
        aPrefilledA_ = true;
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> l1ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    // Multi-stage event id list
    int32_t l1AEventList[STAGES];
    int32_t l1BEventList[STAGES];
    int32_t l0AEventList[STAGES];
    int32_t l0BEventList[STAGES];

    // The id of current stage
    uint32_t l1ListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};

    // 宿主接管事件生命周期标记（= ctor presetEvents 实参）
    bool hostOwnedEvents_{true};

    // [v3.3] loadAPackedBSHD 预装标志（operator() 首次 A 装载据此跳过，调用末尾复位）
    bool aPrefilledA_{false};

    // [O1] B 面驻留标志（setBResident 置位、clearBResident 复位；驻留槽恒 0）
    bool bResident_{false};

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};

} // namespace SplitB

#endif // SPLITB_BM_PINGPONG_HPP_

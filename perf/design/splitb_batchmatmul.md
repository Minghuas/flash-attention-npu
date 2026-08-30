# SplitB QK/PV 批 matmul 路线设计（v3 引擎替换）

> 日期：2026-08-30。前置：devlog #45/#45.1-#45.4（v2 批间错位流水 + Bug⑥ t19 定案）。
> 决策（用户拍板）：弃用 tile 级 FAI 引擎（qk/pv_matmul.hpp），改用 catlass 通用
> BatchedMatmul 引擎路线；batch 维 = dim1 = G*kvN（qHeads）。
> 本文档两部分：① 当前分 tile 方案的存档记录（被替换者）；② 新方案设计。

---

## 第一部分：当前分 tile 实现方案（v1→v2 存档）

### 1. 任务模型（FAInfer 照搬，devlog #34）

- 核间只切 B 轴（aic 基数）：本核负责 `[coreIdx*splitF, ...)` 连续 batch；
- **批内分 tile**：`for qSBlockIdx × for qNBlockIdx` 双重循环，tile = (qS 块 × qN 块)；
  - `curQSBlockTile = GetQSBlockTile(kvSeqlen)`，`curQSBlockNum = CeilDiv(Sq, curQSBlockTile)`；
  - `curQNBlockTile = GetQNBlockTile(Sq, group)`（**多头打包**：把 qNBlockSize 个 q 头的
    qSBlockSize 行拼成 rowNum = qS×qN ≤ 128 行，GQA 时同 kv 头的头才可同 tile）；
  - `tileNumPerBatch T = CeilDiv(Sq,qsTile) × CeilDiv(G, qNTile) × kvN`。
- 每 tile 一次 `blockMmadQK(...)` / `blockMmadPV(...)`（FAInfer 的 `MmadAtlasA2FAIQKT/
  FAIPVT` 偏特化，L1Tile 128×128×128 / 128×128×256，A 面 **l1A 单缓冲** + 分离的
  `loadQGM()` GM→L1 装载调用）。

### 2. workspace 布局（#44.44，不变）

```
每核 coreWsOffset → [tile 区: 2 批槽 × T tile 块] + [P 区: 2 批槽 × T P 槽]
每 tile 块 = [S 区(Q_TILE_CEIL×colsPad fp32) | OTmp 区(128×dPad fp32) | stats 区(256 fp32)]
P 槽 = 128×colsPad half（S 区一半大小，fp32 元素计）
ping/pong = boIdx % 2
```

### 3. v2 批间错位流水（#45，保留不动）

```
CUBE 迭代 t：QK(bo_t) → wait sm(bo_{t-1}) → wait do(bo_{t-3}) → PV(bo_{t-1})
VEC  迭代 t：wait qk(bo_{t-1}) → softmax(bo_{t-1}) → wait pv(bo_{t-2}) → divout(bo_{t-2})
flag：qk(1)/sm(2)/pv(3)/do(4)，mode2，每批收支 1:1；哨兵 CUBE+3/VEC+2
```

### 4. 已定罪缺陷（t19，#45.3）

tile 循环里 `loadQGM(t+1)` 的 MTE2 写（GM→l1A）与上一 tile 的 `CopyL1ToL0A`（l1A→L0A，
`AscendC::LoadData` 矩阵装载，**疑似不在 MTE1 管线域**）并发 → S 首行撕裂
（比值 (h+2)/(h+1) = Q(h+1)·K(h)）。EV5+PipeBarrier 守卫无效（等空管）。FAInfer 原版
不触发 = loadQGM 与下任务隔整个 KV 长循环；PV 免疫 = A 面 L1 双缓冲。

---

## 第二部分：Batch matmul 路线设计（v3）

### 5. 语义（用户指定，= 参考实现的 batch 维）

```
QK：batch = qHeads（= G*kvN，dim1）
     每批 A = Q[b,h]   [Sq, D]  行主序（GM 上天然连续，批步长 Sq*D）
         B = K[b,kvN(h)] 的转置视图 [D, Sk]（ColumnMajor over [Sk,D] 行主序存储）
         C = S[b,h]     [Sq, Sk] fp32（行步长 colsPad）
     GQA：h → kvN(h) = h / G（循环内自算偏移，GQA 广播退化为寻址）
PV：batch = qHeads
     每批 A = P[b,h]   [Sq, Sk] fp16（行步长 colsPad）
         B = V[b,kvN(h)] [Sk, D] 行主序（k=Sk, n=D）
         C = OTmp[b,h]  [Sq, D]  fp32（行步长 dPad）
```

**为何不直接用 `BatchedMatmul` kernel 类**：它的 `operator()<AIC>` 自带
`GetBlockIdx()/GetBlockNum()` 网格跨步循环（全核调度）+ 空的 AIV 分支，是**独立算子**
launch 形态；我们的 v2 融合流水（CUBE/VEC 双程序 + 4 flag）要求段内自控循环。
**取其引擎、弃其调度**：per-head 循环留在 StageQK/StagePV 内（与 BatchedMatmul 内层
循环逐行同构，batched_matmul.hpp:144-169）。

### 6. 引擎选型（example 01 已验证组装，逐项照抄）

```cpp
using ArchTag      = Arch::AtlasA2;
using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;   // ENABLE_UNIT_FLAG（同 example）
using L1TileShape  = GemmShape<128, 128, 128>;   // K≤128：kTileCount=1，无 K 分块
using L0TileShape  = GemmShape<128, 128, 64>;    // 同 example 的 L0 K 粒度
QK: AType=GemmType<half, RowMajor>, BType=GemmType<half, ColumnMajor>, CType=GemmType<float, RowMajor>
PV: AType=GemmType<half, RowMajor>, BType=GemmType<half, RowMajor>,   CType=GemmType<float, RowMajor>
using BlockMmadX = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
```

引擎关键性质（block_mmad_pingpong.hpp 实读）：
1. **A/B 面 L1 双缓冲**（`l1ATensorList/l1BTensorList[STAGES=2]`），两向事件全保护：
   写前 `Wait<MTE1_MTE2>(slot)` / 写后 `Set<MTE2_MTE1>(slot)`；读侧对称（290/297/315/323）。
   **跨调用槽位轮转**（l1ListId 成员持续）→ 头 h+1 的 GM→L1A 装载与头 h 的 L1→L0A
   读取永远异槽 —— t19 竞态类结构性关闭，不依赖任何未确认的管线域语义；
2. L0A/L0B 亦双缓冲 + `M_MTE1` 槽位闸；Mmad 前 `MTE1_M` 闸；copyout 走 unit flag
   （ENABLE_UNIT_FLAG=true，M_FIX-free 的 mfix 同步，example 同款）；
3. 调用形态：`blockMmad(gmA[off], layoutA, gmB[off], layoutB, gmC[off], layoutC, {m,n,k})`
   —— **无分离的 loadQGM**，尾形（Sq/Sk 非 16 对齐）由 mRound/nRound + actualShape 处理；
4. 构造器 `BlockMmad(resource, l1BufAddrStart=0)`：L1 布局 `[l1A×2 | l1B×2]` 自
   l1BufAddrStart 起，且**预置事件**（MTE1_MTE2{0..3}、M_MTE1{0..3}、FIX_M ID0）。

### 7. 集成设计

#### 7.1 StageQK（CUBE）新形态

```cpp
for (uint32_t h = 0; h < qHeads; ++h) {           // batch 维 = 头（GQA: kvN = h/G）
    const uint64_t gmQ = bo*Sq*strideQ + h*Sq*embed;             // A 基址（头连续）
    const uint64_t gmK = bo*Sk*strideK + (h/G)*Sk*embed;         // B 基址（kv 头）
    const uint64_t sOff = batchBase + h*perTileElems;            // C：沿用 tile 块布局
    LayoutQ layoutA(Sq, embed);                  // RowMajor（带步长构造，同现状）
    LayoutK layoutB(embed, Sk, strideK);         // ColumnMajor 转置视图（同现状语义）
    LayoutS layoutC(Sq, Sk, colsPad);            // RowMajor 带行步长（同现状三参构造）
    blockMmadQK(gQ[gmQ], layoutA, gK[gmK], layoutB, gS[sOff], layoutC,
                GemmCoord{Sq, Sk, embed});
}
```
StagePV 对称（A=P 槽、B=V、C=OTmp，`GemmCoord{Sq, embed, Sk}`）。

#### 7.2 workspace / host：**零布局改动，几何简化**

- tile 块布局原样保留（`qNBlockTile ≡ 1` 语义下 tileIdx == headIdx，T == qHeads ×
  curQSBlockNum），softmax/divout epilogue、stats/P 寻址、dump 族全部不动；
- `GetTileGeom` 简化：去 qN 打包维（qNBlockSize≡1、qNStartIdx=h、kvNIdx=h/G）；
- host（splitb_host.cpp）：`qNBlockTile` 钉死为 1（nTilePerBatch 公式同步）。

#### 7.3 v2 流水：**原封不动**

flag 编排（qk/sm/pv/do）、哨兵迭代、守卫表、debug printf 全部保留——引擎替换是
CUBE 段内的事，AIC/AIV 协作拓扑不变。

#### 7.4 事件域整合（唯一的 [需 NPU 验证] 点）

两引擎实例共享物理事件 ID（MTE1_MTE2{0..3} 等），交替调用按 MTE1 FIFO 论证良性
（同现状 QK/PV 共号）；**但构造器预置会双份 Set**（kernel 预置 + 引擎1 ctor + 引擎2
ctor）——若 HardEvent 为单 bit 语义（#44.53g），Set-on-set 挂死风险。处置：
- 方案 A（首选）：给构造器加 `bool presetEvents = true` 尾参（默认值保 example 兼容，
  共享文件零行为差异），SplitB 两实例传 `false`，沿用 kernel 级预置（现状会计已验证）；
- 方案 B（备选）：先原样试（若事件为计数语义则天然平衡），挂死再回退方案 A。

### 8. 性能权衡（诚实记录）

- **小 Sq 的 M 维利用率**：per-head M=Sq（如 32/9 行）vs 旧方案的 qN 打包 M=128。
  这是**参考实现同款语义**（其 batch=heads、s1Blk≤128 亦逐头），其收益来自批间 API
  摊销而非 M 打包；GQA 的 G 头共 kv 打包（M=G*Sq）留作 S5（注意 MHA G=1 无法打包，
  因打包头须共享同一 B 矩阵）。
- K 分块消失（D≤128 一趟）、KV 栈概念整体移除（S2 不切的彻底版）。

### 9. 迁移步骤（建议顺序）

1. 引擎构造器加 `presetEvents` 尾参（方案 A，默认 true）；
2. StageQK 换引擎 + per-head 循环（删 loadQGM/EV5/PipeBarrier 守卫/旧 qk 调用），
   `--softmax-only` 模式验证 S/stats（dump 100/890/700 族现成）；
3. StagePV 换引擎，全链验证（B4/B16 H8 Sq32 Sk96 ×8 应 0 FAIL）；
4. pytest splitb 全量 + bench2 对比 v1/v2 基线；
5. 清理：GetTileGeom 简化、host qNBlockTile 钉 1、删 qk/pv_matmul.hpp include。

### 10. 风险清单

| 风险 | 处置 |
|---|---|
| TileCopy 不支持 (half,ColMajor,half,RowMajor,float) 组合 | 编译期即暴露（模板 static_assert），可加特化 |
| RowMajor(m,n,stride) 三参构造在通用引擎的 GetTileLayout 路径 | 与 FAI 路同源（tile_copy.hpp 共享），编译+smoke 验证 |
| 双实例事件 ID 交互 | §7.4 方案 A/B |
| L1 预算（QK 128KB + PV 128KB = 256KB < 512KB） | static_assert 兜底 |
| 尾形（Sq=9/Sk=47）mRound/nRound 路径 | engine 内建；用既有奇形用例回归 |

### 附：资料索引（用户提供的路线依据）

- 引擎：`csrc/catlass/include/catlass/gemm/block/block_mmad_pingpong.hpp`（与独立仓库
  `/data0/liaojy/workspace/FA/catlass` 同源）
- kernel 类：`gemm/kernel/batched_matmul.hpp`（通用版；内层循环 144-169 为我们的
  per-head 循环模板；`MmadMultiBatch` 特化缺引擎，仍是脚手架，#45.4）
- 已验证组装：独立仓库 `examples/01_batched_matmul/batched_matmul.cpp`
  （MmadAtlasA2Pingpong<true> + 128×256×256/128×256×64 + GemmIdentityBlockSwizzle）
- 文档：catlass docs/zh/1_Practice/03_kernel_development.md §6.1（BatchedMatmul 扩展点）

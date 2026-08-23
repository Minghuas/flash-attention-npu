# SplitB 内存空间布局解读（GM + UB）

> 用途：掌握当前 S/P/OTmp/stats 在 GM workspace 与 UB 的完整空间方案、数据流与
> 生命周期，避免后续调优（尤其 softmax tiling）时误判。
> 对应代码：`mha_fwd_splitb.cpp`（布局注释 + GetPHalfIdx）、`splitb_host.cpp`（公式）、
> `splitb_softmax.hpp` / `splitb_divout.hpp`（UB 偏移）。
> 历史决策：devlog #34（初版）、#44.23（P 链式独立区根因修复）。

---

## 一、GM workspace 布局

### 1.1 分层结构（float 元素计，除非注明）

```
workspace
├── 核 0 段 [coreWsF = 0]
│   ├── batchBuf0（本核偶数批）          ← boIdx % 2 == 0 的批用
│   │   ├── tile 0 块
│   │   ├── tile 1 块
│   │   ├── ...
│   │   └── P-scratch 槽（批尾）        ← 新增（#44.23）
│   └── batchBuf1（本核奇数批）          ← boIdx % 2 == 1 的批用，同构
├── 核 1 段 [coreWsF = 1 × perCoreF]
└── ...
```

### 1.1.1 为什么批槽分奇偶（ping-pong 双缓冲）

每核的批顺序处理，但**内存操作异步飞行**——相邻批的流水是重叠的：

```
批 b：   CUBE: QK(b) ──→ PV(b) ──┐
         VEC:        SM(b) ──→ DO(b)   ← DO(b) 的 MTE2 读 OTmp/stats、MTE3 写 O 还在飞行
批 b+1： CUBE:                  QK(b+1) ← 此时已开跑（CUBE 只等 softmaxReady(b)，不等 DO(b)）
```

- 同槽则 QK(b+1) 写 S 会砸到 DO(b) 未读完的 OTmp/stats → 数据错乱；
- 分奇偶两槽则两侧各用各的，安全重叠（**批间重叠正是吞吐来源**，串行等待会砍半性能）；
- 批 b+2 复用 b 的槽时隔了 b+1 完整四段 + flag 链，b 的异步操作必然已排空——深度 2 恰好够；
- 来源：参考 TilingB 原设计；FAInfer 同思想（PRE_LAUNCH+1 个槽，深度=在飞数+1）。

### 1.2 每 tile 块（perTileF = s1AreaF + oAreaF + statsPerTask）

| 区       | 大小（float）              | 内容                                    | 写者→读者                                  |
| -------- | -------------------------- | --------------------------------------- | ------------------------------------------- |
| S 区     | `s1AreaF = 128×colsPad` | QK 原始分数（fp32，未乘 scale）         | 段1 QK(Fixpipe) → 段2 softmax(MTE2 读)     |
| OTmp 区  | `oAreaF = 128×dPad`     | PV 未归一输出（fp32）                   | 段3 PV(Fixpipe) → 段4 divout(MTE2 读)      |
| stats 区 | `statsPerTask = 256`     | max[0..128) + sum[128..256)（行距 128） | 段2 softmax(MTE3 写) → 段4 divout(MTE2 读) |

- **行距统一 128**（= Q_TILE_CEIL）：S/OTmp/stats 都按 128 行对齐编址，行 r 的
  S 在 `tileBase + r×colsPad`、OTmp 在 `tileBase + s1AreaF + r×dPad`、
  max 在 `tileBase + s1AreaF + oAreaF + r`、sum 在 `+128`。
- 有效行只有 rowNum（≤128），尾部的 128−rowNum 行是 padding。

### 1.3 P 区：链式独立布局（#44.23 根因修复，核心设计）

**P（fp16，softmax 归一前的 exp 值）不再与 S 共用空间**：

```
P[t=0]  → 批尾 P-scratch 槽（batchBase + T×perTileF 起，大小 pScratchF = s1AreaF/2）
P[t≥1]  → S[t-1] 的死区（softmax[t-1] 已读完 S[t-1]，此后无人再读）
```

- P 槽大小 = 128 行 × colsPad **half** = s1AreaF/2 float（fp16 是 fp32 一半）。
- 统一寻址：`GetPHalfIdx(batchBase, tileIdx)`（返回 half 索引，供 `gP[...]` 用）。
- **为什么不能原地覆写 S**（根因）：FAInfer 的 in-place 设计在 tile 级流水下合法
  （QK→softmax 生命周期紧凑），但 SplitB 是批流水——段2 循环内 softmax[t] 的
  MTE2 读 GM[S[t]] 与写 P 到同一批字节冲突 → P 写入确定性丢失（-O0 可复现）。
- **链式安全性依赖**：softmax[t] 顺序后于 softmax[t-1] 的 S 读完成（MTE2_V 事件链）。
  ⚠ 改 softmax tiling（行块划分/tile 顺序/并发）前必须重验此前提。
- 空间开销：每批 +1 个 P 槽（朴素分离是 +T 个）。

### 1.4 总量公式（splitb_host.cpp 与 kernel 严格同步）

```
perTileF   = s1AreaF + oAreaF + statsPerTask
perBatchF  = T × perTileF + pScratchF        （T = 批内 tile 数）
perCoreF   = 2 × perBatchF                   （ping/pong 批槽）
workspace  = ceil(B/核数) × perCoreF（512B 对齐）
```

### 1.5 GM 数据流总图（单 batch，T 个 tile）

```
段1 QK(CUBE)   : gQ,gK ──Mmad──→ S[t]（Fixpipe 写 S 区）
段2 softmax(VEC): S[t] ──MTE2──→ lsUb ──V 计算──→ lpUb(P) ──MTE3──→ P 区（链式）
                  max/sum ──MTE3──→ stats 区
段3 PV(CUBE)   : P[t] + gV ──Mmad──→ OTmp（Fixpipe 写 OTmp 区）
段4 divout(VEC): OTmp + stats ──MTE2──→ UB ──V 除──→ gO（最终输出 GM）
```

跨核同步（批粒度 + #44.19 恢复的逐 tile 握手）：qkReady / softmaxReady / pvReady。

---

## 二、UB 布局（softmax 与 divout 分时复用同一 256KB UB）

### 2.1 softmax 阶段（splitb_softmax.hpp，偏移照抄 FAInfer）

| tensor    | 偏移  | 大小                 | 用途                                          |
| --------- | ----- | -------------------- | --------------------------------------------- |
| lsUb      | 0     | 2 槽 × 32KB（64KB） | S 载入 + exp 计算（fp32；行块 ping-pong）     |
| lpUb      | 64KB  | 2 槽 × 16KB（32KB） | P（fp16 cast 输出；行块 ping-pong）           |
| tvUb      | 160KB | ~4KB                 | 行 max/sum 归约 scratch + max 广播（Brcb）    |
| lmUb      | 168KB | 512B                 | 行 max（fp32，≤128 行）                      |
| llUb      | 171KB | 512B                 | 行 sum（fp32，≤128 行）                      |
| softcapUb | 168KB | —                   | 与 lmUb 同址（softcap 先于 rowmax，时序复用） |

### 2.2 divout 阶段（splitb_divout.hpp，与 softmax **分时**复用）

| tensor | 偏移       | 用途                                                              |
| ------ | ---------- | ----------------------------------------------------------------- |
| goUb   | 128KB      | OTmp 载入 + 除法 + cast（fp32/fp16 双视图）                       |
| tvUb   | 160KB      | 除数/LSE 广播（Brcb）——**与 softmax tvUb 同址但不同时段** |
| gmUb   | 160KB+10KB | stats max 载入                                                    |
| glUb   | 160KB+12KB | stats sum 载入                                                    |
| lseUb  | 160KB+12KB | 与 glUb 同址（Ln 就地读 gl 写 lse）                               |

> 注意：softmax(段2) 与 divout(段4) 之间隔着段3 PV（CUBE），同核 VEC 上
> 时序不重叠，UB 分时复用安全——但若未来压缩段间间隔/流水化，需复查。

### 2.3 关键 in-place 与复用点（易误判清单）

| # | 复用                        | 为什么安全                   | 何时会破                              |
| - | --------------------------- | ---------------------------- | ------------------------------------- |
| 1 | ~~P 覆写 S~~（已废除）     | —                           | **批流水下非法（#44.23 根因）** |
| 2 | P[t]→S[t-1] 死区           | softmax 顺序执行 + MTE2_V 链 | softmax tiling 改动                   |
| 3 | softcapUb = lmUb 同址       | softcap 先于 rowmax          | 若 softcap 后置                       |
| 4 | lseUb = glUb 同址           | Ln 就地读改                  | 若 LSE 需独立存活                     |
| 5 | softmax/divout 的 tvUb 同址 | 段2/段4 时序分离             | 段间流水化                            |
| 6 | UB 内 lsUb/lpUb ping-pong   | 行块预取（preLoad=1）        | rowLoopNum=1 时实际单槽               |

---

## 三、数据尺寸速查（本测例 B2/S32/Sk32/H2/D64）

| 量                                   | 值                                      |
| ------------------------------------ | --------------------------------------- |
| colsPad / dPad                       | 32 / 64                                 |
| s1AreaF / pScratchF / oAreaF / stats | 4096 / 2048 / 8192 / 256                |
| perTileF / perBatchF(T=2)            | 12544 / 27136 float                     |
| P[t=0] 位置                          | batchBase + 25088（批尾）               |
| P[t=1] 位置                          | batchBase + 0（= S[0] 区首，half 视图） |

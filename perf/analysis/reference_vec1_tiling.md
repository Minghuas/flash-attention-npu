# Vec1 专项解读：单 Batch 任务的 softmax tiling 与双层循环

> 对象：[flash_attention_score_bn2gs1s2_b.h](../../ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h) 的 `ProcessVec1()`（源码已加【解读】注释）
> 回答的核心问题：**一个 batch 的任务包含完整的 N2、G、S1、S2，它们如何结合 UB 空间
> 约束做 tiling，形成 `for 头 × for S1 行块` 的双层循环完成计算？**
> 日期：2026-08-16｜配套：[深度解读](reference_splitb_deep_dive.md) §3/§7

---

## 1. 问题设定：一个 boIdx 要算什么

进入 Vec1 时，cube 侧的 `IterateBmm1` 已经把**本 batch 全部头的 S 矩阵**一次算完，落在
GM workspace 的 mm1Res 区（fp32）：

```
S 区域（一个 boIdx）:  [N2×G, S1, S2]   fp32
                      └─ 头 h 的 S^h = [S1 × S2]（行=S1 的 q 位置，列=S2 的 kv 位置）
```

Vec1 要做的：对每个头的每行做 softmax（scale→[mask]→max→exp→sum），产出
`P = exp(S·scale - rowmax)`（未归一）写回 GM 的 stage1Res 区（fp16），并把每行的
`rowmax/rowsum` 统计写到 GM 的 softmaxMax/Sum 区（供 Vec2 归一用）。

**约束**：S/P 数据在 GM 可以很大（触发闸门内 ≤256KB fp32），但**向量计算必须经 UB**，
A2 每 vector 核 UB 192KB、本模板分给 Vec1 的 S 工作块只有 **32KB**（stage1Ping/PongBuf
= 8K 元素 × 4B）。所以必须把 `[N2×G, S1, S2]` 切成 ≤32KB 的小块逐块搬进 UB 计算。

## 2. tiling 决策：三个轴三种处理

| 轴                    | 处理                                                                                                                  | 依据                                                                                                                                                                                        |
| --------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **S2（列）**    | **不切**：`s2BasicBlock = alignedS2`，`s2OuterSize = 1`                                                     | **算法要求**。softmax 的归约维就是 S2——切了 S2 就要跨块合并 max/sum（即我方 online_softmax 的 dm/gm 状态机），小 SeqLen 场景的固定开销就回来了。S2 整块进 UB 是"单遍 softmax"的前提 |
| **S1（行）**    | **按 UB 预算切**：`s1BaseSize = align16(8192 / S2)`，再 min(S1, 256)；`s1OuterSize = ceil(S1 / s1BaseSize)` | 行块面积 ≤ 8K fp32 元素 = 32KB（UB 预算）；16 对齐是 DataCopy 快路径要求；上限 256 是 host 经验值                                                                                          |
| **N2×G（头）** | **串行循环**：`biN2GoIdx ∈ [0, biN2G)`                                                                       | 头之间完全独立（各自的 S/P/stats），无需切分维度，逐个处理即可                                                                                                                              |

于是三维空间被压成**双层循环**：

```
for biN2GoIdx = 0 .. N2×G-1:                 // 头（外层）
    for loopIdx = 0 .. s1OuterSize-1:        // S1 行块（内层）
        处理 tile = [s1BaseSize × S2]  (≤ 8K 元素 / 32KB fp32)
        尾块行数 = s1BaseTailSize
每 batch 总块数 = (N2×G) × s1OuterSize
```

**为什么头在外、S1 块在内**：头的 GM 寻址是粗粒度跳转（每次换头，S 区域跨 S1×S2×4B），
S1 块在同一头内连续（相邻块只差 s1BaseSize×S2×4B）——内层连续对外层跳跃，对 GM 访问
局部性和 stats 攒批（§6）都更友好。

**注意 cube 与 vector 的粒度差**：cube 侧 BMM1 用 `IterateBatch` 一次吃下全部 N2×G 个
`[S1×D]×[D×S2]`（batch 维批处理），Vec1 却要按 32KB 块扫回——两级粒度不同是设计使然：
cube 面向吞吐（batch 摊销指令开销），vector 面向 UB 物理约束。

## 3. 数值走查（两个典型 shape）

**例 A：S1=S2=64, H=8（N2=8, G=1），fp16**（触发闸门：8×64×64×2B=64KB ≤128KB ✓）

```
s2BasicBlock = 64（不切）
s1BaseSize   = align16(8192/64) = 128 → min(alignedS1=64, 256) → 64
s1OuterSize  = 1                     ← S1 也无需切
每 batch：8 头 × 1 块 = 8 块，每块 [64×64] fp32 = 16KB（预算 32KB 的一半）
```

**例 B：S2=16, S1=512, H=8**（闸门：8×512×16×2B=128KB ✓ 压线通过）

```
s2BasicBlock = 16（不切）
s1BaseSize   = align16(8192/16) = 512 → min(512, 256) → 256   ← 撞 256 上限
s1OuterSize  = ceil(512/256) = 2     ← S1 切成 2 块
每 batch：8 头 × 2 块 = 16 块，每块 [256×16] fp32 = 16KB；尾块 s1BaseTailSize=256（整除无尾）
```

例 B 展示了双层循环真正"双层都转"的情形；例 A（我们 bench 的主场景）内层退化为 1 次。

## 4. 双层循环内的执行流（一个 tile 的完整生命周期）

```
进入 (biN2GoIdx, loopIdx)：
 ① WaitFlag(V_MTE2 事件A)            —— loopIdxNew>0 时：上块占用的 MTE2 通路已释放
 ② [hasPse] PSE 偏置块 GM→UB          —— 位置编码（我方 ALiBi 对应物，v1 不启用）
 ③ GetBmm1Result：S 块 GM→UB          —— 落入 stage1PongTensor（32KB）
    16 对齐走 DataCopy 直拷；否则 DataCopyPad 补 0 到 s2AlignSize
    GM 源：mm1Res{Ping/Pong}[biN2GoIdx×S1×S2 + loopIdx×s1BaseSize×S2]
 ④ SetFlag/WaitFlag(MTE2_V)           —— S 搬入完成，V 管线可见
 ⑤ Muls(scale)（在 Pong 上）
 ⑥ [hasPse] PseCompute：加偏置
 ⑦ CopyInAttenMask(-1)：mask 块 GM→UB —— 偏移由压缩模式算（causal/band/prefix…）
 ⑧ Muls(1.0)：Pong → Ping             —— 数据搬到另一个 32KB buffer（Ping 为计算 buffer）
 ⑨ [hasAtten] ComputeAttenMask        —— SelectWithBytesMask：被遮位置 → -inf
 ⑩ [hasDrop] dropMask 拷入
 ⑪ SoftMaxCompute                     —— SoftmaxFlashV2 单遍（原地）：max→sub→exp→sum
    · P（未归一）留在 Ping
    · 每行 rowmax/rowsum 写入 stats UB 区（见 §6 的攒批机制）
 ⑫ [hasDrop] ComputeDropMask
 ⑬ WaitFlag(MTE3_V)                   —— 上块的 P 写回（MTE3）已排空，Ping 可复用
 ⑭ Cast fp16 → vecOut（16KB） + DataCopyPad → GM stage1Res
    （T==INPUT_T 时跳过 cast 直接搬）
 ⑮ SetFlag(MTE3_V)
 循环推进（loopIdxNew 跨头连续计数，供 ①⑬ 的跨块事件保护）
```

**双 buffer（Ping/Pong）的作用**：本块的 S 搬入（MTE2 → Pong）与上块的 P 写回
（MTE3 ← Ping）在两个 buffer 上重叠进行；⑬ 的 MTE3_V 等待保证覆写 Ping 前上块已搬完。
这是 tile 级的 double-buffer 流水，与 boIdx 级的 3 槽流水（Process 主循环）是两个层次。

## 5. UB 布局回顾（Vec1 相关区段，总 ~187KB/192KB）

| Buffer                        | 大小               | 角色                                                                                |
| ----------------------------- | ------------------ | ----------------------------------------------------------------------------------- |
| stage1PingBuf / stage1PongBuf | 各 8K×4B = 32KB   | S 块工作对：Pong=输入落点，Ping=softmax 计算 buffer（③⑧ 的两步搬运即 Pong→Ping） |
| commonTBuf                    | 64×128×4B = 32KB | SoftmaxFlashV2 / SelectWithBytesMask 的 API 临时区                                  |
| softmaxSum/Max Ping/Pong      | 各 256×32B = 8KB  | 行统计攒批区（§6）；Ping/Pong 按**boIdx 奇偶**轮转（跨 batch 双缓冲）        |
| maskTBufPing/Pong             | 11KB / 16KB        | attenMask / dropMask+PSE 输入                                                       |
| vecOut                        | 16KB               | fp16 cast 输出（P 写回前的落脚）                                                    |

32KB tile 预算即由此而来：stage1 buffer 8K 元素 → `s1BaseSize × S2 ≤ 8192`。

## 6. 行统计的攒批与刷写（softmaxCopyOutLimit）

rowmax/rowsum 是**每行一个标量**。逐块算出后不立即写 GM，而是攒在 UB 的 stats 区：

```
softmaxBufSize      = 256 行（UB 攒批容量）
softmaxCopyOutLimit = 256 / s1BaseSize          —— 能攒几个 S1 块
softmaxCopyOutSize  = min(s1OuterSize, limit) × s1BaseSize   —— 每次刷写的行数
```

刷写条件（SoftMaxCompute 尾部）：`loopIdx == s1OuterSize-1 || (loopIdx+1) % limit == 0`
——攒满或到头再一次性 `DataCopy` 到 GM。GM 布局注意：**每行统计按 8 元素（fp32 一个
32B block）padding 存储**（向量指令按 block 处理的代价）：

```
softmaxSumGm: [B, N2×G, S1, 8] fp32  （有效值在每 8 元素组的第 0 个）
```

**关键理解**：因为 S2 不切分，每行的统计在**它所属的唯一块内就是终值**——攒批只是
GM 写效率优化，**不是**跨块数值合并（对照：online_softmax 的 stats 攒批是真·跨块累加）。

## 6.5 设计问答：为什么不在 Vec1 直接归一化 P？（2026-08-16 用户提问）

数学上可行（`P/sum · V ≡ (P·V)/sum`，行和是标量；此处 S2 整行处理、sum 当场已知）。
推迟到 Vec2 是三个工程理由：

1. **精度（主因）**：P 以 fp16 存 GM。未归一的 `P = exp(s−rowmax)` 最大元素恰为 1.0，
   值域贴 fp16 良好区间；Vec1 归一会把全体再除以 sum（≤128），尾部小值被推进
   fp16 非规格化区（min normal ≈6.1e-5），误差经 BMM2 累积进 O。推迟除法到 Vec2
   则作用在 fp32 的 O 上、每行一次、只在最终 cast 前舍入一次。
   （Tri Dao FA2/FA3 同款设计：P 不归一，1/l 修正推迟到 O 的 epilogue。）
2. **模板家族统一**：同家族的 S1s2Bn2gs1（S2>1024）/S1Bn2gs1 是真·online softmax，
   S2 切块时 sum 未定、Vec1 归一在算法上不可能——"P 未归一 + Vec2 末端归一"是家族
   共用骨架，SplitB 沿用而非特化第三种数据流。
3. **性能（次因）**：除法次数 S1×D（O 元素数）vs S1×S2（P 元素数），D<S2 时更少。

我方 splitb_softmax/divout 沿用同契约（理由 1 对单模板依然成立）。

## 7. Vec2 为什么用不同的行块（s1Vec2BaseSize）

Vec2 处理的是 O tile `[行 × D]`（不是 `[行 × S2]`），行块大小由 D 决定：
`s1Vec2BaseSize = align16(8192 / alignedD) × (2/inputDtypeBytes)`——同样守住 8K 元素预算。
例：D=128 → s1Vec2BaseSize = 64；D=64 → 128。所以 Vec1 与 Vec2 的内层块数不同、
互不同步，各自独立扫（中间靠 GM 的 OTmp/stats 交接）。

## 8. 与我方 splitb_softmax.hpp 的映射对照

| 参考 Vec1                            | 我方 splitb_softmax.hpp（v3）                                             | 说明                                                                 |
| ------------------------------------ | ------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `for biN2GoIdx`（头）              | `for h`（kernel 主循环的 h 层）                                         | 同                                                                   |
| `for loopIdx`（S1 行块）           | `for s1o`（kernel 主循环）                                              | 同                                                                   |
| tile`[s1BaseSize × S2]` ≤8K 元素 | tile`[s1Base × colsPad]`，s1Base=align16(8K/S2) min(…, **128**) | 上限 128 是 catlass L1Tile M 约束（因地制宜项）                      |
| （tile 就是 UB 上限，无更细切分）    | `for chunk`（64 行/块）再切一层                                         | 我们 softmax 内部按 64 行分块（SB_ROW_CHUNK），行为等价、UB 用量减半 |
| SoftmaxFlashV2 库调用                | 手写 max→sub→exp→sum 原语序列                                          | 参考走库；我们不可用该库（GE 框架依赖），手写等价原语                |
| stats 攒批 256 行 + 8 元素 padding   | stats 每 (h,slot) 区直写 GM（无攒批）                                     | 简化项；行统计在单块内即终值的性质相同                               |
| P 写 stage1Res（独立 GM 分区）       | P 原地覆写 S tile 起点（fp16 半宽）                                       | 等价，省一半 workspace                                               |
| boIdx 级 ping/pong（taskId 奇偶）    | slot = 任务号 % 2 双缓冲                                                  | 同思想，我们槽粒度是 (boIdx,h,s1o) 任务                              |

## 9. 尾块与对齐细节清单（迁移对照用）

1. `s1BaseTailSize = S1 - (s1OuterSize-1)×s1BaseSize`：末块行数，`vecS1BaseSize` 取尾值
2. `s2AlignSize`（16 对齐）vs `s2Size`（实际）：S 搬入走 DataCopy（对齐）或 DataCopyPad
   （补 0）；softmax 内部用 original shape 掩码，pad 列不参与 max/sum
3. stats 的 8 元素/行 padding：GM 4× 空间换 block 对齐
4. `loopIdxNew` 跨头连续计数：事件保护（⑬⑭ 的 MTE3 排空）跨头边界持续有效
5. mask 的压缩模式（causal 右下/左上、band、prefix）只影响 ⑦ 的 GM 偏移计算，
   施加机制（⑨ SelectWithBytesMask）统一

---

## 附：一图总结

```
一个 boIdx（batch）的 Vec1 任务总空间：[N2×G, S1, S2]

  头 h ──►  ┌─────────────────────┐
            │   S^h = [S1 × S2]   │  fp32, GM workspace（整头连续）
            │                     │
            │  ┌───┐ ┌───┐ ┌───┐  │   S1 切成 s1OuterSize 个行块
            │  │t0 │ │t1 │ │t2'│  │   （块高 s1BaseSize，尾块 t2' 用 tail 值）
            │  └───┘ └───┘ └───┘  │
            └─────────────────────┘
              块宽恒 = S2 整块（不切——单遍 softmax 的前提）

  每块 ti：[s1BaseSize × S2] ≤ 8K 元素（32KB UB 预算）
    GM→UB(Pong) → scale/mask(Pong→Ping) → softmax 单遍(Ping)
    → P(fp16) 写 GM stage1Res；rowmax/rowsum 攒 UB（满 256 行刷 GM）
```

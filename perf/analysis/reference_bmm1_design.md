# BMM1 专项解读：QKᵀ 批 matmul 的形式与 catlass 实现方案

> 对象：[flash_attention_score_bn2gs1s2_b.h](../../ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h) 的 `IterateBmm1()`（源码已加【解读】注释）+ host 侧 `TilingB::SetBmm1TilingInput()`
> 配套：[Vec1](reference_vec1_tiling.md)·[Vec2](reference_vec2_design.md)（同格式系列）
> 日期：2026-08-16

---

## 1. 问题设定：BMM1 算什么

BMM1 是 FA 第①阶段 QKᵀ。一个 boIdx（batch）的任务：**该 batch 全部 N2×G 个 q 头的
S 矩阵一次算完**：

```
对每个 (boIdx)：
  A = Q^h    [S1 × D]      （h = 0..N2×G-1，每个 q 头一个）
  B = K^{kv} [S2 × D]ᵀ     （kv = h/G，GQA 时 G 个头共享同一 K）
  C = S^h    [S1 × S2]     （= Q^h · (K^{kv})ᵀ · scale 前的原始分数）
输出：mm1Res ping/pong 区（GM workspace，fp32=T）
```

**循环流程速记**（与 Vec 系列的切分对照）：

- **三个维度的切分**：
  - S2（C 的列）/ D（K 维）：**不切**——S2 是 SplitB 前提，D 一次算完
  - S1（C 的行）：**cube 内部按 FixSplit 切**，`s1BasicBlock`（与 Vec1 的行块同值，
    `align16(8192/S2) ≤256`）
  - N2×G：**batch 维**，一次 `IterateBatch` 消化（这是与 catlass 逐头的本质差异）
- **没有循环**——`IterateBmm1` 是**单次发射**（异步）：`SetTensorA/B + IterateBatch`，
  完成点由下一迭代的 `WaitBmm1Result` 等待。

## 2. 调用形态（照抄 [IterateBmm1, bn2gs1s2_b.h:669](../../../ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h#L669)）

```cpp
int64_t qCoreOffset  = ComputeQCoreOffset(extraInfo);   // 本 batch 的 Q 基址（布局相关）
int64_t kvCoreOffset = ComputeKVCoreOffset(extraInfo);  // 本 batch 的 K 基址
bmm1.SetTensorA(queryGm[qCoreOffset]);                  // A 基址
bmm1.SetTensorB(keyGm[kvCoreOffset], true);             // B 基址 + 转置标记（K 按 S2×D 存）
bmm1.IterateBatch<false, true>(mm1ResPing/Pong,          // C 输出（taskId 奇偶 ping/pong）
                               tensorABatchSize,         // = N2×G：A 侧 batch 数
                               tensorBBatchSize,         // = N2：B 侧 batch 数（GQA 广播）
                               false);
```

**batch 维怎么表达**（host 侧 [SetBmm1TilingInput](../../../ops-transformer/attention/flash_attention_score/op_host/arch22/flash_attention_score_tiling_general.cpp) 的 Layout 5 元组）：

```cpp
bmm1.SetShape(S1, S2, D);                    // 单个 batch 矩阵的 M×N×K
bmm1.SetALayout(b, s1, n2, g, d);            // A 的 5 元组：[B, S1, N2, G, D]——A 侧步进 N2×G 次
bmm1.SetBLayout(b, s2, n2, 1, d);            // B 的 5 元组：[B, S2, N2, 1, D]——G 维=1（广播）
bmm1.SetCLayout(b, s1, n2, g, s2);           // C 的 5 元组：G 维展开
bmm1.SetBatchNum(batchNum);                  // = bBaseSize × N2 × G
bmm1.SetFixSplit(s1BasicBlock, s2BasicBlock);// cube 单次计算基本块
bmm1.SetBufferSpace(L1_SIZE, L0C_SIZE);      // L1/L0C 容量约束
```

**关键理解**：B 的 Layout 5 元组 G 维=1、A 的 G 维展开——GQA 的"G 个 q 头共享同一 K"
由 matmul API 的 **batch 广播**天然表达，不需要任何特殊代码。这与我们的
`kvNIdx = h/G` 寻址是同一语义的两种实现。

## 3. `BATCH_LESS_THAN_L1` 语义（性能核心）

MatmulConfig 的 `BatchMode::BATCH_LESS_THAN_L1`（[matmul_config.h:60](file:///usr/local/Ascend/cann-9.0.0/aarch64-linux/ascendc/include/highlevel_api/lib/matmul/matmul_config.h)）：
整个 batch 的 A+B 分片**驻留 L1**，`IterateBatch` 内部只做 L1→L0A/L0B 搬运与 mmad，
**不回 GM**。这正是触发闸门第二条的由来：

```
N2×G × S1 × S2 × dtypeBytes ≤ 128KB   ← 全部头的 A 与 B 能否整体驻留 L1
```

（注：128KB 是 64K×2B 的 host 估算；L1 实际 ~512KB 但要双缓冲 S1/S2 两块 + K 转置开销。）

## 4. catlass 实现方案分析（回答"catlass 是否支持 batch matmul"）

**结论：支持 batch matmul 的骨架，但"L1 驻留 batch"语义无实现——用 catlass 复刻 BMM1
有三条路，按难度递进：**

| 路径 | 现状 | 评估 |
|---|---|---|
| **① 逐头调用 `BlockMmadQK`**（我们 v3 现状） | ✅ 已实现可用 | 语义正确：每头一次 `loadQGM + operator()`，`kvNIdx=h/G` 寻址。代价：无跨头 L1 驻留——GQA 时 K 每组重载 G 次；每头独立完成事件序列 |
| **② `BatchedMatmul` kernel**（[batched_matmul.hpp:128](../../../csrc/catlass/include/catlass/gemm/kernel/batched_matmul.hpp#L128)） | ✅ 完整实现 | batch 展平进 `coreLoops = batchCount × M_tiles × N_tiles`（[batched_matmul.hpp:131](../../../csrc/catlass/include/catlass/gemm/kernel/batched_matmul.hpp#L131)），`batchIdx = taskIdx/(M×N)` 反推，一次 launch 处理整个 batch。**但**：每次迭代重新 `batchOffset = batchIdx × strideA` 传新 tensor 给 `blockMmad`（[batched_matmul.hpp:150-163](../../../csrc/catlass/include/catlass/gemm/kernel/batched_matmul.hpp#L150-L163)）→ **每 batch 重新 GM→L1**（[P1 结论](../our_fa_extension_points.md)）——与路径①等价，仅省了手写循环 |
| **③ `BatchedMatmulTla` + `MmadMultiBatch`**（对标 `BATCH_LESS_THAN_L1`） | ⚠️ **只有 kernel 层脚手架** | `maxL1Batch = L1_SIZE/STAGES/(L1A+L1B)` 计算、batch 组迭代、prefetch 全部就绪（[batched_matmul_tla.hpp:329-387](../../../csrc/catlass/include/catlass/gemm/kernel/batched_matmul_tla.hpp#L329)），但配套的 `BlockMmad<MmadMultiBatch>` **偏特化不存在**（触发 DEPENDENT_FALSE）→ **需自研**，正是 D6 方案 B 的风险点 |

**结论与建议**：v3 当前用路径①（正确性优先）；后续性能对齐时，若需追 `BATCH_LESS_THAN_L1`
的 GQA 收益，路径③是唯一出路，属"完成的代码上再改"的既定后话（D6）。

## 5. 与我方 v3 的映射对照

| 参考 BMM1 | 我方 v3（mha_fwd_splitb.cpp CUBE 段） | 说明 |
|---|---|---|
| `SetTensorA/B + IterateBatch`（异步单次） | `loadQGM + blockMmadQK(...)` 每头 | 路径① |
| batch 广播表达 GQA | `kvNIdx = h/G` 寻址 | 语义等价 |
| mm1Res ping/pong（taskId 奇偶） | workspace slot = 任务号 %2 | 同思想（槽粒度=任务） |
| host `SetFixSplit(s1Base, S2)` | L1Tile `GemmShape<128,128,128>` + actualShape | catlass 由 L1Tile 静态约束 + 运行 shape 控制 |
| `SetBufferSpace(L1, L0C)` | `blockMmadQK.init(resource, nDyn, kDyn, ...)` | 对应物（devlog #17 的 init 教训） |
| 完成点 `WaitBmm1Result`（迭代头） | 我们串行模型下 QK 调用返回即完成（flag 继之） | 串行 vs 3 槽流水的差异 |

## 6. 细节清单

1. `SetTensorB(..., true)` 的转置标记：K 按 `[S2 × D]` 行主存，B 侧取转置——我们
   catlass 的 `LayoutK = ColumnMajor` 正是对应物
2. `SetFixSplit` 与 Vec1 行块同值：保证 cube 的单次输出块 = vector 的读入块，GM 中转
   不产生错位
3. `SetOrgShape`（host）：传 `(S1, S2, s1Stride, s2Stride)` 描述未对齐原形，API 内部
   处理尾块（我们 v1 用实际 rows/cols 直接传 actualShape）
4. ping/pong 选择在**发射时**决定（不是等待时）——与 3 槽流水的槽位对应

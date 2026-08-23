# BMM2 专项解读：PV 批 matmul 的形式与 catlass 实现方案

> 对象：[flash_attention_score_bn2gs1s2_b.h](../../ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h) 的 `IterateBmm2()`（源码已加【解读】注释）+ host 侧 `TilingB::SetBmm2TilingInput()`
> 配套：[BMM1 专项](reference_bmm1_design.md)·[Vec2 专项](reference_vec2_design.md)
> 日期：2026-08-16

---

## 1. 问题设定：BMM2 算什么

BMM2 是 FA 第③阶段 PV。一个 boIdx（batch）的任务：**该 batch 全部 N2×G 个 q 头的
O' 一次算完**：

```
对每个 (boIdx)：
  A = P^h    [S1 × S2]     （h = 0..N2×G-1；Vec1 写出的 softmax 结果，fp16=INPUT_T）
  B = V^{kv} [S2 × D]      （kv = h/G；GQA 广播同 BMM1）
  C = O'^h   [S1 × D]      （= P^h · V^{kv}，**未归一**，fp32=T）
输出：mm2Res ping/pong 区（GM workspace，fp32）——Vec2 读回做行除归一
```

**循环流程速记**（与 BMM1 同构，两处差异见下）：

- **三个维度的切分**：同 BMM1（S2/D 不切、S1 由 FixSplit 切、N2×G 走 batch 维）
- **无循环**——单次异步发射，`WaitBmm2Result` 在下一迭代头等待

**与 BMM1 的两处关键差异**：

| | BMM1（QKᵀ） | BMM2（PV） |
|---|---|---|
| A 的来源 | Q（GM 直读，cube 自己搬） | **P（VECCALC 源）**——Vec1 向量侧算好写 GM，cube 再读：`SetAType(TPosition::VECCALC, NZ, ...)` |
| A 的精度 | INPUT_T | INPUT_T（P 是 fp16 存 GM 的 softmax 结果） |
| C 的格式 | ND（D 维无关，S2 已 16 对齐） | ND 或 **NZ**（D 非 16 对齐时走 NZ 格式 + NzToNd 转置，见 [Vec2 §5](reference_vec2_design.md)） |

## 2. 调用形态（照抄 [IterateBmm2, bn2gs1s2_b.h](../../../ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h)（源码已加【解读】注释））

```cpp
bmm2.SetTensorA(stage1ResPing/Pong);         // A = P：Vec1 写出的 softmax 结果（taskId 奇偶）
bmm2.SetTensorB(valueGm[kvCoreOffset]);      // B = V（GM 直读）
bmm2.IterateBatch<false, true>(mm2ResPing/Pong, tensorABatchSize, tensorBBatchSize, false);
```

host 侧 [SetBmm2TilingInput](../../../ops-transformer/attention/flash_attention_score/op_host/arch22/flash_attention_score_tiling_general.cpp)：

```cpp
bmm2.SetShape(S1, D, S2);                    // M×N×K = S1×D×S2（K 维是 S2！）
bmm2.SetALayout(b, s1, n2, g, s2);           // A(P) 5 元组：与 BMM1 的 C 布局一致
bmm2.SetBLayout(b, s2, n2, 1, d);            // B(V) 5 元组：与 BMM1 的 B 布局一致
bmm2.SetCLayout(b, s1, n2, g, d);            // C(O') 5 元组：G 维展开
bmm2.SetBatchNum(batchNum);
bmm2.SetFixSplit(s1BasicBlock, min(L0C 预算, D));
bmm2.SetBufferSpace(L1_SIZE, L0C_SIZE);
```

**注意**：FixSplit 的第二个参数（N 维 = D）受 **L0C 容量**约束——
`maxDBasicBlock = align16(L0C_SIZE / (s1BasicBlock × calcTypeBytes))`——O' 的 fp32 累加
在 L0C 里，行数×列数×4B 不能超 L0C。这是 BMM2 独有的约束（BMM1 的 N=S2≤128 天然小）。

## 3. 数据依赖：为什么 BMM2 的 P 要等 Vec1（事件链）

P 是**向量侧产出、cube 侧消费**的跨流水线数据：

```
Vec1 写 P 完成（MTE3 搬出）──SetFlag(MTE3_MTE2)──► BMM2 读 P 前 WaitFlag(MTE3_MTE2)
```

Process 主循环里：`ProcessVec1(bo_{t-1})` 后 `SetFlag`，`IterateBmm2(bo_{t-1})` 前
`WaitFlag`——保证 cube 读到的 P 是完整的。这正是参考代码中 `eventIdMte3ToMte2` 的职责
（我方 v3 的对应物：softmax 后 `SetFlag(softmaxReady)` + `BlockMmadPV` 内部等待，
devlog #15 的握手对称教训即在此链上）。

## 4. catlass 实现方案分析

**与 BMM1 的结论相同**（详见 [BMM1 §4](reference_bmm1_design.md)）：三条路——

| 路径 | 状态 | 说明 |
|---|---|---|
| **① 逐头 `BlockMmadPV`**（我们 v3 现状） | ✅ 可用 | 每头一次调用，P/V 由 flag 保证就绪（softmaxReady 传入 operator 内部等待） |
| **② `BatchedMatmul` kernel** | ✅ 实现 | 无 L1 驻留，与①等价 |
| **③ `MmadMultiBatch`**（L1 驻留） | ⚠️ 脚手架 | 自研 BlockMmad 偏特化，D6 后话 |

**BMM2 的额外注意**：P 作为 A 输入是"向量刚写完的 GM 数据"——catlass 路径①下，
`BlockMmadPV` 的 A 装载（copyGmToL1A）与 Vec1 的 P 写出之间靠 **CrossCoreFlag 而非
catlass 内部事件**衔接（catlass 无 KFC 的 MTE3_MTE2 事件），这也是我们 v3 用 flag 传入
operator 的原因（FAInfer 同款）。

## 5. 与我方 v3 的映射对照

| 参考 BMM2 | 我方 v3（CUBE 段） | 说明 |
|---|---|---|
| `SetTensorA(stage1Res)`（VECCALC/NZ 源） | `gP[sOff]`（workspace fp16 视图，softmax 原地覆写） | P 的 GM 位置表达 |
| batch 广播 GQA | `kvNIdx = h/G` | 同 |
| mm2Res ping/pong | slot = 任务号 %2 | 同 |
| 完成点 `WaitBmm2Result`（迭代头） | 串行模型：PV 返回即完成 → SetFlag(pvReady) | 串行 vs 流水 |
| NZ 分支（D 非 16 对齐） | 不移植（v1 D∈{64,128} 恒对齐） | 简化项 |
| FixSplit N 维受 L0C 约束 | catlass L0TileShape N=128 静态约束（D≤128 恒满足） | 触发条件内恒成立 |

## 6. 细节清单

1. P 的精度是 INPUT_T（fp16/bf16）——**唯一一个以低精度参与 matmul 的 A 输入**，
   这是"未归一 P 最大元素=1"设计（Vec1 §6.5）的直接受益者
2. `SetShape(S1, D, S2)` 的 K 维是 S2：P 的列与 V 的行对消，O' 的 K 维 = S2
3. D 非 16 对齐时 C 用 NZ 格式 + Vec2 的 NzToNd 转置（v1 不移植）
4. FixSplit 的 D 上限 = L0C 预算（O' 的 fp32 累加驻留 L0C）
5. ping/pong 选择在发射时决定，与 3 槽流水槽位对应
6. BMM2 的 C 布局 5 元组 = BMM1 的 A 布局（P 和 Q 同排布）——host 侧两个 Set*Layout
   的对称性，迁移时可相互校验

---

## 附：BMM1/BMM2 对称速查

```
              BMM1（QKᵀ）              BMM2（PV）
A 矩阵:    Q [S1×D]  GM 直读      P [S1×S2]  VECCALC 源（Vec1 产出）
B 矩阵:    K [S2×D]ᵀ（转置标记）   V [S2×D]   GM 直读
C 矩阵:    S [S1×S2] fp32         O'[S1×D]  fp32（未归一）
K 维:      D                       S2
FixSplit:  (s1Base, S2整块)        (s1Base, min(L0C预算, D))
batch:     A=N2×G / B=N2（GQA 广播）——两者完全一致
事件链:    WaitBmm1 ←┐             WaitFlag(MTE3_MTE2) ← Vec1 SetFlag
（流水）   完成点=下迭代头  └─ 同款
```

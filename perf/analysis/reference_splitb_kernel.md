# SplitB Kernel 深度解读

> 文件：`ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h`（1615 行）
> 类名：`FlashAttentionScoreBn2gs1s2B`，配套 tiling 类 `FlashAttentionScoreTilingB`
> 本文按"移植所需理解"组织，行号引用以仓库当前版本为准。

---

## 1. 名字与定位

`BN2GS1S2` = 计算布局 5D `[B, N2, G, S1, S2]`：

- **B** = batch，**N2** = kv 头数，**G** = group（q头/kv头），**S1** = q 序列，**S2** = kv 序列
- **`_b` 后缀 = SplitB**：核间只切 B 轴

选型依据（官方设计文档原文）：

> "当S1、S2、D都比较小的时候，CV的基本块较小，我们将B.i, N2, G也纳入到CV基本块中，
> 一次CV交互的数据量更大，提升执行性能。"

## 2. 模板参数（编译期）

```cpp
template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop,
          typename INPUT_T, typename T = INPUT_T, bool isBasicBlock = false,
          LayoutMode layout = LayoutMode::BNGS1S2,
          STemplateType s1TemplateType, STemplateType s2TemplateType, DTemplateType dTemplateType>
```

- `implMode`：精度模式（AA_HIGH_PRECISION / AA_INVALID_LINE_HIGH_PRECISION）
- `layOutType`：输入布局 LAYOUT_BSH（BSND）/ LAYOUT_SBH / LAYOUT_BNSD，kernel 内 `if constexpr` 分别计算 GM offset
- `hasPse / hasAtten / hasDrop`：ALiBi 等位置编码 / attention mask / dropout 的编译期开关
- `INPUT_T`（输入 dtype）与 `T`（计算 dtype，可为 float）可不同
- `s1/s2/dTemplateType`：matmul 静态模板尺寸（对齐到 16/32/48/64/80/96 等档位）

## 3. 数据流全景

每个核的 `Process()`（[544-605 行](../../../ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h#L544-L605)）是**三级软件流水**，循环变量是**子 batch 块**（核间分配的 `splitFactorSize` 份 B 块，核内再展开成 `biN2GoIdx` 循环）：

```
            ┌── BMM1 (IterateBmm1) ──┐
            │   Q[bi×N2×G, S1, D]    │
            │   × K[bi×N2, S2, D]ᵀ   │  batch=B.i×N2×G（A侧）/ B.i×N2（B侧广播）
            │   → S[bi×N2×G, S1, S2] │  写 GM workspace（ping/pong 交替）
            └───────────┬────────────┘
                        ▼
            ┌── Vec1 (ProcessVec1) ──┐
            │   从 GM 读回 S 块        │
            │   scale + PSE(ALiBi)    │
            │   + atten mask          │
            │   SoftMaxFlashV2（单次）│  无跨 tile 刷新！
            │   cast 回 INPUT_T       │
            │   → P 写 GM workspace   │  作为 BMM2 的 A（VECCALC 源）
            └───────────┬────────────┘
                        ▼
            ┌── BMM2 (IterateBmm2) ──┐
            │   P[bi×N2×G, S1, S2]    │
            │   × V[bi×N2, S2, D]     │  batch=B.i×N2×G
            │   → O[bi×N2×G, S1, D]   │  写 GM workspace
            └───────────┬────────────┘
                        ▼
            ┌── Vec2 (ProcessVec2) ──┐
            │   读回 O、除以 softmax   │
            │   sum（Bmm2ResultDiv）  │
            │   cast + 按布局搬出 GM   │
            │   （Bmm2DataCopyOut）    │
            └────────────────────────┘
```

三级流水通过 `extraInfo[3]` 环形数组 + `WaitBmm1Result / WaitBmm2Result / event` 解耦三级之间
的依赖（[552-604 行](../../../ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h#L552-L604)）。

## 4. 关键机制逐一解读

### 4.1 Batch Matmul：B.i×N2×G 进 matmul batch 维

两个 BMM 都用 `matmul::Matmul`（AscendC 高阶 matmul API），MatmulConfig 关键位
（[60-87 行](../../../ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h#L60-L87)）：

- `BatchMode::BATCH_LESS_THAN_L1`：batch 数据留在 L1 内迭代，避免反复 GM 搬运
- baseM/baseN/baseK = s1/s2/dTemplateType 的静态模板值（编译期特化，生成时对齐尺寸）
- 静态 tiling：`matmul::GetMatmulApiTiling<...>(mm1Config)` 编译期算好，避免运行时开销

调用方式（[IterateBmm1, 633-647 行](../../../ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h#L633-L647)）：

```cpp
this->bmm1.SetTensorA(this->queryGm[qCoreOffset]);
this->bmm1.SetTensorB(this->keyGm[kvCoreOffset], true);   // true = 转置
this->bmm1.template IterateBatch<false, true>(this->mm1ResPing,
                                              this->tensorABatchSize,   // = bBaseSize × N2 × G
                                              this->tensorBBatchSize,   // = bBaseSize × N2（B 侧广播）
                                              false);
```

- `tensorABatchSize = bBaseSize × N2 × G`（Q 行 batch 数）、`tensorBBatchSize = bBaseSize × N2`
  （K/V batch 数，G 组共享）——GQA 的共享 K/V 由 matmul batch 广播天然表达
- 输出 ping/pong 交替写 GM workspace（`mm1ResPing/mm1ResPong`），**BMM 结果不驻留 UB**，
  经 GM 中转给 Vector 侧（Cube↔Vector 交互的标准模式）

### 4.2 S2 不切分 → 无 FlashSoftmax 刷新

- `s2BasicBlock = alignedS2`（tiling 侧保证），**S 矩阵一次 BMM 算完**
- softmax 用 `SoftmaxFlashV2<T>`（AscendC 库 intrinsic，[1537-1557 行](../../../ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h#L1537-L1557)），
  单次完成 max/sum/exp，**没有 dm rescale 跨 tile 更新流程**
- 对照：我方 [online_softmax.hpp](../../../csrc/ascend910/flash_attn_npu/online_softmax.hpp) 是跨
  KV-tile 的在线 softmax（每次 KV 块都要 max 比较 + 旧结果 rescale + 新块 exp）——小 S 时这笔
  开销是纯浪费

### 4.3 Vector 侧切分 S1（UB 预算驱动）

- `s1BasicBlock = min(8KB / s2BasicBlock / 16 × 16, alignedS1, 256)`（tiling 侧公式，见 tiling 文档）
- Vec1 的 `biN2GoIdx × s1OuterSize` 双重循环（[795-1008 行](../../../ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h#L795-L1008)）：
  对每个 (子batch组, S1 块) 做 mask/scale/softmax/cast
- softmax 的 sum/max 输出写 GM（`softmaxSumGm/softmaxMaxGm`），供 Vec2 做除法归一
- 若 `T != INPUT_T`（高精度计算），softmax 结果 cast 回 INPUT_T 再写 workspace（省 GM 带宽）

### 4.4 GM Workspace 布局（每核私有段）

`InitInput`（[359-414 行](../../../ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h#L359-L414)）按核索引切 workspace：

```
workspace = [dropout 区] + blockIdx × (mm1×2 + mm2×2 + pseAlibi)
mm1ResPing/Pong  : bBaseSize×N2×G×S1×alignedS2×sizeof(T) 各自 512 对齐
stage1ResPing/Pong: softmax 结果（INPUT_T），与 mm1Res 区共享/叠加（同 dtype 时复用）
mm2ResPing/Pong  : bBaseSize×N2×G×S1×alignedD×sizeof(T)
pseAlibi         : pseAlibiBaseS1×pseAlibiBaseS2×sizeof(half)（内生 ALiBi 预生成）
```

- ping/pong 是**按任务奇偶（taskId%2）**选择的，不是按数据流阶段
- 总 workspace = 16MB 预留 + (bmm1Bytes+bmm2Bytes)×2×coreNum（tiling 侧公式）

### 4.5 边界与对齐处理

- **S2 尾块**：`GetBmm1Result` 里 16 对齐场景直接 `DataCopy`；非对齐走 `DataCopyPad` 补 0
  （[727-773 行](../../../ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h#L727-L773)）
- **D 非 16 对齐**：BMM2 输出 NZ 格式，`NzToNd`（[1022-1082 行](../../../ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h#L1022-L1082)）用 vcopy 转置回 ND
- **搬出布局**：`Bmm2DataCopyOut`（[1258-1322 行](../../../ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h#L1258-L1322)）按 layOutType 计算 dstStride 跨 (N2,G) 行距；
  stride 超 uint16 上限（65535）时退化为逐行搬
- **atten mask**：支持 no-compress / causal 压缩 / band / prefix 四种压缩模式，
  `ComputeAttenMaskOffset` 按 delta 定位压缩 mask 偏移，`SelectWithBytesMask` 施加

### 4.6 UB 预算（InitBuffer, [523-541 行](../../../ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h#L523-L541)）

| Buffer                   | 大小    | 用途                            |
| ------------------------ | ------- | ------------------------------- |
| maskTBufPing/Pong        | 11K/16K | atten mask / drop mask+PSE 输入 |
| pseTBuf                  | 16K     | PSE 数据                        |
| stage1Ping/Pong          | 32K     | BMM1 结果块（T 类型）           |
| commonTBuf               | 32K     | API 临时缓冲                    |
| softmaxSum/Max Ping/Pong | 8K×4   | softmax 累计值                  |
| softmaxExpBuf            | 32B     | （当前模板未用 exp 区）         |
| vecOut                   | 16K     | cast 输出                       |

约束：`vecS1BaseSize × S2 ≤ 32K`（stage1 缓冲），即 tiling 侧 `s1BasicBlock ≤ 8KB/s2BasicBlock` 的由来
（fp16 时 8K 元素 ×2B = 16KB，double buffer 32KB）。

## 5. 与我方 FA 的关键差异（移植对照表）

| 维度              | CANN SplitB                                                 | 我方 FAInfer v2                           |
| ----------------- | ----------------------------------------------------------- | ----------------------------------------- |
| 核间切分          | 仅 B 轴（每核处理连续 B 子块）                              | (B, N1, S1) 任务粒度（coreInfo 逐核分配） |
| S2 处理           | **不切分**，一次 BMM 算完                             | kvBlockSize 切块 + 循环                   |
| softmax           | SoftmaxFlashV2 单次，无刷新                                 | online softmax 跨 KV-tile dm rescale      |
| matmul 调用       | 高阶`matmul::Matmul` + batch 维循环（BATCH_LESS_THAN_L1） | catlass BlockMmadFagSdp 自定义块          |
| Cube↔Vector 交互 | GM workspace 中转 + ping/pong                               | UB 直接衔接（待 P1 确认细节）             |
| S1 切分           | Vector 侧重切（8KB UB 约束）                                | qBlockSize 静态档位                       |
| ALiBi             | PSE 机制（内生/外生两类）                                   | alibi.hpp ApplyAlibiRows（BNS 布局）      |
| softcap           | 无                                                          | online_softmax.hpp HAS_SOFTCAP            |

## 6. 移植时的注意点（初步）

1. 我方 matmul 原语与 `matmul::Matmul`+`IterateBatch` 的对应关系是 P1 的首要问题；
   catlass 有 `batched_matmul*.hpp` 可考察
2. `BATCH_LESS_THAN_L1` 依赖 L1 容量：触发条件里的 `N2×G×S1×S2×dtype ≤ 128KB` 检查必须照搬
3. 我方 feature（softcap/ALiBi/SWA）在 SplitB 结构下的表达：softmax 单次化后 softcap 更简单；
   ALiBi 需在 Vec1 阶段按 BNS→BN2GS1S2 布局施加（参考 PSE 机制）
4. 流水深度 3 级 + ping/pong 的选择、event 同步序列是性能关键，移植时不要"简化"

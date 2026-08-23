# SplitB Tiling 深度解读

> 文件：`ops-transformer/attention/flash_attention_score/op_host/arch22/flash_attention_score_tiling_general.cpp`
> 类：`FlashAttentionScoreTilingB`（2932 行起），注册优先级 98（fallback）
> 本文回答：**什么 shape 会走 SplitB？tiling 参数怎么算出来？workspace 多大？**

---

## 1. 模板选择机制

### 1.1 注册与优先级（[4638-4645 行](../../../ops-transformer/attention/flash_attention_score/op_host/arch22/flash_attention_score_tiling_general.cpp#L4638-L4645)）

```
DropMask(90) > VarLenScore(94) > S1s2Bn2gs1SameAB(95) > S1s2Bn2gs1(96) > S1Bn2gs1(97) > TilingB(98)
```

框架按优先级逐个尝试 `MatchTemplate()`，**第一个匹配的模板胜出**。TilingB 优先级最低 =
**fallback 模板**（"无法走到上述两个模板的其他 shape"）。

### 1.2 匹配流程（`MatchTemplate`, [1847 行](../../../ops-transformer/attention/flash_attention_score/op_host/arch22/flash_attention_score_tiling_general.cpp#L1847)）

1. `CalcS1S2BasicBlock`：在 UB 预算内搜索 (s1BasicBlock, s2BasicBlock) 组合
2. `CalcDBasicBlock`：dBasicBlock
3. 由计算结果反推 `actualTemplate.splitS1/S2/D`，与 `expectTemplate` 比对（`IsTemplateMatched`）
4. 高优先级模板先匹配成功（如 S2>1024 时 S1s2Bn2gs1 的 expectTemplate.splitS2=1 匹配成功），
   大 B 小 S 时高优先级模板匹配失败（或 IsCapable 拒绝），落到 TilingB

### 1.3 TilingB 的 IsCapable 触发条件（[3144 行](../../../ops-transformer/attention/flash_attention_score/op_host/arch22/flash_attention_score_tiling_general.cpp#L3144)）

```cpp
bool IsCapable() override {
    if (alignedS2 > HIGH_PERF_SUPPORT_S2_BASIC)  notMatched = true;   // alignedS2 > 128 拒绝
    if (n2 * g * alignedS1 * alignedS2 * inputDtypeBytes > blockBSizeLimit_ * DATA_TYPE_FP16)
        notMatched = true;                       // > 64K×2B = 128KB 拒绝
    ...
}
```

**精确触发条件：**
1. **`alignedS2 ≤ 128`**（`HIGH_PERF_SUPPORT_S2_BASIC = 128`，[57 行](../../../ops-transformer/attention/flash_attention_score/op_host/arch22/flash_attention_score_tiling_general.cpp#L57)）
2. **`N2×G×alignedS1×alignedS2×dtype_bytes ≤ 128KB`**（`blockBSizeLimit_ = 64×1024`，×2B）——
   即单个 (N2,G) 的 S 矩阵（S1×S2 个元素）需适配 L1 预算。设计文档原文：
   "如果 Q×K 和 P×V 中任何一个矩阵乘法的输入大于了 L1 的 Size，那么走切B模板就没有收益"

## 2. TilingB 参数推导链

### 2.1 基本块计算（`CalcS1S2BasicBlock`, [2944 行](../../../ops-transformer/attention/flash_attention_score/op_host/arch22/flash_attention_score_tiling_general.cpp#L2944)）

```cpp
s2BasicBlock = alignedS2;                                  // S2 不切分（核心！）
s1BasicBlock = blockBUBSizeLimit_ / s2BasicBlock / FRACTAL_NUM * FRACTAL_NUM;  // 8KB/s2，16 对齐
s1BasicBlock = min(s1BasicBlock, alignedS1);
s1BasicBlock = min(maxS1BaseSize_, s1BasicBlock);          // 上限 256
```

- `blockBUBSizeLimit_ = 8KB`（UB 预算）、`FRACTAL_NUM = 16`、`maxS1BaseSize_ = 256`
- 例：S2=128 → s1BasicBlock = min(8K/128/16×16=64, S1, 256)；S2=64 → s1BasicBlock 可到 128

### 2.2 Vec2 侧 S1 块（同函数内）

```cpp
s1Vec2BasicBlock = blockBUBSizeLimit_/alignedD/16×16×2/inputDtypeBytes（dVec2BasicBlock ≤ 上限时）
s1Vec2BasicBlock = min(s1Vec2BasicBlock, alignedS1)
```

BMM2 结果 O 的块 = s1Vec2BasicBlock × D，同样受 8KB UB 约束。

### 2.3 核间拆分（`SetMultiCoreParams`, [3004 行](../../../ops-transformer/attention/flash_attention_score/op_host/arch22/flash_attention_score_tiling_general.cpp#L3004)）

```cpp
totalSize = bOuterSize;                                  // = B（bBaseSize=1）
tempUsedAivNum = min(totalSize, aivNum);                 // Vector 核数
splitFactorSize = ceil(totalSize / tempUsedAivNum);      // 每核分几个 B 子块
coreNum = ceil(totalSize / splitFactorSize);
```

**核间只切 B 轴**：B 巨大时每个核分到 B/coreNum 个子 batch；B < aivNum 时每个核 1 个子 batch。
（注意：用的是 Vector 核数 aivNum，因为 BMM 结果处理在 Vector 侧。）

### 2.4 BMM tiling 设置（`SetBmm1TilingInput`, [3060 行](../../../ops-transformer/attention/flash_attention_score/op_host/arch22/flash_attention_score_tiling_general.cpp#L3060)）

```cpp
bmm1: A=GM/ND(S1×D) × B=GM/ND(S2×D)ᵀ → C=GM/ND(S1×S2)
      SetALayout(bSize, s1Size, n2Size, gSize, dSize);   // A 的 batch 轴分解
      SetBLayout(bSize, s2Size, n2Size, 1,     dSize);   // B 广播 G
      SetCLayout(bSize, s1Size, n2Size, gSize, s2Size);
      SetBatchNum(batch); SetFixSplit(tmpS1BasicBlock, tmpS2BasicBlock);
bmm2: A=VECCALC/NZ(S1×S2 软max结果) × B=GM/ND(S2×D) → C=GM/ND(S1×D)
      SetFixSplit(tmpS1BasicBlock, min(l0c 预算, dBasicBlock));
```

- BMM2 的 A（P 矩阵）来自 **VECCALC**（Vector 侧算出的 softmax 结果，经 workspace GM），NZ 格式
- `SetFixSplit` 固定 cube 单次计算的基本块，与 UB 侧的 s1/s2BasicBlock 对齐

### 2.5 Workspace 大小（`GetWorkspaceSize`, [3170 行](../../../ops-transformer/attention/flash_attention_score/op_host/arch22/flash_attention_score_tiling_general.cpp#L3170)）

```cpp
bmm1Bytes = 1 × N2×G × S1 × alignedS2 × calcTypeSize（bBaseSize=1）
bmm2Bytes = 1 × N2×G × S1 × alignedD × calcTypeSize
workspace = 16MB 预留 + (bmm1Bytes + bmm2Bytes) × 2(ping-pong) × coreNum
```

## 3. 三个模板的边界（速查）

| 模板 | S2 条件 | 切分方式 |
|---|---|---|
| S1s2Bn2gs1（96） | **S2 > 1024**（s2BasicBlock=1024，带 FlashSoftmax 刷新） | 核间 B/N2/G/S1，核内 S1+S2 |
| S1Bn2gs1（97） | **128 < S2 ≤ 1024**，或 S 矩阵 ≥ L1 | 核间 B/N2/G/S1，核内仅 S1，S2 不切 |
| **TilingB（98）** | **alignedS2 ≤ 128 且 S 矩阵 ≤ 128KB** | 核间仅 B，核内 S1 重切 + batch matmul 循环 |

## 4. 移植到我方时的对照要点（供 P2 设计参考）

1. 我方 host tiling 在 [flash_api.cpp](../../../csrc/ascend910/flash_attn_npu/flash_api.cpp) 手工
   计算任务分布（coreInfo start/end BIdx/N1Idx/S1Idx/S2Idx）——SplitB 的 tiling 与之**完全不同**：
   cores 只按 B 分块，其余由 kernel 内 batch 循环消化。需要新增一套 tiling 结构或新 tiling 分支
2. 我方 dispatch 是编译期 (dtype, IS_TND) 轴 + autogen TU——新增 SplitB 变体需扩展
   `autogen/generate_kernels.py`（P1 确认注入点）
3. 触发条件照搬：`alignedS2 ≤ 128 && N2×G×alignedS1×alignedS2×dtype ≤ 128KB`，可在此基础上
   加我方自己的约束（如 batch 下限：B 太小时通用模板即可，SplitB 无收益）
4. workspace 公式需适配我方 tiling 结构（是否按 core 私有段分配）

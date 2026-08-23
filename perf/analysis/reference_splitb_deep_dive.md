 

# SplitB 参考实现深度解读（动手前必读）

> 目的：在改代码之前，把参考方案（`flash_attention_score_bn2gs1s2_b.h` 及配套）的实现细节
> **完整吃透**并成文，供双方共同掌握。本文是 P0 两篇文档（[kernel 结构](reference_splitb_kernel.md)、
> [tiling 参数](reference_splitb_tiling.md)）的**深化与整合**，补齐了 host→kernel 调用路线、
> 逐步执行轨迹、四阶段计算形态、与我方 `mha_fwd_kvcache` 的计算模式对照。
> 甲方要求：**先照搬，再因地制宜微调**。
> 日期：2026-08-14

---

## 0. 阅读地图

| 你想搞懂                                              | 章节       |
| ----------------------------------------------------- | ---------- |
| 从`npu_fusion_attention` 到 kernel 的完整调用链     | §1        |
| 核间到底怎么切任务（真的只切 B？）                    | §2        |
| 一个核拿到任务后怎么算（含具体数字走查）              | §3、§4   |
| FA 四阶段（QK/softmax/PV/rescaleO）在 SplitB 里的形态 | §5        |
| 基本块的维度和 shape                                  | §3.2、§6 |
| UB 怎么分、workspace 怎么排                           | §7、§8   |
| 3 级软件流水怎么 stagger                              | §9        |
| matmul 高阶 API 怎么用（batch 维怎么进）              | §10       |
| 与我方 mha_fwd_kvcache 的系统差异                     | §11       |
| 哪些照搬、哪些必须因地制宜                            | §12       |

## 1. 端到端调用路线（host → kernel 全链路）

### 1.0 前置澄清：npu_fusion_attention 与本算子的关系（2026-08-14 验证）

**调用链成立：`npu_fusion_attention` → `aclnnFlashAttentionScore*` → 本算子 → SplitB kernel。** 证据：

1. **环 1（本机实证）**：FA2 环境 `torch_npu/lib/libtorch_npu.so` 符号表中直接存在
   `aclnnFlashAttentionScore` / `aclnnFlashAttentionScoreGetWorkspaceSize` /
   `aclnnFlashAttentionScoreV3` 等——torch_npu 的 `npu::npu_fusion_attention` 算子内部
   就是调这套 aclnn 两段式接口
2. **环 2（接口逐字段吻合）**：ops-transformer 的
   [aclnnFlashAttentionScore.md](../../../ops-transformer/attention/flash_attention_score/docs/aclnnFlashAttentionScore.md)
   签名（query/key/value/realShift/dropMask/paddingMask/attenMask/prefix/scale/keepProb/
   preTokens/nextTokens/headNum/inputLayout/innerPrecise/sparseMode...）与
   npu_fusion_attention 参数**逐字段一致**（含 `pre_tockens/next_tockens/sparse_mode/ actual_seq_qlen` 等非常规命名），计算公式亦一致
3. **环 3（仓库内代码路径）**：aclnn 入口 → 本算子 → kernel 按 tiling key 分支到
   `FlashAttentionScoreBn2gs1s2B`

**两个限定**：① ops-transformer 是 CANN 算子库的开源发布，与本机安装的 CANN 版本存在
快照漂移（本机 torch_npu schema 已有 `sink/dropout_mask` 等仓库快照没有的参数）——
模板家族与选型逻辑同源，具体阈值可能略有差异；② npu_fusion_attention 按版本/参数可能
选 V1/V3/V4 变体，但都属 `flash_attention_score` 算子家族。

**对本项目的意义**：移植目标由甲方指定（仓库快照）不依赖此链；此链的作用是解释
bench 中 baseline 为何快（其内部多模板体系自动为小 shape 选 SplitB 类模板），从而证实
"我们与 baseline 的差距 = 通用 tiling vs 多模板体系的差距"。

### 1.1 调用链全景

```
torch_npu.npu_fusion_attention(q,k,v,...)                     # Python（我们 bench 的 baseline 入口）
  └─ aclnnFlashAttentionScore(...)                             # op_api/aclnn_flash_attention_score.cpp
       └─ GE 运行时调 tiling 函数（每次算子调用执行一次）
          ├─ TilingPrepareForFlashAttentionScore               # op_host/flash_attention_score_tiling.cpp:414
          │    一次性读平台参数：aivNum/aicNum/ubSize/l1Size/l0cSize/l2CacheSize → CompileInfo
          └─ TilingForFlashAttentionScore → TilingRegistryArch::DoTilingImpl(context)   # :408
               └─ 按【注册优先级】依次尝试 tiling 模板类：
                  DropMask(90) → VarLen(94) → S1s2SameAB(95) → S1s2Bn2gs1(96)
                  → S1Bn2gs1(97) → 【TilingB(98) ← 大B小S 落到这里】
                  每个类依次过 IsCapable()（shape 闸门）和 MatchTemplate()（UB 内可行性搜索）
               TilingB 命中后产出三样东西：
               ① FlashAttentionScoreGeneralTilingData（含 bmm1/bmm2 的 TCubeTiling + coreParams）
               ② SetTilingKey( GET_TPL_TILING_KEY(...) )      # 把模板轴（dtype/layout/mask开关/
                                                             #   splitB/... 共 20 个字段）打包成 key
               ③ SetBlockDim( CalcTschBlockDim(coreNum, aicNum, aivNum) )
               ④ SetWorkspaceSize( 16MB预留 + (mm1+mm2)×2×coreNum )
       └─ GE 按 tiling key 选出编译期实例化好的 kernel 并 launch（tiling buffer 作为参数传入 GM）
            flash_attention_score<KernelTypeKey, UB0, UB1, Block, ImplMode, DataType, Layout,
                                  Bmm1Format, Bmm2Source, Sparse, BigDoubleBuffer, HasDrop,
                                  HasAtten, HasPse, EnableL1Reuse, HasRope, MMPolicy,
                                  S1T, S2T, dT>(query, key, ..., workspace, tiling)
            # op_kernel/flash_attention_score.cpp:384，__global__ 模板参数与 tiling key 字段一一对应
            └─ if constexpr (UB0==9 && UB1==9 && Block==0)     # :628 "B模板"分支
                 INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B,
                     ImplMode, Layout, hasPse, hasAtten, hasDrop,
                     half|bfloat16_t|float,  /* INPUT_T */
                     float,                 /* T：计算精度（B模板固定 fp32 计算） */
                     true, LayoutMode::BSNGD/SBNGD/BNGS1S2, s1T, s2T, dT)   # :642
                 展开为：
                 ① COPY_TILING_DATA：从 GM tiling buffer 抽出 bmm1TilingData/bmm2TilingData（TCubeTiling）
                 ② REGIST_MATMUL_OBJ(&tPipe, sysWs, op.bmm1, bmm1tiling, op.bmm2, bmm2tiling,
                                     op.bmm2Nz, bmm2tiling)
                    # 向三个 matmul 对象注册运行时上下文（host 算好的 cube tiling + 系统 workspace）
                 ③ op.Init(query,...,softmaxMax, softmaxSum, softmaxOut, attentionOut,
                           userWorkspace, tilingData, &tPipe)
                 ④ op.Process()
```

**要点**：

1. **tiling key 就是 host→kernel 的分发机制**：host 把模板选择结果编码进 key，kernel 的
   `__global__` 模板参数在编译期为所有 key 组合生成实例，运行时零开销落入对应 `if constexpr`
   分支。**没有运行时的模板选择代码**。
2. **matmul 的 tiling 是 host 算好、GM 传入的**：`TCubeTiling`（bmm1/bmm2 各一份）在
   TilingB 里由 `matmul_tiling::MatmulApiTiling` 生成，kernel 侧 `REGIST_MATMUL_OBJ` 注册后
   `matmul::Matmul` 对象直接可用。
3. **B 模板的 `T` 恒为 float**（实例化处 `half, float`）：输入 fp16/bf16，QK 累加、softmax、
   PV 累加全 fp32，只有 GM 中转的 P（softmax 结果）和最终输出 cast 回低精度。
4. kernel 类型 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`：1 cube + 2 vector 混合核。

## 2. 核间切分：以 AIV 为基数、只切 B 轴

### 2.1 切分机制（host：`TilingB::SetMultiCoreParams`，[tiling_general.cpp:3004](../../../ops-transformer/attention/flash_attention_score/op_host/arch22/flash_attention_score_tiling_general.cpp#L3004)）

```
totalSize      = bOuterSize = B            （bBaseSize 恒 = 1，见 §3.1）
usedAivNum     = min(B, aivNum)            aivNum = GetCoreNumAiv() = 2 × aicNum（A2 架构）
splitFactorSize= ceil(B / usedAivNum)      每个"核"分到的 batch 数
coreNum        = ceil(B / splitFactorSize) 实际使用的切片数（≤ aivNum）
blockDim       = CalcTschBlockDim(coreNum, aicNum, aivNum) = ceil(coreNum × aicNum / aivNum)
                                            （把 AIV 单位的切片数换算成 AI Core block 数）
```

### 2.2 kernel 侧的对应（[bn2gs1s2_b.h:546](../../../ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h#L546)）

```cpp
this->blockIdx = GetBlockIdx();                              // 原始值：AIV 全局索引 ∈ [0, coreNum)
int64_t multiCoreInnerOffset = blockIdx * splitFactorSize;   // 本核 batch 区间起点
int64_t multiCoreInnerLimit  = min(offset + splitFactorSize, B);
for (boIdx = offset; boIdx < limit + 2; boIdx++) { ... }     // +2 是流水预热的哨兵迭代（§9）
```

**核间切分的完整答案**：

- **只切 B 轴**，N2/G/S1/S2 完全不跨核（"核间切分B轴"）
- 切片基数是 **AIV（vector 子核）**而非 AI Core：一个 AI Core 的两个 vector 子核
  （GetBlockIdx 为 2k 和 2k+1）**各自独立处理不同的 batch 区间**，各自向本 Core 的 cube
  发 matmul 任务，cube 按到达顺序串行执行。这正是 [FA算子设计介绍.md §6.1](../../../ops-transformer/attention/flash_attention_score/docs/FA算子设计介绍.md)
  "Vector0 和 Vector1 会独立发起 Matmul 的任务，两者没有关联性"的落地。
- **对照我方**：`mha_fwd_kvcache.cpp:235` 用 `GetBlockIdx()/GetSubBlockNum()` 把索引归一化到
  AI Core（aic 单位），两个 vector 子核合作为一个任务单元 → 我方切片数只有参考的 **1/2**，
  且任务粒度是 (B,N1块,S1块) 复合轴。**这是两种根本不同的核间计算模型**。

## 3. 单核执行模型与基本块 shape

### 3.1 两层任务结构：boIdx 外层 × (N2×G) batch matmul 内层

`bBaseSize = 1`（[tiling_general.cpp:2974](../../../ops-transformer/attention/flash_attention_score/op_host/arch22/flash_attention_score_tiling_general.cpp#L2974)：`int64_t bIn = 1`）——
**每个 boIdx = 恰好 1 个 batch 的全部头**：

```
tensorABatchSize = bBaseSize × N2 × G = N2×G     // BMM 的 A 侧 batch 数（= 全部 q 头）
tensorBBatchSize = bBaseSize × N2     = N2       // BMM 的 B 侧 batch 数（kv 头；GQA 时 G 头共享）
```

"B.i, N2, G 作为循环轴" 的准确含义：**B.i（=1/批）是外层循环轴，N2×G 是 matmul 的
batch 维**（一次 `IterateBatch` 消化），N2 单独作为 B 侧 batch 维（GQA 广播）。

### 3.2 基本块维度（host 定死，kernel 只读）

| 块                                          | 公式（TilingB）                                         | 约束来源                                                          |
| ------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------- |
| **s2BasicBlock**                      | `= alignedS2`（**S2 整块，不切**）              | 触发条件保证 ≤128                                                |
| **s1BasicBlock**（Vec1 softmax 行块） | `min(8KB/s2BasicBlock 按16对齐, alignedS1, 256)`      | stage1 UB 缓冲 8K×sizeof(T)=32KB（fp32）能装下`s1×s2` 的 S 块 |
| **s1Vec2BasicBlock**（Vec2 行块）     | `min(8KB/alignedD 按16对齐×2/dtypeBytes, alignedS1)` | O 块（fp32 读）+ 输出（fp16 写）装 UB                             |
| **dBasicBlock**                       | `= alignedD`（D 不切）                                | L0C 预算                                                          |
| **bBaseSize**                         | `= 1`                                                 | 每 boIdx 一整批                                                   |

单次 Cube 计算（L0 基本块）由 matmul API 内部按 `SetFixSplit(s1BasicBlock, s2BasicBlock)`
组织——**FA 层面能看到的"基本块"就是 `s1BasicBlock × s2BasicBlock`（QK 的 S 输出块）和
`s1Vec2BasicBlock × dBasicBlock`（O 块）**，Cube 内部再按 L0Tile 切。

### 3.3 数值走查（worked example，对照我们的 bench 用例）

**输入**：`B=1024, S1=S2=64, H=8 (MHA: N2=8, G=1), D=128, fp16, non-causal`
**触发检查**：alignedS2=64 ≤128 ✓；N2×G×S1×S2×2B = 8×64×64×2 = 64KB ≤ 128KB ✓ → 走 SplitB。

**host 产出**：

```
s2BasicBlock=64, s1BasicBlock=min(8192/64=128→128, 64, 256)=64, s1OuterSize=1
s1Vec2BasicBlock=min(8192/128=64, 64)=64, s1Vec2OuterSize=1
bBaseSize=1, bOuterSize=1024
设 aivNum=80, aicNum=40（910B4 双 die [示意，待上板确认]）
→ splitFactorSize = ceil(1024/80) = 13, coreNum = ceil(1024/13) = 79
→ blockDim = ceil(79/2) = 40 个 AI Core block；79 个 AIV 切片各自处理 13 个 batch
   （末尾切片 1024−78×13=10 个）
```

**某个核（blockIdx=k，batch 区间 [13k, 13k+13)）的完整执行轨迹**：

```
对每个 boIdx（=1 个 batch，含全部 8 个头）:
  ① BMM1  cube: bmm1.SetTensorA(queryGm + boIdx×(8×64×128))     // Q: [8头batch, 64, 128]
                bmm1.SetTensorB(keyGm   + boIdx×(8×64×128), 转置) // K: [8头batch, 64, 128]
                bmm1.IterateBatch(mm1ResPing, tensorABatchSize=8, tensorBBatchSize=8)
                // 8 个 [64×128]×[128×64] 的 batch matmul，一次调用
                // 输出 S: 8 × [64×64] fp32 → GM workspace（本核私有区，ping/pong 按 taskId 奇偶）
  ② Vec1  vector: for biN2GoIdx ∈ [0,8):                        // 逐头
                for loopIdx ∈ [0, s1OuterSize=1):               // S1 块（本例只有 1 块）
                  DataCopyPad S 块[64×64] fp32 从 GM → UB(stage1Pong 32KB)
                  Muls(scale) → [PSE 加偏置] → [attenMask SelectWithBytesMask]
                  SoftmaxFlashV2 单遍：max→exp→sum，原地得 P(fp32)
                  Cast → fp16 → DataCopyPad 写 GM stage1Res（BMM2 的 A 输入）
                  softmaxSum/Max fp32 留 UB，块尾（本例每块都是尾块）写 GM
  ③ BMM2  cube: bmm2.SetTensorA(stage1ResPing /* VECCALC/NZ 格式 */)
                bmm2.SetTensorB(valueGm + boIdx×…)
                bmm2.IterateBatch(mm2ResPing, 8, 8)
                // 8 个 [64×64]×[64×128] → O: 8 × [64×128] fp32 → GM workspace
  ④ Vec2  vector: for biN2GoIdx ∈ [0,8): for s1oIdx ∈ [0,1):
                  DataCopy O 块[64×128] fp32 + softmaxSum[64] 从 GM → UB
                  Bmm2ResultDiv：O[i][j] /= sum[i]（fp32 除法，行广播）
                  Cast fp16 → DataCopyPad 按 BSH 布局步长写 attentionOut GM
```

每核总工作量：13 batch × (1 次 batch-BMM1 + 8 块 softmax + 1 次 batch-BMM2 + 8 块 div)。
**同步点每 batch 每 stage 边界一次**（BMM1↔Vec1 靠 matmul 的 WaitIterateBatch + 事件），
而非我方的每 (b,头,块) 任务 3 个 CrossCoreFlag。

### 3.4 GQA 时（如 H=32, kvH=4 → N2=4, G=8）

- A 侧 batch = N2×G = 32（32 个 q 头的 Q/P 各自成 batch 矩阵）
- B 侧 batch = N2 = 4（只有 4 份 K/V）；matmul API 的 batch 广播让每 8 个 A-batch 共享 1 份
  B-batch——**GQA 共享在 batch 维天然表达，不需要任何特殊代码**
- 触发闸门相应变化：N2×G×S1×S2×dtype = 32×64×64×2 = 256KB > 128KB → **此 shape 不走 SplitB**
  （会落回 S1Bn2gs1 模板）。即 CANN 的 SplitB 容许的头数×序列乘积有限，见 §12 讨论

## 4. Process() 主循环与软件流水

[Process(), bn2gs1s2_b.h:544](../../../ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h#L544)：

```cpp
SplitBExtraInfo extraInfo[3];            // 3 槽环形，3 个 boIdx 在飞
for (multiCoreInnerIdx = offset; idx < limit + 2; idx++) {   // +2 个哨兵迭代排空流水
    if (taskId >= 1 && notLast)      WaitBmm1Result();       // 等 boIdx(t-1) 的 BMM1 完成
    if (notLastTwoLoop)              IterateBmm1(extraInfo[t%3], boIdx(t));   // 为 boIdx(t) 发 BMM1
    if (taskId > 0 && notLast)       ProcessVec1(extraInfo[(t+2)%3]);         // 处理 boIdx(t-1)
    if (taskId > 1)                  WaitBmm2Result();       // 等 boIdx(t-2) 的 BMM2
    if (taskId > 0 && notLast)       IterateBmm2(extraInfo[(t+2)%3]);         // boIdx(t-1) 发 BMM2
    if (taskId > 1)                  ProcessVec2(extraInfo[(t+1)%3]);         // 处理 boIdx(t-2)
    taskId++;
}
```

stagger 关系（时间 →；**2026-08-16 按代码实证修正**：BMM2 滞后 BMM1 **1** 个 batch、
与 Vec1 同 batch——`IterateBmm2` 与 `ProcessVec1` 用同一个 `extraInfo[(taskId+2)%3]` 槽）：

```
迭代 t:        t0          t1              t2              t3
cube:        BMM1(bo0)   BMM1(bo1)       BMM1(bo2)       BMM1(bo3)   ← BMM1 领先 1 个 batch
vec:                     Vec1(bo0)       Vec1(bo1)       Vec1(bo2)   ← 落后 BMM1 1 个
cube:                    BMM2(bo0)  ←wait BMM2(bo1)      BMM2(bo2)   ← 与 Vec1 同 batch（等 P 可见）
vec:                                     Vec2(bo0)       Vec2(bo1)   ← 落后 2 个（BMM2 结果已在迭代头 Wait）
```

每次迭代内顺序（见 Process 注释 ①~⑥）：`WaitBmm1 → 发BMM1(bo_t) → Vec1(bo_{t-1}) → SetFlag(P可见) → WaitBmm2(bo_{t-2} 完成) → WaitFlag(P可见) → 发BMM2(bo_{t-1}) → Vec2(bo_{t-2})`——cube 上 BMM1/BMM2 交替处理相邻 batch，vector 上 Vec1/Vec2 交替。
ping/pong workspace 按 `taskId % 2` 选择，3 槽 extraInfo 保存每个在飞 boIdx 的块参数，
`+2` 哨兵迭代排空流水。

## 5. FA 四阶段在 SplitB 中的形态（重点回答）

| FA 阶段                | 我方 mha_fwd_kvcache                                                                                                                                | SplitB 参考                                                                                                        | 本质差异                                                                      |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| **① QK**        | 每 (b, q头块, s块) 任务：Q[128行打包]×K[512列/tile]，KV 按`MAX_KV_STACK_LEN=512` 外层循环                                                        | 每 batch 一次`IterateBatch`：N2×G 个 `[s1Blk×D]×[D×S2整块]`                                                | batch 进 matmul batch 维；**S2 不切**（≤128 一次算完）；Q 不再跨头打包 |
| **② softmax**   | online softmax：跨 KV-tile 循环，每 tile 做 max 比较 + dm = exp(gm−hm) +**旧 O/P 结果 rescale**（`online_softmax.hpp` 的 gm/dm/gl 状态机） | **SoftmaxFlashV2 单遍**：一个 `[s1Blk × S2整块]` 上 max→sub→exp→sum 一次完成，**无任何跨块状态** | S2 不切 ⇒ softmax 天然单遍 ⇒ 整个"在线更新"机制不存在                       |
| **③ PV**        | P[128×512/tile]×V[512/tile×D]，随 KV-tile 循环，O 在 GM workspace 累加                                                                           | 每 batch 一次`IterateBatch`：N2×G 个 `[s1Blk×S2]×[S2×D]`                                                   | 同①，batch 维 + S2 整块                                                      |
| **④ rescale O** | `rescale_o.hpp`：**每个 KV-tile 迭代都要** rescale + 累加，末 tile 才除以 sum                                                               | **退化为一次性除法** `Bmm2ResultDiv`：O[i][:] /= sum[i]，在 Vec2 里行广播 Div                              | S2 不切 ⇒ O 无需跨块合并 ⇒ "rescale" 阶段消失，只剩最终归一化               |
| mask 施加              | 每 tile 判断 triu 窗口/SWA 窗口与 tile 的相交（`mha_fwd_kvcache.cpp:683-818`）                                                                    | Vec1 内一次：按压缩 mask 偏移`ComputeAttenMaskOffset` 取 `[s1Blk×s2Blk]` mask，`SelectWithBytesMask` 施加   | 每 (头,S1块) 一次而非每 tile                                                  |
| scale/softcap          | softmax epilogue 内                                                                                                                                 | Vec1：Muls(scale)→[softcap]→mask→softmax                                                                        | 位置等价                                                                      |
| LSE                    | rescale_o 末尾写                                                                                                                                    | softmaxSum/softmaxMax 独立 GM 输出（host 侧合成）                                                                  | 输出形态差异                                                                  |

**一句话**：SplitB 之所以快，是因为**把"KV-tile 循环 + 在线 softmax + rescale"这套为长序列
设计的机制整体移除**（S2≤128 不需要），同时把任务粒度从 (b,头,S1块) 提升到 (batch × 全部头)，
让 cube 的 batched matmul 和 vector 的分块处理各自满载。

## 6. Tiling 参数体系（TilingB 完整推导链）

（详见 [reference_splitb_tiling.md](reference_splitb_tiling.md)，此处补执行序）

```
MatchTemplate():  CalcS1S2BasicBlock → CalcDBasicBlock → IsTemplateMatched(B恒true) → CalcUBSize
IsCapable():      alignedS2 ≤ 128  &&  N2×G×alignedS1×alignedS2×dtypeBytes ≤ 128KB
SetCoreParams():  bBaseSize=1, bOuterSize=B, s1/s2/s1Vec2 块与 outer 数
SetBmm1TilingInput(): matmul_tiling::MatmulApiTiling
    A: GM/ND [S1×D],  BLayout(b, s2, n2, 1, d)   ← B 侧 G 维=1（GQA 广播）
    B: GM/ND [S2×D]ᵀ, BLayout(b, s2, n2, 1, d)
    C: GM/ND [S1×S2], CLayout(b, s1, n2, g, s2)  ← C 侧展开 G
    SetBatchNum(batch); SetFixSplit(s1Blk, s2Blk); SetBufferSpace(L1, L0C)
SetBmm2TilingInput():
    A: VECCALC/NZ [S1×S2]（softmax 结果，从 GM workspace 读）
    B: GM/ND [S2×D]
    C: GM/ND [S1×D]（D 非16对齐时 C 用 NZ + NzToNd 转置回 ND）
SetSoftMaxTiling(): SoftMaxFlashV2TilingFunc([s1Blk, s2Blk], ...)
SetMultiCoreParams(): §2.1 的核间切分
GetWorkspaceSize(): §8 公式
```

**SetALayout/SetBLayout/SetCLayout 是 batch 维的声明方式**：5 元组 (b, s?, n2, g, d) 描述
每个 batch 矩阵的轴构成，matmul API 由此推导 batch 步进（A 步进 N2×G 次、B 步进 N2 次）
和 GQA 广播。**我方移植时需要等价物**（catlass 无此 API，见 §12）。

## 7. UB 空间分配（[InitBuffer, bn2gs1s2_b.h:523](../../../ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h#L523)）

| Buffer                  | 大小                                | dtype            | 用途                                                       |
| ----------------------- | ----------------------------------- | ---------------- | ---------------------------------------------------------- |
| maskTBufPing            | 11 KB                               | u8               | atten mask 输入块                                          |
| maskTBufPong            | 16 KB                               | u8/half          | dropout mask / PSE 输入                                    |
| pseTBuf                 | 16 KB                               | INPUT_T          | PSE（ALiBi 等）数据；Vec2 时复用作 softmaxSum 读入         |
| **stage1PingBuf** | 8K×4B=32 KB                        | **T=fp32** | S 块读入 + softmax 原地计算（[s1Blk×s2Blk] 上限 8K 元素） |
| **stage1PongBuf** | 32 KB                               | fp32             | ping/pong 交替（流水用）                                   |
| commonTBuf              | 64×128×4B=32 KB                   | u8/fp32          | SoftmaxFlashV2 / SelectWithBytesMask 的 API 临时区         |
| softmaxSumPing/Pong     | 256×32B=8 KB ×2                   | fp32             | sum 累计（256 行 × fp32 pad 8）                           |
| softmaxMaxPing/Pong     | 8 KB ×2                            | fp32             | max 累计                                                   |
| softmaxExpBuf           | 32 B                                | —               | （本模板 exp 输出原地，占位）                              |
| vecOut                  | 16 KB                               | INPUT_T          | Vec2 的 cast 输出                                          |
| **合计**          | **≈187 KB ≈ 192KB UB 打满** |                  |                                                            |

注意 pseTBuf/vecOut 的**复用**（Vec2 时 pseTBuf 兼任 softmaxSum 读入、vecOut 兼任输出 cast）——
UB 紧张时的设计手法，我方移植时同样需要。

## 8. GM Workspace 布局（[InitInput, bn2gs1s2_b.h:359-414](../../../ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h#L359)）

**每核私有段**（核间无共享、无原子）：

```
workspace = [dropmask 区(512对齐)] + Σ_{coreIdx} [ coreIdx × (mm1×2 + mm2×2 + pseAlibi) ]

mm1区/份  = ceil(bBase×N2×G×S1×s2Base×sizeof(T)×(dtype/2), 512)   // S 矩阵 fp32，ping+pong 两份
mm2区/份  = ceil(bBase×N2×G×S1×dBase×sizeof(T), 512)              // O 矩阵 fp32，两份
stage1Res = 紧跟 mm1Res 之后再隔 1 份 mm1Offset（fp32 时独立 2 份；T==INPUT_T 时与 mm1Res 叠加复用）
pseAlibi  = pseAlibiBaseS1×pseAlibiBaseS2×half（内生 ALiBi 预生成区）

host 总量 = 16MB 预留 + (mm1+mm2)×2×coreNum        （coreNum 以 AIV 计，§2）
```

ping/pong 选择一律 `taskId % 2`（boIdx 奇偶）——与流水 stagger 配合保证读写不冲突。

## 9. matmul 高阶 API 用法（batch 维的关键）

```cpp
// 类型：MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, /*transpose*/false, layout>
//       layout 参数即 §6 的 5 元组 LayoutMode（BNGS1S2/BSNGD/SBNGD）
constexpr static MatmulConfig mm1Config = GetBn2gs1s2BMm1Config<...>();
//   关键位：baseM=s1Template, baseN=s2Template, baseK=dTemplate（静态 shape 时编译期特化）
//           BatchMode::BATCH_LESS_THAN_L1 ← batch 数据整体驻留 L1 的语义开关
constexpr static MatmulApiStaticTiling stcMm1Tiling = GetMatmulApiTiling<...>(mm1Config);
modeTypeMm1 = s1Template==UNK ? Matmul<...>(动态tiling) : Matmul<..., stcMm1Tiling>(静态);

// 调用（IterateBmm1, :633）：
bmm1.SetTensorA(queryGm[qCoreOffset]);           // A 基址 = 本 batch 的 Q 首
bmm1.SetTensorB(keyGm[kvCoreOffset], true);      // B 基址 + 转置标记
bmm1.IterateBatch<false, true>(mm1ResPing,       // C 输出到 GM workspace（ping）
                               tensorABatchSize,  // = N2×G：A/batch 步进次数
                               tensorBBatchSize,  // = N2：B/batch 步进次数（A/B 异步 → GQA 广播）
                               false);
...
bmm1.WaitIterateBatch();  bmm1.End();            // 流水同步点（Process 内）
```

**`BATCH_LESS_THAN_L1` 的语义**（配合触发闸门 `N2×G×S1×S2×dtype ≤128KB`）：整个 batch 的
A、B 分片常驻 L1，IterateBatch 内部只做 L1→L0A/L0B 搬运与 mmad，不回 GM。**这正是 catlass
没有的机制**（[P1 §4.1](our_fa_extension_points.md)：catlass 每 batch 重新 CopyGMToL1，
MmadMultiBatch 只有脚手架）——是移植时最大的"需要自己实现"的部分，也解释了为什么闸门
按 N2×G×S1×S2（而不是 D）算：它约束的是"一个 batch 全部头的 A+B 能否驻留 L1"。

## 10. 特性施加位置速查

| 特性                               | 施加点                                                                      | 机制                                                                      |
| ---------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| scale                              | Vec1`Muls`                                                                | 标量乘                                                                    |
| PSE（ALiBi/位置编码）              | Vec1`PseCompute`（内生 ALiBi 在 Init 时预生成到 workspace pseAlibi 区）   | 编译期 hasPse                                                             |
| attenMask（causal/band/prefix 等） | Vec1`SelectWithBytesMask`，偏移由 `ComputeAttenMaskOffset` 按压缩模式算 | 编译期 hasAtten                                                           |
| dropout                            | Vec1 softmax 后`ComputeDropMask`                                          | 编译期 hasDrop                                                            |
| softcap                            | **参考实现没有**（CANN FA 无此特性）                                  | 我方需在 Vec1 的 Muls 之后 softmax 之前插 tanh——照我方 HAS_SOFTCAP 模式 |
| LSE                                | softmaxSum/softmaxMax 独立输出                                              | host 侧合成 log(sum)+max                                                  |

## 11. 与我方 mha_fwd_kvcache 的系统性对照

| 维度             | 我方 FAInfer v2                                                                                                   | 参考 SplitB                                                                                                 |
| ---------------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 核间切分         | (B,N1块,S1块) 复合任务流，**aic 单位**切片（`GetBlockIdx()/GetSubBlockNum()`），`coreInfo[]` 显式区间表 | **仅 B 轴**，**aiv 单位**切片（原始 `GetBlockIdx()`），`blockIdx×splitFactorSize` 隐式区间 |
| 2 个 vector 子核 | 合并为一个任务单元（同任务）                                                                                      | **各自独立处理不同 batch 区间**，独立发 matmul                                                        |
| 任务粒度         | 1 任务 = 1 个 (b, q头块, s块)                                                                                     | 1 任务 = 1 个 batch 的全部 N2×G 头                                                                         |
| Q 布局           | qS×qN 打包成 128 行（`rowNum=qSBlockSize×qNBlockSize`）                                                       | 每 batch 矩阵 = 单头的 [S1×D]，batch 维承载头                                                              |
| KV 循环          | `MAX_KV_STACK_LEN=512` 外层 + `PRE_LAUNCH=2` 三槽预热                                                         | **无**（S2 整块）                                                                                     |
| softmax          | online，gm/dm/gl 状态机跨 tile                                                                                    | SoftmaxFlashV2 单遍，无状态                                                                                 |
| rescale          | 每 tile rescale + 末 tile 归一                                                                                    | 无 rescale，Vec2 一次 Div                                                                                   |
| 同步             | 每 tile 3 个 CrossCoreFlag（qkReady/softmaxReady/pvReady）                                                        | 每 batch 每 stage 一次（matmul Wait + MTE 事件）                                                            |
| Cube↔Vector     | GM workspace 中转（S/P 同址覆写，O 分离）                                                                         | GM workspace 中转（S、P、O 分区 + ping/pong）——模式相同，布局不同                                         |
| 流水             | tile 级 3 槽（PRE_LAUNCH=2）                                                                                      | **boIdx 级 3 槽**（§4 stagger）                                                                      |
| matmul 栈        | catlass BlockMmad（自管 L1/L0）                                                                                   | AscendC`matmul::Matmul` + `BATCH_LESS_THAN_L1` + host TCubeTiling                                       |
| mask             | kernel 内构造 triu（COMP_TRIU_MASK）                                                                              | GM 输入 attenMask + 压缩模式偏移计算                                                                        |
| 触发             | 无条件走通用路径                                                                                                  | host tiling 模板注册表按优先级选（B 模板 priority 98）                                                      |

## 12. 照搬 vs 因地制宜（给 P2 设计的修订输入）

### 可以直接照搬的

1. **任务模型**：仅切 B、bBaseSize=1、A-batch=N2×G、B-batch=N2、ping/pong 按 boIdx 奇偶
2. **四段结构与 3 槽 boIdx 流水**（§4 stagger 时序）
3. **tiling 参数公式**（s1/s2/s1Vec2 块、splitFactorSize、workspace 公式）
4. **Vec1/Vec2 的分块逻辑与 UB 复用手法**
5. **mask 施加位置**（softmax 前、SelectWithBytesMask）

### 必须因地制宜的（差异根源：我们用 catlass + torch 扩展，不是 CANN 算子框架）

1. **调用路线**：我们没有 tiling registry / tiling key / REGIST_MATMUL_OBJ——照我方
   `flash_api.cpp` 手工 tiling + `FwdLaunchArgs` + autogen TU 的模式（P2 设计已有）
2. **matmul 栈**：`matmul::Matmul`+`BATCH_LESS_THAN_L1` 不可用 → catlass `BlockMmadQK/PV`
   逐头调用起步（阶段1），L1 驻留 batch 需自研（阶段2，`[需 NPU 验证]`）
3. **核间基数**：是否也采用 aiv 单位切片（每 Core 两子核独立任务）？我方现有 kernel 是
   aic 单位。**建议 v1 先保持 aic 单位（与现有 launch/同步机制一致），把"每 Core 双任务"
   列为独立优化项**——它使可用切片翻倍，但需要 workspace/同步全链路适配
4. **softcap**：参考没有，仿我方 HAS_SOFTCAP 在 Vec1 插入
5. **mask**：参考走 GM attenMask 输入；我方现有 kernel 内构造 triu 的方式更省带宽，
   保留我方方式（这属于"微调"）
6. **ALiBi**：参考的 PSE 机制可参考布局思路，但接入方式等我方 alibi 分支合并后定

### 两个决策问题（2026-08-14 用户已拍板）

- **Q-A：触发闸门 → ①严格照搬，不放宽**。`alignedS2 ≤ 128 && N2×G×alignedS1×alignedS2×dtype ≤128KB` 原样保留。理由（用户）：参考方案是深思熟虑后的设定，先照搬；放宽闸门属于
  "进一步优化改进"，是跑通之后的后话。(H=8,S=128) 等超闸门 shape 继续走现有路径。
- **Q-B：小 B → 不回落旧路径**。照搬参考实现（SplitB 是 fallback 模板，小 B 也这样跑）。
  同样理由：不自作主张加优化分支。

**总原则（用户原话精神）**：不要一下子追求太多，否则潜在麻烦导致目标难以实现。
**先完整照搬 → 集成到项目 → 完整适配所有功能特性和场景 → 跑通测试 → 确认性能提升，
之后再去考虑进一步的优化改进。**

---

## 附：参考文件索引（本文引用）

| 文件                                                                                       | 角色                                                                  |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| `ops-transformer/attention/flash_attention_score/op_api/aclnn_flash_attention_score.cpp` | aclnn 入口（npu_fusion_attention 的底层）                             |
| `op_host/flash_attention_score_tiling.cpp`                                               | tiling 注册入口 + 平台参数读取                                        |
| `op_host/arch22/flash_attention_score_tiling_general.cpp`                                | 全部 tiling 模板类（TilingB 在 2932 行起）                            |
| `op_kernel/flash_attention_score.cpp`                                                    | `__global__` 入口 + tiling key → 模板实例化（B 模板分支 628 行起） |
| `op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h`                                    | SplitB kernel 主体（1615 行）                                         |
| `docs/FA算子设计介绍.md`                                                                 | 官方设计文档（模板选择依据、CV 基本块概念、主从核模式）               |

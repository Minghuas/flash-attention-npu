# 我方代码库扩展点分析（P1）

> 目标：回答"SplitB 模板应该注入到哪里、如何优雅扩展"。
> 基线：当前分支 `perf-shortSeqLargeBatch`（含其它开发者合入的 dropout/ret-probs 等新增，
> **不含 ALiBi**——ALiBi 在独立分支 `alibi`/`feat-alibi`，commit 507bcd9，待合并）。

---

## 1. 我方 FA v2 结构全景

### 1.1 Kernel 侧（`csrc/ascend910/flash_attn_npu/`）

**主 kernel：`mha_fwd_kvcache.cpp`**（1106 行，一个文件承载 BSND/TND 两条路径）

- `SplitFuse::FAInfer<...>`（1015 行）模板参数 9 个：
  `InputDtypeQ, InputDtypeKv, IntermCalcPrec, PagedCacheFlag, MaskType(NO_MASK/MASK_CAUSAL/MASK_SWA),
  inputLayout(BSND/TND), LseModeT, IS_FD, HAS_SOFTCAP`
- `FAInferKernel` 类（42 行）主循环两段：
  1. **任务定位**（246-513 行）：读 tiling 的 `coreInfo[coreIdx]`（start/end BIdx/N1Idx/S1Idx/S2Idx），
     把属于本核的任务区间解析成 (batch, qN块, qS块) 三元组
  2. **KV-tile 流水**（632-968 行）：`MAX_KV_STACK_LEN=512` 的 KV 循环 + `PRE_LAUNCH=2` 三级
     workspace 槽位（qkReady/softmaxReady/pvReady 三个 CrossCoreFlag 做 cube↔vector 同步）
- 数据流（每个 (batch, qN块, qS块) 任务）：
  ```
  BlockMmadQK (L1Tile 128×128×128) → S 写 workspace GM
    → EpilogueOnlineSoftmax（mask/scale/softmax，跨 KV-tile dm 刷新）→ P 写 workspace GM
    → BlockMmadPV (L1Tile 128×128×256) → OTmp 写 workspace GM
    → EpilogueRescaleO（逐 tile rescale，末 tile 除 sum）→ O 写 GM
  ```
- **关键常量**（`kernel_common.hpp`）：`PRE_LAUNCH=2`、`MAX_KV_STACK_LEN=512`、
  `Q_TILE_CEIL=128`、`WORKSPACE_BLOCK_SIZE_DB=128×512`

**支撑文件：**

| 文件 | 角色 |
|---|---|
| `fa_block.h` | dispatch policy 定义：`MmadAtlasA2FAIQKT/FAIPVT`(STAGES=2)、`EpilogueAtlasA2OnlineSoftmaxT<LSE_MODE, SM_DTYPE, HAS_SOFTCAP>`、`EpilogueAtlasA2RescaleOT`、`EpilogueAtlasA2InitOutWhenZero` |
| `qk_matmul.hpp` / `pv_matmul.hpp` | `BlockMmad<MmadAtlasA2FAIQKT/FAIPVT>` 特化实现（L1/L0 tile copy + mmad） |
| `online_softmax.hpp`（1362 行） | `BlockEpilogue<EpilogueAtlasA2OnlineSoftmaxT>` 特化：3 个 operator()（有 mask / 无 mask / SWA），UB 布局 LS/LP/mask/softcap/lm/hm/gm/dm/ll/tv/gl（dm 占 (PRE_LAUNCH+1)=3 槽）；softcap 经 HAS_SOFTCAP 编译期门控 |
| `rescale_o.hpp` / `init_outputs.hpp` | RescaleO epilogue / SWA 全 mask 时输出初始化 |
| `CombineScale.hpp` | FlashDecode splitKV 合并 |

### 1.2 Dispatch 侧

- `fwd_dispatch.hpp` — `FwdLaunchArgs`（纯指针+标量）+ `launch_fwd<IS_TND>()` 运行时入口，
  编译期轴只有 **(dtype, IS_TND)**
- `fwd_dispatch_impl.hpp` — launch tree：paged × mask(3) × FD(仅 BSND) × softcap 的 if 分支
  展开，实例化 FAInfer（BSND 6 变体 / TND 4 变体）
- `autogen/generate_kernels.py` — 按 (dtype, layout) 生成 4 个独立 TU 并行编译：
  `fwd_dispatch_{fp16,bf16}_{bsnd,tnd}.cpp`

### 1.3 Host 侧（`flash_api.cpp`）

- 三个入口：`mha_fwd_kvcache`(359 行) / `mha_fwd`(642 行) / `mha_varlen_fwd`(851 行)
- **tiling 计算**：
  - `GetQNBlockTile(qSeqlen, groupSize)`：`Q_TILE_CEIL(128)/qSeqlen` 向下取偶，≥1 ——
    小 seqlen 时 head 块缩小（qNBlockNumPerGroup 变大 → 任务数变多）
  - `GetQSBlockTile(kvSeqlen)`：固定 128（S1 切块）
  - 任务总数 = Σ_b `qNBlockNum × qSBlockNum`；`fillCoreInfo` 按
    `perCoreTaskNum = ceil(totalTaskNum/blockDim)` 给每核分配连续任务区间
- **workspace**：per-core 4 块区（mm1Out/smOnlineOut/mm2Out/Update），每区
  `blockDim × 128×512 × (PRE_LAUNCH+1)` 字节
- **ALiBi/rotary/leftpad 当前被 TORCH_CHECK 拒绝**（417/666/889 行）；ALiBi 实现位于独立
  `alibi` 分支（`online_softmax.hpp` 加 `HAS_ALIBI_` 模板参数 + `EpilogueAtlasA2OnlineSoftmaxT`
  第 4 参数），合并时需同步支持

### 1.4 catlass 原语盘点（`csrc/catlass/include/catlass/gemm/`）

| 原语 | 位置 | 与 SplitB 的相关性 |
|---|---|---|
| `BatchedMatmul` kernel | `kernel/batched_matmul.hpp` | **关键候选**：batchCount + 批调度器，等价参考实现的 batch matmul 循环 |
| `BlockSchedulerAswt` / `BlockSchedulerIterateK` | `block/block_scheduler_*.hpp` | batch 迭代调度（ASWT = 自适应窗口？），待 P2 验证 L1 复用语义 |
| `BlockMmadSmall` (`MmadAtlasA2Small`) | `block/block_mmad_small.hpp` | 小块 matmul 变体（STAGES/ENABLE_UNIT_FLAG/ENABLE_SHUFFLE_K） |
| `BlockMmad` pingpong 系列 | `block/block_mmad_pingpong*.hpp` | 流水 matmul，可参考其 L1 管理 |
| `block_mmad_fai_qk/pv_*` | `block/block_mmad_fai_*.hpp` | FAInfer 现有块（normal/tla/head_tail/mx 变体） |

**待 P2 深挖**：catlass `BatchedMatmul`/`BlockScheduler` 的 batch 迭代是否等价参考实现的
`BATCH_LESS_THAN_L1`（batch 数据留 L1 免重复搬运）——这是 SplitB 性能的关键假设。

## 2. 与小 seqlen 性能直接相关的现状

1. `MAX_KV_STACK_LEN=512`：S≤128 时 KV 循环只有 1 次迭代，但**整套 KV-tile 流水机制照跑**
   （qkReady/softmaxReady/pvReady 同步、PRE_LAUNCH 预热、dm 刷新逻辑、rescale 逻辑）——
   每任务的固定开销占比极高
2. 任务粒度 = (B × N1块 × S1块)：大 B 时任务数足够多核均分，但**每个任务的有效计算量
   ∝ S1×S2×D 极小**，任务间 GM 搬运（Q/K/V/O/LSE 的读写）+ 同步开销不随任务变小而消失
3. `GetQNBlockTile` 在小 qSeqlen 时把 head 块缩到 1~128，进一步切碎任务

→ 与 SplitB 的对比：SplitB 把 B 并进 matmul batch 维，一次 BMM 消化 B.i×N2×G 个 (S1×S2)
小矩阵，GM 搬运次数从"每任务一次"降为"每核每 B 块一次"，同步也按 B 块摊销。

## 3. 扩展缝隙（注入点候选）

### 缝 A：kernel 形态 —— **独立新 kernel 文件（推荐）**

- 参考实现本身就把三个模板放三个文件（s1s2_bn2gs1.h / s1_bn2gs1.h / bn2gs1s2_b.h），
  我方跟随此模式：**新增 `mha_fwd_splitb.cpp`**（或类似名），不改 FAInfer 主路径
- 复用模式：仿 `fa_block.h` 定义新 dispatch policy（如 `MmadAtlasA2FAISplitB`），
  kernel 主体结构参考 `FAInferKernel` 但循环换成"B 块循环 + batch matmul + 单次 softmax"
- 不侵入现有 kernel → 对现有路径（含 FD/kvcache/paged）零风险
- 代价：softmax epilogue / rescale / mask 处理需要新实现或参数化复用（见缝 D）

### 缝 B：dispatch 注入 —— **独立 launch 函数（推荐）**

- `fwd_dispatch.hpp` 现有 `launch_fwd<IS_TND>`；新增 `launch_fwd_splitb`（或 FwdLaunchArgs
  加 `is_split_b` 标志 + 独立 launch tree），**不动现有 launch tree**
- autogen 增加新 TU（如 `fwd_dispatch_{fp16,bf16}_splitb.cpp`），generate_kernels.py 加
  一个 layout key——机械扩展，编译并行度不变
- host 判定后直接调 `launch_fwd_splitb`，与现有路径互斥分支

### 缝 C：host tiling 分支

- `mha_fwd` 入口（642 行）在形状判定后分叉：
  ```
  if (alignedS2 ≤ 128 && N2×G×alignedS1×alignedS2×dtype ≤ 128KB) → SplitB tiling
  else → 现有 tiling
  ```
- **新 tiling 结构**（不混用 `FAInferTilingData`——语义完全不同）：
  仿参考 `coreParams/multiCoreParams`：`bBaseSize=1, bOuterSize=B, splitFactorSize,
  s1BasicBlock(UB 预算), s2BasicBlock=alignedS2, s1Vec2BasicBlock` + 新 workspace 布局
- workspace 公式：`预留 + (mm1 + mm2) × 2(ping-pong) × coreNum`（参考式）
- 注意点：我方 UB 布局（online_softmax 的十几个 buffer）与参考 InitBuffer 不同，
  **s1BasicBlock 的 UB 预算公式必须按我方实际 UB 占用重推**，不能照抄 8KB 假设

### 缝 D：feature 支持（D3 原则：全量支持，与 feature 正交）

当前分支 fwd feature 集 = {NO_MASK, CAUSAL, SWA} × {softcap} × {paged/FD(kvcache), 非 paged}；
ALiBi 在 `alibi` 分支待合并。SplitB 移植设计要点：

1. **mask 三种**在"batch 循环 + S2 整块"结构下的表达：
   - NO_MASK/CAUSAL：参考实现已有对应（attenMask 压缩模式），照搬逻辑
   - SWA：参考实现无 SWA（用 band mask 近似）；我方 SWA 的 kvStart/kvEnd 窗口逻辑需在
     SplitB 的 batch 结构下重新推导（S2 不切分后窗口计算反而更简单，只需算一次）
2. **softcap**：单次 softmax 结构下更容易（无跨 tile 状态），仿 HAS_SOFTCAP 编译期门控
3. **paged/FD**：SplitB 触发条件天然排除了 paged 大 seqlen 场景（S2≤128），但小 seqlen +
   paged 理论存在（如推理首 token 场景）；第一版可先不支持并显式校验，后续补
4. **ALiBi**：`alibi` 分支合并后，在 SplitB 的 softmax 阶段仿 `HAS_ALIBI_` pattern 加
   偏置路径（ALiBi 的 BNS 布局 → SplitB 的 BN2GS1S2 布局换算需在 P2 设计）

### 缝 E：autogen 与构建

- `autogen/generate_kernels.py` 的 layout 列表加 splitb 项，生成新 TU
- `setup.py`/构建脚本确认新 .cpp 被纳入编译（P2 确认）

## 4. 遗留问题与解答

### ✅ 问题 1：catlass batch matmul 的 L1 复用语义（2026-08-14 解决）

**结论：现有路径无跨 batch L1 复用；"batch 留 L1"的语义在 catlass 里只有脚手架、无实现。
SplitB 的 L1-resident batch 要自己写。**

证据（深读 `csrc/catlass/include/catlass/gemm/`）：

| 维度 | 结论 |
|---|---|
| batch 迭代位置 | **kernel 层**（AIC `operator()` 内）：`batched_matmul.hpp:131` `coreLoops = batchCount × M_tiles × N_tiles`，batch 展平进一维 task 空间；`block_swizzle.hpp:73` `batchIdx = taskIdx/(M_tiles*N_tiles)` 反推。一次 launch 处理整个 batch |
| A/B 的 L1 加载（标准路径） | **每 batch 重新 CopyGMToL1**：`batched_matmul.hpp:164` 每次循环算 `batchOffsetA = batchIdx×strideA` 传新 tensor view；`block_mmad_small.hpp:185` 每次 `copyGmToL1A`。**无跨 batch L1 复用** |
| L1-resident batch 脚手架 | **存在但未实现**：`batched_matmul_tla.hpp:329` `maxL1Batch = L1_SIZE/STAGES/(L1A_SIZE+L1B_SIZE)`（精确对应 CANN `BATCH_LESS_THAN_L1`）+ batch 组迭代 + prefetch（329-387 行）。但配套的 `BlockMmad<MmadMultiBatch,...>` 偏特化**不存在** → 触发 `DEPENDENT_FALSE`（`block_mmad.hpp:99`）。grep `MmadMultiBatch` 全仓库除自身外零引用 |
| FAInfer 现状 | batch 在 kernel 外层循环（`mha_fwd_kvcache.cpp:265/321`），BlockMmadQK/PV 无 batch 维；L1 复用仅限"batch 内 Q 跨 KV-tile 复用"，非跨 batch |

**对 SplitB 移植的影响**（更新缝 A 的技术选型）：
- 照搬现有 `BatchedMatmul`/`BlockMmadSmall` 路径 → 退化成 `B.i×N2×G` 次独立小 matmul，**无性能收益**（L1 不复用 = GM 带宽爆炸）
- 走 `BatchedMatmulTla` + `MmadMultiBatch` → kernel 层调度逻辑现成，但**需补 BlockMmad 偏特化**（构造函数按 maxL1Batch 分配 L1、operator() 加 batch 维内循环从 L1 取 tile）
- **更现实的路径**：不套 BatchedMatmulTla 框架，直接在现有 `block_mmad_fai_qk_normal.hpp`/`block_mmad_fai_pv_normal.hpp` 里加 batch 维内循环 + L1 复用——因为 FA 的 matmul 不是纯独立 batch（Q 跨 KV-tile 复用、P 需先 softmax），套纯 GEMM batch 框架反而别扭
- **风险标记 `[需 NPU 验证]`**：自写 L1-resident batch 的性能收益需上板验证（L1 容量、maxL1Batch 实际值、是否眕住 GM 带宽）

### 问题 2：我方 UB 预算重推（P2 待解）

online_softmax 的 UB 布局（LS/LP/mask/softcap/lm/hm/gm/dm/ll/tv/gl）在"单次 softmax、
无 dm 刷新"时的最小占用，反推 s1BasicBlock 上限。dm 当前占 (PRE_LAUNCH+1)=3 槽，SplitB
单次 softmax 后可省 → UB 预算比我方现状更宽松。

### 问题 3：workspace 区分（P2 待解）

SplitB 的 workspace（mm1/mm2 ping-pong per-core）与现有 `WORKSPACE_BLOCK_SIZE_DB` 布局
不同，host 分配逻辑要区分。

### 问题 4：性能基准缺口（✅ 2026-08-14 补齐）

bench 脚本已写：[../bench/bench_attention.py](../bench/bench_attention.py)（FA v2 vs
npu_fusion_attention，可配置网格）。冒烟测试（batch=4, seqlen=64/128, nheads=8, D=128）已
确认两边跑通且我方慢约 2.3~2.4x。完整网格数据见 [../results/](../results/)。

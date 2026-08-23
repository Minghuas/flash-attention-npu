# perf — 大 Batch 小 SeqLen 性能优化项目

> 本目录是本项目的记忆跟踪文档库：记录问题、决策、分析见解与阶段计划，供长期追踪。
> 状态：**P0/P1 完成，性能数据已测，待进 P2 集成设计**（2026-08-14 更新）

---

## 性能现状速览（2026-08-14 实测，910B4-1）

我方 FA v2 vs baseline（`npu_fusion_attention`），fp16 / nheads=8 / D=128：

| 场景 | fa_ms | base_ms | speedup | 结论 |
|---|---:|---:|---:|---|
| batch=1024, s=64, non-causal | 104.3 | 0.82 | **0.01x** | 慢 **127 倍** |
| batch=1024, s=128, causal | 135.3 | 1.58 | **0.01x** | 慢 **86 倍** |
| batch=512, s=128, non-causal | 39.5 | 0.82 | 0.02x | 慢 48 倍 |
| batch=16, s=2048, non-causal | 44.7 | 2.22 | 0.05x | 慢 20 倍 |
| batch=1, s=128, non-causal | 0.68 | 0.63 | 0.92x | 最接近（数据量极小）|

**核心发现：**
1. **issue 完全证实且极其严重**——所有配置下我方都慢，大 B 小 S 劣化 80~127 倍。
2. **我方算力天花板 ~6 TFLOPS（~3% 峰值）且与问题规模无关** = 稳态效率瓶颈（通用 tiling 每任务固定开销），不是 launch 开销。
3. **baseline 可达 124 TFLOPS（~78% 峰值）**，延迟随 batch 增长极缓（SplitB 的 batch 进 matmul batch 维摊销有效）。
4. **全范围无交叉点**——即使 seqlen=2048 我方仍劣化（0.05~0.12x）；本项目聚焦 `S2≤128`（参考 SplitB 区间），中大 seqlen 留后续。

完整数据与分析见 [results/ANALYSIS.md](results/ANALYSIS.md)；原始 CSV/JSON 在 [results/](results/)。

---

## 1. 问题定义

**Issue：** 大 Batch Size + 小 Sequence Length（S < 128）时，FlashAttention 算子性能显著劣于
Standard Fused Attention（即 `torch_npu.npu_fusion_attention`，CANN 自研融合算子）。

**根因（已验证的推理）：** 我方 FA v2 是通用 tiling 架构——S1×S2 都切块 + 跨 KV-tile 的
FlashSoftmax 刷新流程（dm rescale），核间按 (B, N1, S1) 轴切任务
（见 [mha_fwd_kvcache.cpp](../../csrc/ascend910/flash_attn_npu/mha_fwd_kvcache.cpp) 任务分布代码）。
当 S < 128 时 S2 切块退化（块数≈1），但每个 (b, h, q块) 任务的 mask 处理、softmax 同步、事件
开销不减少，每个核分到的有效工作量很小，**开销占比爆炸**。

而 baseline（npu_fusion_attention = CANN FlashAttentionScore 算子）内部按 shape 自动选择模板，
对小 shape 走专门优化的 **SplitB 模板**，因此显著快于我们。

## 2. 决策记录

| # | 日期 | 决策 | 理由 |
|---|------|------|------|
| D1 | 2026-08-13 | **否决 "BSND→TND" 方案** | 语义正确（等长 varlen≡定长）但无效：我方 TND 路径同样按 batch×head×q块 在核间切任务，与 BSND 同构（证据：[mha_fwd_kvcache.cpp:323-343](../../csrc/ascend910/flash_attn_npu/mha_fwd_kvcache.cpp#L323-L343) 中 TND 分支同样逐 batch 累计任务数），换汤不换药 |
| D2 | 2026-08-13 | **采用 CANN 官方 SplitB 模板方案**（移植 `FlashAttentionScoreTilingB` + `flash_attention_score_bn2gs1s2_b.h`） | CANN 对大 B 小 S 的官方答案：核间切 B 轴；核内 S1/S2 不切分（S2 整块一次算完、无 FlashSoftmax 刷新）；B.i×N2×G 作为 matmul batch 维循环；Vector 侧综合切分 B.i×N2×G×S1。CV 基本块变大、交互次数变少 |
| D3 | 2026-08-13 | **feature 正交性原则**：SplitB 是模板级（tiling 级）底层优化，与 kvcache/paged/softcap/ALiBi/SWA 等 feature 是不同层次——新模板需**全量支持**这些 feature，不按 feature 裁剪范围 | 用户明确：这些 feature 无论是否开启，优化都要支持；它们是正交维度 |
| D4 | 2026-08-13 | 基准测试角色调整：从"先测后优"的探索工具，改为 **P4 验证收益 + P2 确定触发条件的数据输入** | 方案已拍板（D2），无需先测再定方向；但实现前仍需跑一轮当前 FA vs baseline 拿到差距数据 |
| D5 | 2026-08-14 | **完整照搬原则**（Q-A/Q-B 拍板）：触发闸门严格照搬不放宽（`alignedS2≤128 && N2G×S1×S2×dtype≤128KB`）；小 B 不回落旧路径；**一次完整移植**参考 kernel 结构（四段+3槽流水+ping/pong+matmul batch 维），不做简化版先行；matmul 栈用 **AscendC 高阶 API**（参考原生写法，含 BATCH_LESS_THAN_L1），不绕道 catlass | 用户明确：参考方案是深思熟虑的设定，先照搬、完整适配所有功能特性和场景、跑通测试、确认性能提升，之后的优化改进是后话；不要一下子追求太多，避免潜在麻烦导致目标难以实现 |
| D6 | 2026-08-14 | **matmul 栈取舍（方案 A）**：新 SplitB kernel 用 AscendC 高阶 API（照搬），旧 kernel 保持 catlass，两栈在同一 .so 共存；统一 catlass（方案 B）被否决但留档——需自研 batch 驻留 matmul，高风险且偏离照搬。**后续如有必要，在完成并验证的代码上再改写为 catlass** | 用户倾向 catlass 但认可自研风险；决定一步一步来不冒险：先 A 完成移植建立正确性/性能基线，改写有对照后再考虑 B |

## ⚠️ 当前问题记录（2026-08-16，P3 步 1 执行中）

**进展**：P3 步 1 骨架全部就位（8 文件），**R1 编译验证通过**（用户手动编译成功链接）——
AscendC matmul 高阶 API（`matmul::Matmul`/`REGIST_MATMUL_OBJ`/`matmul_tiling::MatmulApiTiling`）
在我们 torch 扩展构建下可编译可链接。期间修复：AscendC 命名空间 using、`__gm__` 指针
reinterpret 禁令（tiling GM→栈拷贝）、命名空间统一 `SplitB`、inline 符号发射位置
（`launch_fwd_splitb` 移轻量头）。

**阻塞（R2 兑现）**：冒烟测试 aicore 异常 507015（空 Process 也崩，锁定 matmul 注册链路）。

**根因分析（已查明）**：CANN 存在两套**平行的**混合核编程范式，不互通——

| | 范式 1：显式双段 | 范式 2：KFC 隐式 |
|---|---|---|
| 代表 | 我方 FAInfer、ops-transformer mla_preprocess | ops-transformer flash_attention_score 全家（含 SplitB） |
| cube 侧 | 开发者手写（`__DAV_C220_CUBE__` 段 + mmad intrinsics/catlass） | 源码不存在，`REGIST_MATMUL_OBJ` 经 `ASCEND_IS_AIC`+`KfcCommClient` 自动生成 |
| matmul | catlass BlockMmad / 手写 intrinsics | `matmul::Matmul` 高阶 API |
| 核类型声明 | DAV 宏段自证（auto-infer=false 可编） | `KERNEL_TASK_TYPE_DEFAULT(MIX_AIC_1_2)` + **auto-infer 必须开启** |
| 同步 | CrossCoreFlag（fftsAddr） | KFC 内建 |

实证：全 ops-transformer 无一 kernel 混用两者。我们的 SplitB 骨架照搬参考（范式 2 风格，
无 DAV 段），但 setup.py 全局 `--cce-auto-infer-kernel-type=false`（范式 1 约定）→ kernel
以错误类型编译 → KFC workspace（`GetSysWorkSpacePtr`）拿到垃圾 → 注册时写坏内存 → 507015。
补声明 `KERNEL_TASK_TYPE_DEFAULT` 后编译报 `unknown type name '__builtin_cce_kernel_type_set'`
（builtin 被 `__CCE_ENABLE_AUTO_INFER__` 守卫，auto-infer=false 下不存在，见
bisheng `__clang_cce_aicore_functions.h:30-52`）。

**候选出路**：
- **A. per-TU 开 auto-infer**：仅 splitb 2 个 TU 加 `--cce-auto-infer-kernel-type=true`，
  与参考编译模式对齐；现有 kernel 不动。兜底：`REGIST_MATMUL_OBJ` 的 workspace 是显式参数，
  KFC 区不可用时可传自分配 workspace
- **B. catlass 实现参考的 tiling+计算模式**（范式 1 重写，见设计文档更新讨论）：保留
  SplitB 算法核心（切B/S2不切/单次softmax/批×全头粒度），matmul 换 catlass 逐头调用；
  损失 `BATCH_LESS_THAN_L1` L1 驻留语义（GQA 时 K/V 每组重载）与 aiv 切片基数
- ~~全局开 auto-infer~~（连累全部现有 kernel，已否决）
- ~~显式 DAV 段 + matmul API 混用~~（实证两范式不互通，等于放弃 matmul API，已否决）

| D7 | 2026-08-16 | **路线切换：catlass 实现**（范式 1 重写）。算法层照搬参考（切B/S2不切/单遍softmax/批×全头粒度/闸门照搬），机制层用 FAInfer 已验证范式（BlockMmadQK/PV + DAV 双段 + CrossCoreFlag + fftsAddr）。范式 A 代码归档至 [archive/ascendc-matmul-paradigm-v2/](archive/ascendc-matmul-paradigm-v2/)，host tiling 公式/dispatch 骨架继承。设计 v3 + 执行步骤 S1~S6 见 [design/splitb_integration.md](design/splitb_integration.md) | 风险对比反转：范式 A 撞在运行时基础设施（auto-infer/KFC/MIX launch 三重未知），范式 1 每个机制都在本仓库端到端验证过；用户确认与原有代码范式兼容的诉求 |

## 3. 关键结论（分析速记）

- **baseline 身份**：`npu_fusion_attention` 的底层实现就是 ops-transformer 库的
  FlashAttentionScore 算子（即我们参考的代码）。它按 S2 大小自动选模板，小 shape 走 SplitB。
  → 我们与 baseline 的差距 = 通用 tiling vs SplitB tiling 的差距。
- **TriDao FA 同样有此问题且被隐藏**：所有 benchmark 的 `bs_seqlen_vals` 从 seqlen=512 起步
  （保持 B×S=16384），论文只报 ≥512；GPU tile 128×128 在 S<128 时大半浪费，靠 GPU 巨量 SM 掩盖。
  → 大 B 小 S 是 FA 类 tiling 架构的固有弱点，不是 NPU 特有。
- **SplitB 模板触发条件**（`IsCapable`，精确值）：
  1. `alignedS2 ≤ 128`（HIGH_PERF_SUPPORT_S2_BASIC）
  2. `N2×G×alignedS1×alignedS2×dtype_bytes ≤ 128KB`（blockBSizeLimit_=64K×2B，L1 适配检查）
- **模板选择优先级**（注册顺序即尝试顺序）：DropMask(90) > VarLen(94) > S1s2Bn2gs1SameAB(95) >
  S1s2Bn2gs1(96, S2>1024) > S1Bn2gs1(97, 128<S2≤1024) > **TilingB(98, fallback=小 shape)**。

## 4. 阶段计划

| 阶段 | 内容 | 产出 | 状态 |
|------|------|------|------|
| **P0** | 参考实现深度研读（kernel + tiling 两侧） | [analysis/reference_splitb_kernel.md](analysis/reference_splitb_kernel.md)、[analysis/reference_splitb_tiling.md](analysis/reference_splitb_tiling.md) | ✅ 完成（2026-08-13） |
| **P1** | 我方代码库扩展点分析（FAInfer 结构映射、catlass batch 原语盘点、扩展缝隙识别） | [analysis/our_fa_extension_points.md](analysis/our_fa_extension_points.md)（含 catlass batch 语义结论） | ✅ 完成（2026-08-14） |
| **P1.5** | 基准测试（FA v2 vs npu_fusion_attention） | [bench/bench_attention.py](bench/bench_attention.py) + [results/ANALYSIS.md](results/ANALYSIS.md) | ✅ 完成（2026-08-14） |
| **P2** | 集成设计方案（新模板形态、tiling 结构、dispatch 触发条件、feature 全量支持方案、测试与基准计划） | [design/splitb_integration.md](design/splitb_integration.md) | 🔍 **待评审**（2026-08-14 出稿） |
| **P3** | 实现（顺序：host tiling → kernel → dispatch → 测试） | 代码改动 | ⏳ |
| **P4** | 验证：正确性（对比 golden）+ 性能（对比 npu_fusion_attention 与当前 FA） | [bench/](bench/) 扩展 + 结果报告 | ⏳ |

## 5. 目录索引

```
perf/
├── README.md                          # 本文件：问题、决策、性能现状、计划、索引
├── analysis/
│   ├── reference_splitb_deep_dive.md  # ⭐ SplitB 深度解读（动手前必读：调用路线/计算模式/
│   │                                  #    四阶段形态/数值走查/UB/workspace/流水/对照表/照搬边界）
│   ├── reference_vec1_tiling.md      # Vec1 专项：单 Batch 的 softmax tiling 与双层循环
│   ├── reference_vec2_design.md      # Vec2 专项：归一化与输出设计（Div 广播/布局步长/UB 分时复用）
│   ├── reference_bmm1_design.md      # BMM1 专项：QKᵀ 批 matmul + catlass 三条实现路径
│   ├── reference_bmm2_design.md      # BMM2 专项：PV 批 matmul + catlass 实现方案
│   ├── reference_splitb_kernel.md     # SplitB kernel 结构解读（P0）
│   ├── reference_splitb_tiling.md     # TilingB 参数推导 + 模板选择条件链（P0）
│   └── our_fa_extension_points.md     # 我方扩展点分析 + catlass batch 语义结论（P1）
├── design/
│   └── splitb_integration.md          # 集成设计方案（P2，评审稿 2026-08-14）
├── bench/
│   └── bench_attention.py             # 基准脚本：FA v2 vs npu_fusion_attention
└── results/
    ├── ANALYSIS.md                    # 三段 bench 结果分析（2026-08-14）
    ├── bench_smallseq_fp16.csv/.json  # 段1：小 seqlen 全 batch（issue sweet spot）
    ├── bench_seqlensweep_fp16.csv/.json # 段2：seqlen 扫描找交叉点
    └── bench_causal_fp16.csv/.json    # 段3：causal 确认
```

## 6. 参考代码位置

- 参考实现（CANN ops-transformer 开源库，已迁入仓库根目录）：
  - kernel：`ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h`
  - 姊妹模板：`flash_attention_score_s1_bn2gs1.h`、`flash_attention_score_s1s2_bn2gs1.h`（对比用）
  - tiling host：`ops-transformer/attention/flash_attention_score/op_host/arch22/flash_attention_score_tiling_general.cpp`
    （类 `FlashAttentionScoreTilingB`，注册优先级 98）
  - 设计文档：`ops-transformer/attention/flash_attention_score/docs/FA算子设计介绍.md`（模板选择依据原文）
- 我方 FA v2 主体：`csrc/ascend910/flash_attn_npu/`（kernel 主体 `mha_fwd_kvcache.cpp`、
  host tiling `flash_api.cpp`、dispatch `fwd_dispatch.hpp` + `autogen/`）
- TriDao FA（GPU 对照）：`/home/liaojy/workspace/FA/flash-attention-dao/`

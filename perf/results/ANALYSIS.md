# 基准测试结果分析（2026-08-14）

> 设备：Ascend910B4-1（单卡）｜dtype=fp16｜nheads=8｜headdim=128｜warmup=3 repeat=10
> 脚本：[../bench/bench_attention.py](../bench/bench_attention.py)｜原始数据：bench_*.csv / .json

## 结论一句话

**issue 完全证实，且比预期严重得多**：我方 FA v2 在所有测试配置下都慢于 baseline
（`npu_fusion_attention`），大 batch 小 seqlen 场景劣化达 **80~127 倍**。我方算力利用率
长期卡在 ~3-6 TFLOPS（~3% 峰值），baseline 可达 ~124 TFLOPS（~78% 峰值）。

## 三段数据要点

### 段1：小 seqlen 全 batch 扫描（non-causal）—— issue 核心 sweet spot

| batch | seqlen | fa_ms | base_ms | speedup | fa_TF | base_TF |
|------:|-------:|------:|-------:|--------:|------:|--------:|
| 1 | 64 | 0.70 | 0.46 | 0.66x | 0.0 | 0.0 |
| 32 | 128 | 1.72 | 0.27 | 0.16x | 1.2 | 8.0 |
| 128 | 128 | 6.34 | 0.36 | 0.06x | 1.4 | 23.6 |
| 512 | 128 | 39.48 | 0.82 | 0.02x | 0.9 | 41.7 |
| **1024** | **64** | **104.3** | **0.82** | **0.01x** | 0.2 | 21.1 |
| **1024** | **128** | **118.9** | **1.43** | **0.01x** | 0.6 | 48.0 |

→ batch 越大、seqlen 越小，劣化越极端。**speedup 随 batch 单调下降**（0.66x → 0.01x），
正是 issue 描述的"大 batch 小 seqlen 性能缺陷"的签名特征。

### 段2：seqlen 扫描（找交叉点）

| batch | seqlen | speedup | fa_TF | base_TF |
|------:|------:|--------:|------:|--------:|
| 1 | 128 | 0.92x | 0.1 | 0.1 |
| 1 | 2048 | 0.12x | 5.3 | 45.2 |
| 8 | 2048 | 0.06x | 6.1 | 109.7 |
| 16 | 2048 | 0.05x | 6.2 | 123.9 |

→ **全范围无交叉点**：我方在所有 seqlen 下都更慢。最接近持平的是 batch=1 s=128（0.92x，
数据量极小两者都没跑满）。我方 TFLOPS **天花板约 6 TFLOPS**（与问题规模无关——稳态效率瓶颈）；
baseline 随规模逼近峰值（batch=16 s=2048 达 124 TFLOPS ≈ 78% 峰值）。

### 段3：causal 确认（SplitB 必须支持 causal）

| batch | seqlen | fa_ms | base_ms | speedup |
|------:|-------:|------:|-------:|--------:|
| 128 | 128 | 18.4 | 0.80 | 0.04x |
| 512 | 128 | 55.5 | 0.95 | 0.02x |
| 1024 | 64 | 118.8 | 0.96 | 0.01x |
| 1024 | 128 | 135.3 | 1.58 | 0.01x |

→ causal 与 non-causal 同构劣化。**SplitB 支持 causal 是刚需，收益空间巨大。**

## 机理印证（与 [../analysis/](../analysis/) 的分析一致）

1. **我方 TFLOPS 天花板 ~6 TFLOPS 且与规模无关** → 稳态效率瓶颈，非 launch 开销。
   根因即通用 tiling 的每任务固定开销：KV-tile 流水（`MAX_KV_STACK_LEN=512`，小 seqlen 时
   仅 1 次迭代但 PRE_LAUNCH=2 预热 + qkReady/softmaxReady/pvReady 三同步照跑）+
   跨 KV-tile dm rescale + workspace GM 往返。
2. **baseline 延迟随 batch 增长极缓**（s=64 时 batch 1→1024 仅 0.46→0.82ms）→
   SplitB 的"batch 进 matmul batch 维、一次 BMM 消化多 batch、同步按 B 块摊销"确实有效。
3. **speedup 随 batch 单调下降** → 印证"每个 (b,h,q块) 任务有效计算量小、固定开销占比高"，
   batch 越大浪费累积越多。

## 对 SplitB 项目的指导（更新 P2 输入）

1. **触发条件应放宽**：参考实现的 `alignedS2 ≤ 128` 太窄——数据显示 seqlen=2048 时我方仍
   劣化（0.05~0.12x）。但 seqlen>1024 后参考实现走的是 `S1s2Bn2gs1`（带 FlashSoftmax 刷新）
   而非 SplitB。**我方当前在所有 seqlen 都劣化，说明通用 tiling 路径整体效率低下**，
   SplitB 只是补"小 seqlen"这一段；中大 seqlen 可能需要别的优化（非本项目范围，记录留档）。
2. **本项目聚焦 `S2 ≤ 128`（与参考 SplitB 触发条件一致）**：这是 issue 明确的场景，也是
   参考 SplitB 设计的目标区间。预期收益：把 batch=1024 s=64 的 104ms 拉到接近 baseline 的
   ~1ms 量级（~100x 空间）。
3. **causal 路径必须一并支持**（段3 证实刚需）。
4. **catlass 限制**（见 [../analysis/our_fa_extension_points.md](../analysis/our_fa_extension_points.md) §4.1）：
   现有 catlass 路径无跨 batch L1 复用，SplitB 的 L1-resident batch 语义需自写 BlockMmad
   扩展——这是性能收益成立的关键假设，**必须上板验证**。

## 数据局限与后续

- **repeat=10 偏小（用户指出，2026-08-14）**：本页数据为 repeat=10 的中位数，量级结论可信
  （差距达 1~2 个数量级），但后续正式测试 repeat 取 **≥100**（bench 脚本默认值已改为 100）。
- nheads=8（偏小）：真实负载常 nheads=32+，但 nheads 对两侧影响近似同向，speedup 量级可信；
  后续可加 nheads=32 抽查。
- 只测 fwd（bwd 待 Q7 明确后补，优先级低）。
- 未测 GQA（H≠H_kv）、softcap、bf16——SplitB 设计需覆盖，但 bench 优先级可后置。
- causal 的 atten_mask 构造（torch.triu）有少量 host 开销，但相对 ~100ms 的 fa 延迟可忽略，
   不影响 speedup 量级结论。

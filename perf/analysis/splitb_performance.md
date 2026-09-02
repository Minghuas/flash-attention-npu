# SplitB 性能分析与优化规划（v1，2026-08-31 建立）

> 本文档持续维护：**版本迭代记录**（§1）、**测试方法学**（§2）、**数据档案**（§3）、
> **瓶颈清单**（§4，带证据）、**优化方案规划**（§5，带状态跟踪）、
> **指标收集计划**（§6）。每次剖析/bench/优化后更新对应章节并在 §1 记一行。
> 关联：perf/devlog2.md（逐日日志）、perf/results/profile/（原始数据）、
> perf/analysis/measurement_lessons.md（测量方法学教训，#60-#65 全案）。

---

## 1. 版本迭代记录

| 版本 | 日期 | 变更 | 形状 | kernel 时长 | 关键指标 |
|---|---|---|---|---|---|
| v3.4a | 08-31 | 闸门放宽+多核翻转+清理（-g 构建） | B1024/S128/H8/kvH4/D128, 20 核 | 1228.6us | cube 28-30%、MTE2 70%、FIXP 78%、vec 44% |
| v3.4b | 08-31 | 去 -g 重编（全指标采集版） | 同上 | **1262.0us** | 同上 + Occupancy 不均衡 41.7%（尾块效应） |
| v3.5a | 09-01 | O7+O1+ENABLE_DEBUG（MHA 形态） | B1000/S128/H8/kvH8/D128 | 1491us | MTE2 92%、Roofline=latency-bound(compute) |
| **v3.5a-gqa** | 09-01 07:32 | 同上，GQA 驻留路径激活（G=4），**unit_flag=true** | **B1024/S128/H8/kvH2/D128** | **1186us** | MTE2 62%（vs MHA 92%↓30pp）、FIXP 82%、cube 26%、vec 44%、Occupancy 4.3% |
| **v3.5a-false** | 09-01 09:28 | 同上，**unit_flag=false**（A/B 实验） | 同上 | 1224us | FIXP 82→44%（-38pp 但 wall +3%——**FIXP 瓶颈假说证伪** #59）；scalar+52、stall+141 |
| **v3.5b-gqa** | 09-02 02:17 | 切回 **unit_flag=true**（含 #57 FIX_M 条件修复） | 同上 | **1102us** | **历史最优**；MTE1 stall 392us（#56 的 502↓22%）；MTE2 57%、FIXP 78%、cube 28% |
| （CANN 基线） | 09-02 03:16 | torch_npu npu_fusion_attention | 同上 | **963.2us** | FlashAttentionScore_mix_aic；scalar_mte1_stall **0us**、FIXP 96%、scalar 98%、GM 读 52.7MB/核（无 GQA 组复用，详析 #62） |
| （旧路径基线） | 08-31 | FAInfer 参考 | B1024/S128/H8/kvH4/D128 | 6196us | cube 4.4%、scalar 86.6% |

## 2. 测试方法学

- **bench（端到端）**：`perf/bench/bench2.py`（MHA）/`bench2_gqa.py`（GQA）——
  fa_ms 含 host 全链（routing/tiling/workspace 分配/launch/sync）；repeat=50-100。
  ⚠ #65 教训：flash_attn_func 返回**裸张量**，调用侧禁止 `out, *_ =` 解包
  （沿 batch 维迭代创建 B 个 view，~1-3ms/次，曾伪装成 host 开销）。
- **prof（真机内核指标）**：`bash perf/profile/run_test.sh`（msopprof prof 模式，
  默认 = Default 指标集；`AIC_METRICS=...` 追加）。kernel 时长 = OpBasicInfo 的
  Task Duration——**不含 host 开销**。
- **torch 基线**：`bash perf/profile/run_test.sh --test-torch`（→ prof_torch/ 目录，
  内核 `*FlashAttention*`）。用法详见 perf/profile/msopprof_guide.md。
- **sim（指令级）**：`bash perf/profile/run_test.sh sim`——功能级计时（非周期精确），
  用于指令归属/事件计数/源码行热点；窗口受 --timeout 限制。
- **bench vs prof 的差值 = host+launch 开销**（§4 瓶颈 A 的量化手段）。
- 已知坑（devlog2 #51）：`--aic-metrics` 含 Source 需 -g 且曾致分析器 core dump；
  采集成功≠分析成功，逐指标 bundle 单独跑可隔离失败。

## 3. 数据档案（perf/results/profile/ 内索引）

| 运行目录 | 时间 | 模式 | .so 状态 | 形状 | 结论 |
|---|---|---|---|---|---|
| prof_torch/archive/OPPROF_...031641 | 09-02 03:16 | prof(torch) | CANN .so | B1024/S128/H8/kvH2/D128 | **基线 963us**；stall=0、FIXP 96%、GM 读 52.7MB/核——对比详析 #62 |
| prof/archive/OPPROF_...021711 | 09-02 02:17 | prof | v3.5b, unit_flag=true | 同上 | **1102us 历史最优**；stall 402us、MTE2 57%、GM 读 33.3MB/核（O1 生效） |
| prof/archive/OPPROF_...073224 | 09-01 07:32 | prof | v3.5a, unit_flag=true | **B1024/S128/H8/kvH2/D128（GQA G=4）** | **1186us**；MTE2 62%（vs MHA 92%↓30pp）、FIXP 82%、cube 26%、vec 44%、Occ 4.3%；**O1 驻留路径激活** |
| sim/archive/OPPROF_...073359 | 09-01 07:33 | sim | 同上 | 同上 | 待分析 |
| prof/archive/OPPROF_...031512 | 09-01 03:15 | prof | v3.5a | B1000/S128/H8/kvH8/D128（MHA） | 1491us；MTE2 92%、Roofline=latency-bound(compute) |
| prof/archive/OPPROF_...024759 | 09-01 02:47 | prof | v3.5a | 同上 | 1501us（与 03:15 一致） |
| prof/OPPROF_...014710 | 09-01 01:47 | prof | v3.5a | B1000/S128/H1/D128 | 168.7us——**对照 bench 3.2-4.0ms ⇒ host 固定开销 ~3ms** |
| prof/OPPROF_...100735 | 08-31 10:07 | prof | v3.4b | B1024/S128/H8/kvH4/D128 | 1262us；Occupancy 41.7% = 尾块效应 |
| prof/OPPROF_...094810 | 08-31 09:48 | prof | v3.4a | 同上 | 首份完整真机画像 |
| sim/OPPROF_...091447 | 08-31 09:14 | sim | v3.4a | 同上 | 窗口≈批 0；670 flags/核/批 |
| prof/OPPROF_...081801 | 08-31 08:18 | prof | 旧 .so | 旧路径 FAInfer | 旧路径基线（cube 4.4%） |

## 4. 瓶颈清单（按证据强度排序）

### A. ~~host 开销~~ → **✅ 结案：是 bench2.py 解包 bug，host 真实 0.4ms（#65）**
- bench2.py 的 `out, *_ = flash_attn_func(...)` 对**裸张量**解包（该函数非 tuple
  返回），沿 batch 维迭代创建 B=1024 个 view（~1-3ms/次）——伪装成 host 开销。
  九轮二分定案（devlog #65），两 bench 已修复。
- **修正后 E2E**：MHA 1.840 vs torch 1.294（0.70x）；GQA 1.457 vs 1.054（0.72x）。
  组成：GQA kernel 1.10+host 0.36 vs torch 0.96+0.09。
- 残余 host 0.36ms 中可优化项：workspace 按形状缓存（~0.1ms 级，次优先）。

### B. MTE2 装载（cube 核 70% 忙，第一忙管道）
- **证据**：PipeUtilization `aic_mte2_ratio=0.7`；GM_to_L1 52.4MB/核。
- **成分**：Q/K/P/V 四类装载；**K 在 GQA 组内重复装载**（kv 头 K 每 q 头重读一次，
  G=2 两倍冗余、G=8 八倍冗余）——无冗余部分（Q/P/V 各一次）≈ 一半。

### C. FIXP 旗标编排（❌ 已证伪为非瓶颈）
- **证据**：`aic_fixpipe_ratio=0.8`；sim 计数 ≈670 SET_FLAG+660 WAIT_FLAG/核/批。
- **A/B 证伪（#59）**：unit_flag true→false 使 FIXP 82%→44%（-38pp），wall-time
  **不降反升 +3%**（1186→1224us）——FIXP 是并行/双发射后台管道，占比≠阻塞。

### D. scalar↔MTE1 装载链串行化（37% 停滞）——**与基线差距的主病灶**
- **证据**：v3.5b `aic_scalar_mte1_stall_time=402us`；`aic_mte1_wait_ratio=0.66`。
- **基线对照（#62）**：torch 同形状同流量模式下 stall=**0us**——零 stall 装载链在
  同硬件存在，O5 可达；wall 差 139us 与 stall 差 402us 同量级（部分被并行掩盖）。

### E. AIV 搬运依赖链（wait 0.9）
- **证据**：`aiv_mte2_wait_ratio=0.9`、`aiv_mte3_wait_ratio=0.9`——AIV 的搬运
  多数时间在等依赖（softmax 行块流水深度不足/事件链串行）。

### F. GM 总流量（聚合 ≈1TB/s ≈ HBM 60-65%）
- **证据**：每核读 35MB+写 26.4MB；S 区 fp32 双程（L0C→GM 52.4MB + GM→UB 26.4MB）
  是大头。目前非第一瓶颈，但优化 B/C 后可能成为下一个。

## 5. 优化方案规划

| ID | 方案 | 目标瓶颈 | 预期收益 | 风险/依赖 | 状态 |
|---|---|---|---|---|---|
| O1 | **K/V 的 L1 组内复用**：kv 头 tile 跨其 G 个 q 头留驻 B 槽 0（跳过重复 GM→L1；GQA 且 qNBlockTile==1 时启用） | B（K 冗余部分） | K/V 装载 ÷G；MTE2 双降；S=128 GQA 收益最大 | fork 驻留模式 + 事件收支严格推演（#54）；MHA/打包 GQA 零行为差异 | ✅ **已实施（#54，待验证）** |
| O2 | 事件收敛 | ~~C~~ | ~~fixpipe 占比下降~~ **FIXP 已证伪非瓶颈（#59），预期收益大幅下调** | 每处删除需独立验证 | **降级/搁置** |
| O3 | host 开销削减 | ~~A~~ | ~~若 A 坐实 >2ms~~ **host 仅 0.23ms，非主要瓶颈（#60 证伪）** | — | **降级/搁置** |
| O4 | S 区 fp32→fp16 中转 | F | S 读写流量减半 | 精度评估（S 量级 ≤128×scale，fp16 可表示范围够，需 ULP 测试） | **提议** |
| O5 | **装载链深流水/预取提前**（解 scalar-MTE1 stall——v3.5b 实测 402us；**基线 torch=0us 证明可达**，#62） | D/E | 内核侧最大单项收益；消 stall 后有望越过 963us 基线 | 引擎预取深度/事件链改动，风险中 | **升为首选内核优化** |
| O6 | KernelScale 分段指标（MetricsProfStart/Stop 插桩） | 全部 | 阶段级归因（QK/softmax/PV/divout 各占多少） | API 语义需用户查官方文档（api-doc 规则） | **待 API 确认** |
| O7 | **调度均衡**：尾块余数摊平（B 不整除核数时余数 +1 分散到前 R 块） | 负载均衡 | 块间时长不均衡 41.7%→~2% | 纯 kernel 批区间算术，host 零改动 | ✅ 已实施（#54）。**注意**：03:15 的 3.7% 不均衡是 B=1000（整除）测得——旧方案在 B=1000 也无尾块，O7 效果尚需 B=1024 类不整除形状的 prof 确认 |

## 6. 指标收集计划（去 -g 构建，逐 bundle 隔离运行）

```bash
# ① 全量 bundle（用户指定四指标；Roofline/MemoryDetail 各自绑定 Default）
AIC_METRICS=Roofline,Occupancy,MemoryDetail,Source bash perf/profile/run_test.sh

# ② 若 ① 分析器再崩（#51 前科），逐项隔离定位：
AIC_METRICS=MemoryDetail bash perf/profile/run_test.sh   # MTE1/MTE2 活跃带宽→瓶颈 B/D
AIC_METRICS=Occupancy   bash perf/profile/run_test.sh   # 核间负载均衡
AIC_METRICS=Roofline    bash perf/profile/run_test.sh   # 计算/访存判定图
AIC_METRICS=Source      bash perf/profile/run_test.sh   # 代码热点图（无 -g = 无调用栈，
                                                         # 热点图本身可用；调用栈需 -g）
# ③ host 开销量化（bench 对照 prof 1.23ms）
python perf/bench/bench2_gqa.py --device 6 --nheads 8 --kv-heads 4 \
  --headdim 128 --seqlen 128 --batch 1024 --warmup 10 --repeat 50
```

注意：Source 的**代码调用栈**需要 -g（已随去 -g 重编放弃，代码归因由 sim 承担）；
#51 崩溃嫌疑 = Source+-g 组合（9.3 万 addr2line 关系），去 -g 后该组合不复存在。
采集完成后逐项回填 §3 数据档案与 §4 证据，更新 §1 版本行。

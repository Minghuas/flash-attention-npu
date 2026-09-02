# msopprof 使用指南（从官方文档提取）

> 来源：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/devaids/optool/docs/zh/user_guide/msopprof_user_guide.md
> 提取日期：2026-09-02。供 SplitB 性能分析团队长期参考。

---

## 1. 工具定位

MindStudio Ops Profiler（msOpProf）= 算子调优工具。采集和分析运行在 AI 处理器上算子的关键性能指标，定位软/硬件性能瓶颈。

两种运行模式：
- **上板**：`msopprof` —— 真机采集
- **仿真**：`msopprof simulator` —— 模拟器逐指令分析

## 2. 环境要求

- 配置好 CANN 环境变量
- MindStudio Insight 单独安装（可视化用）
- **输出目录不能含软链接**（`soft link is not supported`）
- 性能数据采集建议 ≤5min，推荐内存 ≥20GB
- 同一 Device 只能同时跑一个采集任务
- **CTRL+C 两段式**：第一次按停止算子执行、仍用已有数据生成结果文件；
  第二次按则直接退出不生成
- 未指定 `--output` 时，需保证上一级目录**其他用户不可写**（/tmp 类目录会被拒）

## 3. 命令行参数（关键项）

### 筛选链（逐层过滤）

```
--launch-skip-before-match N   → 跳过前 N 个算子
--mstx                         → 只采集 mstxRangeStartA/End 范围内的算子
--kernel-name <pattern>        → 只采集名称匹配的算子
--aic-metrics <list>           → 选择性能指标采集项
--launch-count N               → 最多采集 N 个算子
--kill=on                      → 达到 launch-count 后自动停止程序
```

### --kernel-name 规则

- 支持通配符 `*`，字符限制：`A-Z a-z 0-9 _ *`
- **未指定时：只采集程序运行过程中调度的第一个算子**（官方原文确认）
- 多个模式用逗号分隔

**⚠ 对本项目的影响**：`fa_test.py --test-torch` 里 `transpose().contiguous()×3`
在 `npu_fusion_attention` 之前先调度了若干搬运/转置核——若不指定 `--kernel-name`，
profiler 抓到的是**第一个转置核**而非 attention 核。
**实测（09-02，910B4）**：torch_npu 基线 S=128 单核完成 fusion，真实内核名 =
`FlashAttentionScore_<hash>_<id>_mix_aic`。
**内核名发现技巧**：用一个必然不匹配的 `--kernel-name` 跑一次，msopprof 会在
日志逐条打印 `Kernel <真实名> skipped: not selected via --kernel-name`，
等于免费枚举全部被调度的内核名。
注意 glob **大小写敏感**：`*attention*` 匹配不到 `FlashAttentionScore`，
须用 `*FlashAttention*`。

### --launch-skip-before-match / --kill=on 配合

- `--launch-skip-before-match N`：跳过前 N 个调度的算子，之后才开始采集
- `--kill=on` + `--launch-count N`：实际采集数达到 N 自动停止程序

### --aic-metrics 选项

| 选项 | 说明 | 绑定 |
|---|---|---|
| **Default** | ArithmeticUtilization, L2Cache, Memory, MemoryL0, MemoryUB, PipeUtilization, ResourceConflictRatio | 默认启用 |
| **KernelScale** | 指定代码段范围采集（需 MetricsProfStart/Stop 插桩） | 需显式开启 |
| **Roofline** | Roofline 瓶颈分析图 | 绑定 Default |
| **TimelineDetail** | 指令流水图 + 算子代码热点图 | 需 -g 编译 |
| **PipeTimeline** | Pipe 流水图 | 仅 Atlas 350 |
| **Occupancy** | 核间负载分析图（Core Occupancy）：各物理单核耗时/吞吐量/Cache 命中率 | Atlas A2/A3/350 |
| **MemoryDetail** | 内存负载分析展示 MTE 各通路**活跃带宽**；不开则不显示 Cube 侧 MTE1/MTE2 通路带宽 | 绑定 Default |
| **BasicInfo** | 仅 OpBasicInfo | — |
| **Source** | 算子代码热点图 | 需 -g 编译 |
| **PcSampling** | SIMT stall 信息 | 仅 Atlas 350 |

### sim 模式参数

```
msopprof simulator
  --kernel-name=<pattern>
  --soc-version=<chip>      # 如 Ascend910B4
  --timeout=<seconds>
  --output=<dir>
  <application>
```

## 4. 输出文件

目录结构：`OPPROF_{timestamp}_{random}/`

| 文件 | 内容 |
|---|---|
| `OpBasicInfo.csv` | 算子名、类型、Task Duration(us)、Block Dim 等 |
| `PipeUtilization.csv` | 各管道（Cube/Vector/MTE1-3/FIXP/SCALAR）耗时与占比 |
| `ArithmeticUtilization.csv` | FLOPS、指令数、fp16/int8 占比 |
| `Memory.csv` | GM/L1/UB 带宽与数据量 |
| `L2Cache.csv` | L2 命中率（读/写/总） |
| `MemoryL0.csv` | L0 级数据 |
| `MemoryUB.csv` | UB 级数据 |
| `ResourceConflictRatio.csv` | 管道等待/冲突比例 |
| `visualize_data.bin` | MindStudio Insight 可视化数据 |
| `trace.json` | sim 模式：Chrome/MindStudio 时间线 |
| `dump/` | 设备原始数据 + 内核二进制 |

## 5. Roofline 判定规则（重要）

| 算子性能百分比 | Bound 类型 | 细分 |
|---|---|---|
| >80% | **Compute Bound** 或 **Memory Bound** | 按所在区域 |
| <80% | **Latency Bound** | pipeline ratio <80% → `latency bound:pipeline caused` |
| | | pipeline ratio ≥80% 且 max=compute(cube/vec/scalar) → `latency bound:compute caused` |
| | | pipeline ratio ≥80% 且 max=memory(MTE1/2/3) → `latency bound:memory caused` |

### 可视化视图注意事项（官方原文）

- **-g 依赖**：Source 代码热点图、Cache 热力图跳转源码均需算子带 `-g` 编译；
  无 `-g` 则不展示热点图、不调用 llvm-symbolizer。带调试信息的二进制注意权限。
- **Cache 热力图**：不适用于 Atlas 推理系列；MC2/LCCL 算子不支持。
- **Pipe 流水图**：基于采样实现，最终只展示 **6 个核**的数据（与开核数无关）；
  MarkStamp 打点数据丢失时降低打点数目/密度。
- **L2 命中率口径**：MindStudio Insight **时间线页与详情页数值有差异**
  （官方有对比表，比较时须同口径）。
- **MemoryDetail 与 active_bw 的关系**：Cube 侧 `aic_mte1/mte2_active_bw`
  只有开 MemoryDetail 才显示——**#61 口径疑问（10.8 vs 56.8GB/s）的部分答案：
  先确认对比值是否同为 MemoryDetail 开启口径**。
- **通算流水图 trace.json**：可拖入 Chrome `chrome://tracing`（W 放大/S 缩小/A 左移/D 右移）。

## 6. 已知坑（本项目实测经验）

| 坑 | 症状 | 解法 |
|---|---|---|
| 输出目录含软链 | `soft link is not supported` | `cd -P` 解析物理路径 |
| -g + Source 组合 | 分析器 core dump（9.3 万 addr2line） | 去 -g 或逐指标隔离 |
| /tmp 目录权限 | `writable by any other users` | 用项目内目录 |
| kernel-name 含不支持字符 | `invalid kernel name` | 只用 `A-Za-z0-9_*` |
| 多模式逗号分隔含逗号 | 同上 | 用单模式或合法字符 |
| CANN 内置算子名不匹配 | `Profiling data parse failed` | 不匹配的 --kernel-name 试跑，从 skip 日志枚举真名 |
| glob 大小写敏感 | `*attention*` 匹配不到 `FlashAttentionScore` | 用 `*FlashAttention*` |
| 采集服务偶发故障 | `Get profiling data failed` 连续失败 | 重启设备或等待恢复 |

## 7. 本项目常用命令速查

```bash
# 我们的 SplitB（prof + 标准指标）
bash perf/profile/run_test.sh

# 我们的 SplitB（sim 模拟器）
bash perf/profile/run_test.sh sim

# torch_npu baseline（fa_test.py 内部切 npu_fusion_attention）
bash perf/profile/run_test.sh --test-torch

# 追加高级指标
AIC_METRICS=Occupancy,MemoryDetail bash perf/profile/run_test.sh

# 自定义 kernel 名
KERNEL_NAME='*MyKernel*' bash perf/profile/run_test.sh

# 透传 fa_test.py 参数
bash perf/profile/run_test.sh --batch 64 --seqlen 96 --nheads 8 --kv-heads 4
```

# 开发日志（二）

> 2026-08-31 起用本文件记录；此前全部历史见 `perf/devlog.md`（#1-#47，已封存不再追加，
> 期间条目插入位置较乱属历史遗留——检索时建议按编号 grep 而非按位置阅读）。
> 记录纪律不变：每个 bug/根因/修复/决策当轮记录（用户要求，#17 起延续）。

## 当前状态快照（开卷有益）

- **代码形态 v3.3**：批间错位流水（v2，4 个 mode2 flag：qk/sm/pv/do）+ 通用 Pingpong
  引擎 fork（splitb_bm_pingpong.hpp，自 catlass submodule v1.6.1，独立类 + presetEvents
  + AOperandBSHD 打包装载）+ BSHD 布局适配（Q 行距 H·D、K^T/V ldm=Hkv·D）。
  输入布局铁律：**Q/K/V = [B,S,H,D]（BSHD）**。
- **正确性**：MHA/GQA（打包 qN·Sq≤128，守卫 Sq%16==0）/bf16/D64/LSE/单核/多核全绿；
  pytest splitb 21/22（残 1/2097152 单点 = 既有环境性）。
- **性能**（多核，vs 旧路径 DISABLE A/B）：MHA(H8/D64 s32/64) 1.2~1.8x；GQA(H32/kv4/s32)
  1.27~2.42x；vs baseline 0.14~0.43x——S5 目标 ~3x（b1024 下 s32/s64 同耗时 = 开销主导）。
- **#47 清理**（2026-08-31，devlog.md 末条）：divout ② 冗余 Set/Wait 对删除、kernel 事件
  预算收敛（CUBE {0-3}×2、VEC MTE3_V{0-3}/MTE3_MTE2{0,2}）、过时注释/命名清理——
  **[编译验证中]**，验收 = 全量回归 + bench 抽点零漂移（对照 bench_v3_h8d64_mc.csv
  b1024/s32≈3.19ms）。
- **下一步**：S4（causal/SWA mask 穿透 softmax + softcap 上板，features 硬闸门逐项放开）
  → S5（性能）→ S6（默认启用收尾）。
- **硬规则**（记忆库）：csrc/catlass 是 submodule 禁改（定制 fork 到内核目录）；
  API 语义非 100% 确定必须请用户查文档；bench/测试我自跑（设备 6，FA2 conda +
  cann-9.0.0 env），编译由用户执行。

## 日志条目

**#65**｜**"host 3ms"结案：bench2.py 的 `out, *_ =` 把裸张量沿 batch 维解包，每次调用创建 B=1024 个 view（2026-09-02，九轮二分定案）**：
**现象**：bench2.py fa_ms 3.6-4.2ms vs kernel 1.49ms → 反推 host ~2.7ms；但
host_decompose async=0.4ms（#64），干净进程 event 仅 1.9ms。同一 device 无争用。
**二分链**（/tmp/bisect_bench~bisect9.py）：排除 warmup/repeat 数量、转置副本
（--bench-like）、张量创建方式、device 写法、NPU Event、GC（禁用仍慢）、
预触发；2×2 组合锁定 fa_fwd 函数层 → 六变体锁定 **`out, *_ = CALL` 解包语句**
（B 解包=3.36 / C 返回整个结果=0.43）→ 分段计时发现分开写全快。
**根因**：`FlashAttnFunc.forward` 返回 `out if not return_softmax else (...)`
——flash_attn_func 返回**裸张量**而非 tuple。`out, *_ = tensor` 触发 Python 沿
batch 维迭代：out=batch-0 切片 [S,H,D]，`_`=1023 个 view 列表，~1-3ms/次
（B 线性）。且 fa_fwd 返回的"out"一直是错的 batch-0 切片（计时循环碰巧丢弃）。
torch 侧 npu_fusion_attention 返回 tuple，同写法无害——这就是只有我们 fa 列
"host 巨大"的原因。旧包同返回裸张量，其 0.7ms 里也含较小份解包开销。
**修复**：bench2.py / bench2_gqa.py 的 fa_fwd 改 `return flash_attn_func(...)`
（附注释防回归）。
**修正后 E2E（device4, 20/100）**：MHA fa 1.840 vs torch 1.294（0.70x）；
GQA kvH2 fa 1.457 vs torch 1.054（0.72x）。E2E 组成：GQA kernel 1.10+host 0.36
vs torch 0.96+0.09。**结论：瓶颈回到 kernel 侧 O5（402us stall）+ host 0.36ms**
（workspace 缓存可再省，次优先）。

**#64**｜**host 裁决定案：host=0.4ms（#60 平反），4.238ms 为 bench 进程状态效应；allocator 嫌疑（2026-09-02，用户跑裁决实验）**：
host_decompose（device4）：
| 形状 | async | event | sync | thru |
|---|---|---|---|---|
| MHA kvH8 | **0.40** | 1.86-1.98 | 1.87-1.90 | 1.63-1.66 |
| GQA kvH2 | 0.44-0.65 | 1.44-1.47 | 1.45-1.47 | 1.20 |
thru≈kernel+0.15（1.49/1.10）✓ host 被流水隐藏。**async=0.4ms ⇒ host 非瓶颈**，
与 #60 一致；**4.238ms 在干净进程不可复现**（同 event 口径仅 1.9ms）。
no_grad 无关：bench2/fa_test 均未加 no_grad，但 randn requires_grad=False，
cProfile 证实走 forward_no_grad，grad/nograd async 相同。
**新嫌疑（唯一剩下的解释域）**：bench2.py 进程内存状态——fa 计时前持有 6 个大张量
（0.8GB BSHD + 0.8GB 转置副本），我们每调用 321MB churn（out 268+LSE 4+ws 53）
可能掉出 caching allocator 快路径 → 驱动级 malloc/free（~ms/次）。旧路径/torch
同环境干净（0.7/0.33ms）⇒ 排除设备争用，指向分配器尺寸类差异。
**复现实验**：host_decompose.py 新增 `--bench-like`（先建 3 份转置副本存活再计时）；
+重跑 bench2.py 确认 4.238 本身可复现（可换 --device 6）。若坐实 allocator：
修复方向 = workspace 按形状缓存复用（省 53MB churn）或 out/LSE 分配策略调整。

**#63**｜**O3 重开：用户 bench 反推 host ~2.7-3ms，与 #60 的 0.39ms 正面冲突（2026-09-02，用户提供四组数据）**：
用户对比（bench2.py device4 / kernel 来自 device6 prof）：
| 路径 | E2E fa_ms | kernel | 反推 host |
|---|---|---|---|
| 我们 v3.5b | **4.238** | ~1.49（MHA！bench2.py 无 kv-heads=8） | **~2.75ms** |
| 旧 FAInfer | 6.847 | 6.14 | 0.70ms |
| torch 基线 | 1.296 | ~0.96 | ~0.33ms |
注意：用户扣减用的 1.1ms 是 **GQA kvH2** 内核，bench2.py 实为 **MHA**（v3.5a MHA=1.49ms）——
修正后 host ≈ 2.75ms，量级不变。**旧路径同 bench 同 device 仅 0.7ms** 是最强内部证据。
**与 #60 冲突**：#60 四法（async 0.389 / event 1.418 / 吞吐 1.202）当时测得 host 仅 0.23-0.39ms。
混杂变量四个：device 6↔4、GQA↔MHA、false↔true 构建、时间点。静态排查（本轮）：
host C++ 全链（tiling 填充/53MB workspace 走 caching allocator/4KB tiling H2D/
rtGetC2cCtrlAddr/launch）均为 µs 级；wrapper 无 sync/item/cpu；但 bench2.py **未加
no_grad**（flash_attn_func 走 autograd Function.apply）。另：删除 splitb_host.cpp 全部
dbgEnv 语句（5 printf 块+set_debugFlag+SPLITB_BUILD_TAG，用户要求）。
**裁决实验**：`perf/profile/host_decompose.py`——四口径（async/event/syncwall/thru）
× {grad, no_grad} × {MHA, GQA}，device 4 复现 bench 条件。async≈2.7ms ⇒ host 真瓶颈
（--profile cProfile 定位）；async≈0.4ms ⇒ 4.2ms 来自设备时间线空泡（enqueue 延迟）。
待用户跑。**O3 暂时重开为高优先**（若坐实，收益 ~2.7ms > O5 的 0.4ms）。

**#62**｜**CANN 基线对比：torch 963us vs 我们 1102us（+14.4%），差距定位到 scalar-MTE1 stall（2026-09-02，用户采数我分析）**：
torch 基线剖析成功（内核名 `FlashAttentionScore_<hash>_mix_aic`，前情：`*attention*` 大小写不匹配→`*FlashAttention*` 命中；skip 日志可枚举内核名，已入 msopprof_guide.md）。
同形状 B1024/S128/H8/kvH2/D128、20 核、同 mix 类型、同指标 bundle，逐项对比（#61 vs 03:16 基线）：
- **时间**：963.2 vs 1101.6us（**+14.4%**）；aic_time 946.8 vs 1068.8us。
- **流量我们反超**：GM 读 **33.3 vs 52.7MB/核**（-37%）——反推 torch 无 GQA 组内 K/V
  复用（13.1Q+26.2KV+13.1P=52.4≈实测），我们的 O1 驻留（13.1+6.6+13.1=32.8≈实测）
  实打实省了 19.4MB/核；L2 命中 76.2% vs 69.8%。**总流量 -18% 却更慢 → 差距不在访存**。
- **决定性差异 = scalar_mte1_stall：torch 0us vs 我们 402us**。wall 差 139us 与 stall
  差 402us 同量级（部分被并行管道掩盖）。→ **O5 被基线证明可行**：同硬件、同流量
  模式下存在零 stall 装载链。
- **FIXP 证伪终结**：torch FIXP **96%**、scalar **98%**，均高于我们（79%/51%）却更快
  ——"占比高≠瓶颈"的官方级证据，#59 结论加固。
- torch cube_wait 0.96 > 我们 0.76（其 cube 更饥饿）、MTE2 65%（供数压力更大）——
  其快不靠 cube 喂饱。
- **active_bw 口径澄清（部分关闭 #61 遗留疑问）**：同 bundle 下 mte1_active_bw 两边可比
  （233.9 vs 236.0GB/s，几乎一致）；mte2_active_bw **双双为 0**——该指标 AIC 侧不可用，
  MTE2 实际带宽看 main_mem_read_bw（52.2 vs 28.8GB/s）。
- 写流量双方同为 ~53MB/核（S 区 fp32 L0C→GM 往返两边都做）→ O4（fp16 中转）对双方
  等效有效，排 O5 之后。
**结论**：O5 消 402us stall = 越过基线的主路径；O4 次之。

**#61**｜**unit_flag=true 回归数据：1102us（历史最优），MTE1 stall 显降（2026-09-02）**：
用户切回 true 后重采（B1024/S128/H8/kvH2/D128，同 #56 形状）。**kernel 1102us**——
比 #56 true 基线（1186us）快 **7%**，比 false（1224us）快 **10%**。
FIXP 78%（与 true 的 82% 接近，确认 true 模式）；**MTE1 stall 392us**（比 #56 的
502us 降 22%——比 true/false 两个基线都好）。其余指标（MTE2 57%、cube 28%、
vec 47%、GM 流量）与 #56 一致。改进来源待确认：#57 的 FIX_M 修复在 true 模式下
编译为空（if constexpr），代码应与 #56 等效——7% 差值可能为环境因素（L2/温度/
运行间方差）。**用户确认是否有其他改动**。

**#60**｜**#60**｜**O3 host 开销调查：~3ms 假设被推翻，实际仅 ~0.4ms（2026-09-01）**：
四种计时法交叉验证（B1024/S128/H8/kvH2/D128，unit_flag=false 构建）：
| 方法 | 值 | 含义 |
|---|---|---|
| NPU Event（bench 同款） | **1.418ms** | 端到端（GPU 时间线含 launch 间隙） |
| wall-clock（含 sync） | 1.427ms | 同步等待 kernel 完成 |
| **纯 Python 返回（异步）** | **0.389ms** | **host 侧真实开销** |
| 连续 30 次（吞吐模式） | **1.202ms/call** | host 与 kernel 完全流水 |

kernel = 1186us（prof）。分解：**host 开销 = 1.418 − 1.186 = 仅 0.232ms**；
吞吐模式 1.202ms ≈ kernel 1.186ms（host 已被流水隐藏）。
cProfile 进一步分解：Python wrapper ~0.25ms + C++ fwd ~0.9ms（含 launch，与
kernel 重叠后净增 wall 仅 0.23ms）。
**推翻原因**：此前 ~3ms 推导基于旧代码/旧形状的 bench 数据（H1/B1000 用 bench2.py
测得 3.2-4.0ms）。当前代码 host 已大幅缩减（ENABLE_DEBUG 宏去除 printf 分支、
多核默认化等），且小形状 bench2.py 的额外 Python 层可能放大了旧读数。
**O3 降级为低优先**。当前优化优先级修正为：①O5 内核装载链（scalar-MTE1 stall
502us = 42%，头号病灶）；②内核计算饱和（cube 仅 23-26%）；③O3 host（仅 0.23ms）。
H1 形状同步验证：sync=0.481ms、async=0.394ms → host ≈ 0.31ms、kernel ≈ 0.09ms。

**#59b**｜**#59b**｜**FIXP 占比机理解释（用户推断 + 代码数据三方印证，2026-09-01）**：
用户指出：①unit_flag=true 时 fixpipe 内部完成「搬运+信号」协调（mmad unit flag /
copyL0CToGm 0b11），FIXP 自身周期变多；②false 时改由 scalar 发射 M_FIX/FIX_M
软件事件对协调，FIXP 周期降、scalar 负载升；③瓶颈本不在 FIXP ⇒ 降其周期零收益。
数据三方印证：FIXP 1379→541us（-61%）、scalar +52us、**stall +141us**（3 倍于
scalar 忙碌增幅）、wall +35us。精化：false 变慢的直接代价大头不是 scalar 多干活
而是**软件事件 wait 落在 MTE1 装载关键路径上拉长依赖链**——与 Roofline
latency-bound 判定一致：任何加长依赖链的改动都直接伤 wall-time（O5 设计原则）。

**#59**｜**unit_flag A/B 定案：FIXP 假阳性瓶颈证伪，true 保留（2026-09-01，用户 09:28 数据）**：
B1024/S128/H8/kvH2/D128（GQA 驻留）unit_flag true vs false 对照：Task 1186 vs
1224us（**true 快 3%**）；**fixpipe 82%→44%（-38pp）但 wall-time 无改善**——
FIXP 是并行/双发射后台管道，"占比高"≠阻塞，**"unit flag 削减 FIXP"假设正式证伪**
（此前已按 api-doc 规则标注为待验证假说，现以数据结案）。false 的代价：scalar
+52us（软件事件对）、MTE1 stall +141us（事件链串行化）→ 总时长 +3%。
**决策：保留 unit_flag=true**。瓶颈图景修正（按 wall-time 影响）：①scalar-MTE1
stall 502us/42%（装载链延迟，Roofline latency-bound 一致）——**头号内核病灶**；
②MTE2 65%（数据搬运）；③cube 23-26%（计算远未饱和）；④host ~3ms（端到端大头）。
O2（事件收敛）重新评估：fixpipe 非瓶颈 ⇒ 预期收益大幅下调；O5（装载链深流水/
预取提前）升为内核侧首选；O3（host）仍为最高 ROI。

**#58**｜**#58**｜**unit_flag=false 修复验证全绿（2026-09-01，我自跑）**：
#57 的 FIX_M 条件预置/排水修复经编译验证：5 组正确性全 PASS（MHA 经典例——上次挂死
用例 8/8、GQA 驻留 G=4、MHA S128、打包 GQA、GQA 驻留多核；每组 timeout 防挂）。**
unit_flag=false 可安全采集性能数据**——下一步用户跑同形状（B1024/S128/H8/kvH2/D128）
prof 与 #56 的 true 基线（1186us/MTE2 62%/FIXP 82%）A/B。

**#57**｜**#57**｜**unit_flag=false 挂死：#47 清理时误删 FIX_M 预置（2026-09-01）**：
用户切 ENABLE_UNIT_FLAG=false 后 MHA 经典例（B2/H8/Sq9）卡死。根因：#47 事件预算
收敛时以"unit-flag 模式 FIX_M 不使用"为由删除了 FIX_M 预置/排水——当时正确（true
模式确实不用），但 false 模式引擎入口 `WaitFlag<FIX_M>(ID0)` + 末尾
`SetFlag<FIX_M>(ID0)`——**首次 Wait 无票 → 死锁**。修复：kernel 预置/排水加
`if constexpr (!BlockMmadQK::ENABLE_UNIT_FLAG)` 条件对（与模板参数同步切换），
收支 1:1（预置 1 Set、排水 1 Wait）。教训：**编译期开关影响事件面时，预置/排水
必须跟着条件化——清理时只看了一种模式的事件需求**。

**#56**｜**#56**｜**GQA 驻留（O1）首个有效 prof + unit_flag=true 基线（用户 07:32，2026-09-01）**：
形状 B1024/S128/**H8/kvH2**/D128（G=4，驻留路径激活，B=1024 不整除 20 核——O7 也同时
被测到）。**kernel 1186us**（vs MHA 同形状 1491us = GQA 天然减少 K/V 装载量）。
**关键指标**：MTE2 **62%**（vs MHA 92%↓30pp——O1 驻留 + G=4 省装载的组合效果）；
FIXP 82%、cube 26%、vec 44%（**vec 从 MHA 的 30% 升到 44%**——K/V 装载减少后 VEC
有更多可干）；scalar-MTE1 stall 502us（43%——仍为最大停顿项）；Occupancy 4.3%
（B=1024 不整除 20=51×20=1020 余 4，O7 摊平生效：旧方案单一尾块应产生 ~7%
不均衡，实测仅 4.3%——**O7 效果首次实证**）。
GM 流量：GM_to_L1 10.9MB（vs MHA 形状 52.4MB↓79%——O1 驻留贡献）；
L0C_to_GM 17.5MB（S+OTmp 写出，计算量决定）；read 28.5MB+write 26.4MB。
**此数据将作为 unit_flag=false A/B 实验的 true 基线**（用户即将跑 false 对照）。
追踪文档 §1 新增 v3.5a-gqa 版本行、§3 数据档案已更新。

**#55**｜**#55**｜**v3.5a 效果追踪（用户 03:15 prof，2026-09-01）**：
用户重采 prof（Roofline+Occupancy+MemoryDetail，shape=B1000/S128/**H8/kvH8**/D128
= MHA——fa_test 默认 kvH=8）：kernel 1491us、MTE2 92%（1352us）、fixpipe 90%、
**scalar-MTE1 stall 915us（62%）**、cube 20%、vec 30%、Occupancy 3.7%。
**Roofline 官方判定 = latency-bound(compute)**——主瓶颈是**依赖链延迟**而非带宽，
与 scalar-MTE1 stall 62% 互相印证：装载链（GM→L1→L0→cube）串行化是核心病灶。
三条结论：①**O7**：本 shape B=1000 整除无尾块，3.7% 不均衡**不能**证明 O7（旧方案
同样无尾）——需 B=1024 类不整除形状复测；②**O1 未测到**：kvH=8 是 MHA，驻留路径
未激活——需 --kv-heads 4 复测才能给 GQA 驻留下结论；③**host 固定开销 ~3ms 实锤**
（H1 kernel 168.7us vs bench 3.2-4.0ms；H8 1.49ms vs 3.7ms；三设备一致非争用；
workspace 分配仅 0.1ms 已排除）——嫌疑 wrapper torch 分发链/H2D 拷贝/路由检查，
待 cProfile host 侧剖析。追踪文档 perf/analysis/splitb_performance.md §1/§3/§4/§5
已更新（v3.5a 版本行 + 数据档案 + 瓶颈 A 量化 + O7 测量口径修正）。

**#54**｜**#54**｜**O7 + O1 + ENABLE_DEBUG 三件套实施（2026-08-31，待编译验证）**：
①**O7 批区间摊平**（kernel 侧）：batchStart/End 由 ceil 均分+尾块裁剪改为余数摊平
（base=B/coreNum、前 R 核 +1）——1024/20 核从 19×52+36（41.7% 不均衡）变 4×52+16×51
（~2%）。host 零改动（splitFactorSize 保留每核上限语义）。
②**ENABLE_DEBUG 编译宏**：mha_fwd_splitb.cpp 全部 16 处 `if (debugFlag) printf` 机械
改写为 SB_DEBUG_PRINTF（宏开=原语义，宏关=空展开零标量开销）；setup.py 读
FLASH_ATTN_ENABLE_DEBUG=1 注入 -DENABLE_DEBUG。默认构建即无调试开销。
③**O1 K/V L1 组内驻留**（GQA 且 qNBlockTile==1 时启用）：fork 新增 setBResident/
waitBResidentPrev/clearBResident——kv 头 K（段1）/V（段3）固定驻留 B 槽 0 跨组内 G 个
q 头复用，省 G-1 次 GM→L1 装载（prof 实证 MTE2 70% 忙的第一削减项）。**事件收支
（单 bit set 语义下严格推演）**：每组 waits=1(load)+(G-1)(prev)=G、sets=G(每头末 tile)
→ 组内净零；MTE2_MTE1 每头 re-arm Set+首 tile Wait 成对；launch 末组末头 +1 由 drain
Wait(ID2) 消费；**槽 1（ID3）驻留模式全程不用 → 预置/排水按 residentQK 条件跳过**
（孤儿预置=跨 launch 死锁，#44.53g）。QK/PV 跨段共享 ID2：QK 组余量由 PV 首组 wait
消费、PV 组余量由下一批 QK 首组消费——批次交错下全局闭合。MHA/打包 GQA 走原扁平
循环（零行为差异）。打包 GQA 的 V 冗余（同组 2 tile 共享 V）未覆盖，留扩展。
**[需 NPU 验证]**：编译 → 正确性全家桶（MHA 逐位不变 + GQA 驻留路径 H8/kvH4/S128
重点 + 打包 GQA 回归）→ prof 复测（MTE2 占用应显著下降、kernel 时长应缩短、Occupancy
不均衡 41.7%→~2%）。

**#53**｜**#53**｜**性能分析文档建立 + 指标收集计划（2026-08-31，用户提供 msopprof 指标文档）**：
新建 perf/analysis/splitb_performance.md（版本迭代记录/测试方法学/数据档案/瓶颈清单/
优化方案规划/指标收集计划六节，持续维护）。关键澄清（官方文档）：Default = 7 项标准
CSV（ArithmeticUtilization/L2Cache/Memory/MemoryL0/MemoryUB/PipeUtilization/
ResourceConflictRatio——普通 prof 默认即产出，09:48 数据已覆盖）；MemoryDetail 绑定
Default 且增 MTE1/MTE2 活跃带宽；Roofline 绑定 Default（计算/访存判定图）；Occupancy
核间负载均衡；Source 需 -g（用户已去 -g 重编，归因走 sim）；TimelineDetail 需"使用
前准备"配置且仅 PyTorch 单算子场景；KernelScale 可分段采集（MetricsProfStart/Stop
插桩，API 语义待查文档，列 O6）。收集计划 6 条指令（Default/MemoryDetail/Roofline/
Occupancy/host 量化 bench/可选 TimelineDetail），逐 bundle 隔离防分析器崩溃。
优化菜单 O1-O6 入文档（O1 K/V L1 组内复用为首选）。

**#52**｜**#52**｜**SplitB 首个真机 prof 完整画像（2026-08-31，09:48 数据）**：
形状 B1024/S128/H8/kvH4/D128、多核 20 块、kernel **1228.6us**。管道占用（每核）：
CUBE 核：**MTE2 70%**（GM→L1 装载第一忙！）、FIXP 78%（flag 操作）、CUBE 仅
**28-30%**、SCALAR 48%、MTE1 22%、**scalar 因 MTE1 停滞 444us（38%）**；VEC 核：
VECTOR 44%、MTE2 44%、SCALAR 49%、MTE3 22%。冲突：aic_cube_wait 0.7、
aic_mte1_wait 0.6、aiv_mte2/mte3_wait 0.9（搬运被依赖链拖死）。GM 流量：每核
读 35MB+写 26.4MB（61.4MB/1.18ms = 52GB/s/核，×20 核 ≈ 1TB/s 聚合 ≈ HBM 60-65%）；
GM_to_L1 52.4MB、GM_to_UB 26.4MB、L0C_to_GM 52.4MB；L2 读命中 75%。
**结论：SplitB 是数据搬运+同步开销主导，不是计算主导**（cube 仅 28-30%）；对照
旧路径 cube 4.4%/scalar 86.6% 已 7x 改善，但离"计算主导"还远。三大开销源：
①MTE2 的 Q/K/P/V 装载（K 在 GQA 组内**重复装载**——G=2 时 2 倍冗余、G=8 时 8 倍）；
②FIXP 的 flag 编排（sim 计数 670 flags/核/批）；③scalar-MTE1 装载链串行化。
**优化菜单（ROI 序）**：1)K/V 的 L1 组内复用（kv 头 tile 留 L1 槽跨其 G 个 q 头，
跳过重复 GM→L1——参考 BATCH_LESS_THAN_L1 的手工等价物；MHA 无冗余故无收益）；
2)事件削减（每调用 flag 对收敛）；3)S 区 fp32→fp16 中转（减半 S 流量，精度需评估）；
4)AIV 更深的行块预取（解 aiv_mte2_wait 0.9）。**待验证谜团**：bench H24/s128 测
9.5ms vs 本形状 1.23ms（3x 头差外推 3.7ms 仍差 2.5x）——嫌疑 host 侧 workspace
分配（H24/s128 每调用 ≈158MB）×20 核；实验：同形状跑 bench2_gqa 对照 1.23ms
量化 host 开销，若 >2ms 则 host 优化升为 S5 最高优先。

**#51**｜**#51**｜**msopprof prof 模式分析器崩溃（-g + aic-metrics=Source 组合，2026-08-31）**：
-g 重编译后 prof 运行：采集成功（日志确认选中 SplitB::FAInferSplitB 实例，kernel-name
过滤器 *SplitB* 生效；raw 数据已落盘 OPPROF_.../device6/<kernel>/0/{visualize_data.bin,
dump/}），但**分析阶段 core dump**（"Extract 13090 relations" + "Parse 93099
addr2line relations" 之后 Aborted）。嫌疑：--aic-metrics=Roofline,Occupancy,
MemoryDetail,Source 中 Source 指标与 -g 符号量（9.3 万条 addr2line）的组合。
处置：run_test.sh 的 prof 分支去掉默认 --aic-metrics（标准产出 PipeUtilization/
ArithmeticUtilization/ResourceConflictRatio 足够瓶颈归因），高级指标经 AIC_METRICS
env 按需追加（逐项试错）。崩溃轮的 visualize_data.bin 仍可直接导入 MindStudio。

**#50**｜**#50**｜**msopprof 首个有效 SplitB sim 数据分析（2026-08-31，我分析/用户提供数据）**：
用户澄清：仅 09:14 的 sim 运行有效（.so 09:09 编译=闸门放宽+多核翻转；默认
B1024/S128/H8/kvH4/D128 → SplitB 多核 20 块，core0-19 印证）；08:18/08:27 两运行
为旧 .so（旧路径 FAInfer），且 prof 模式需 -g 编译才有归因，弃用。trace 分析
（sim 功能级计时，非设备周期精确）：
①**窗口 ≈ 首个 batch 迭代**：span 142.6us/核（设备 9.5ms ÷ 51 批/核 ≈ 186us/批）；
20 核 span 完全一致（mean=max=142.6）→ 入口负载均衡良好。
②**窗口内管道占用**（资源槽计数，>100% 为双发射会计）：CUBE 核 CUBE 83.3%、
MTE2 84.9%、SCALAR 104.6%、FIXP 164%（flag 操作密集的旁证）；VEC 核 VECTOR 仅
22.3%、MTE3 159%、SCALAR 55.7%——batch0 的 QK/PV 背靠背让 cube 满、vec 等
qkReady（prologue 期无重叠对象，属预期；稳态重叠需中段 batch 窗口才可见）。
③**同步事件密度**：单窗口 40386 SET_FLAG + 39600 WAIT_FLAG + 19800 flow/60 核
≈ 670+660 flags/核/批——scalar 侧 flag 编排开销大（launch 地板/开销主导假说的
正证据之一）。④trace 无 f 事件（s 为 instant 标记），flow 间隙 stall 无法从本格式
提取。**下一步**：①kernel 编译加 -g → prof 模式拿设备真实管道占用/冲突（对照
旧路径基线 cube 4.4%/scalar 86.6%）；②sim 加大 timeout 覆盖 ≥2 批中段窗口，验证
批间重叠（QK(bo_t)∥softmax(bo_{t-1}) 是否兑现）。

**#49**｜**#49**｜**多核默认化 + 剖析工具链就位 + env 存在性判定陷阱（2026-08-31）**：
①**多核改为默认**（用户指令）：splitb_host.cpp 的 FLASH_ATTN_SPLITB_MULTI_CORE →
FLASH_ATTN_SPLITB_SINGLE_CORE（语义反转：默认 min(B,aicNum)，设 SINGLE_CORE=1 才单核）；
测试脚本同步（test_fa_splitb_multi_core→_single_core、test_splitb_stage_full.py
--multi-core→--single-core）。**[编译中/待编译]**——生效后 bench 无需再带 env。
②**剖析工具链**（perf/profile/）：fa_test.py 单次 launch（timeline 干净，MindStudio
可视化友好；用户修正：去掉我加的预热/重复/计时）+ run_test.sh 纯 msopprof 包装
（用户改写：--soc-version=Ascend910B4 --kernel-name="*FA*" --timeout=5，参数透传）。
③**env 存在性判定陷阱（重要）**：宿主侧 DISABLE/SINGLE_CORE 均为
`getenv()!=nullptr` 存在性判定——**赋 "0" 也算开启**！fa_test.py 初版 else 分支赋
"0" 会静默切到「旧路径+单核」，剖析对象全错。修复：关闭时 `os.environ.pop(...)`。
以后凡 env 开关：查宿主判定方式（存在性 vs 值）再写测试脚本。

**#48**｜**#48.1**｜**v3.4 闸门放宽验证：全绿 + 性能大幅改善（2026-08-31，我自跑）**：
正确性：H24/s64、s128、s96（非对齐尾块）×4 全 PASS（LSE 精确/OUT 良性残差）。
性能（H24/D128，多核，b1024，vs 旧路径 DISABLE A/B）：
| s | SplitB | 旧路径 | vs 旧 | baseline | vs baseline |
|---|--------|--------|-------|----------|-------------|
| 32 | 3.39 | 16.31 | **4.81x** | 1.16 | 0.34x |
| 64 | **3.35** | **16.73** | **4.99x** | 2.88 | **0.86x** ← 接近 baseline！ |
| 128 | 9.52 | 19.41 | **2.04x** | 3.64 | 0.38x |
s=64 从旧路径 0.18x → SplitB 0.86x（差 5.5x→差 1.16x）；s=32 与 s=64 同耗时
（开销主导优势在此体现）；s=128 开始 GM 流量显现但仍胜旧路径 2x。
全 batch 网格 b1~b1024 均 SplitB 胜旧路径 1.19~4.99x。
**剩余差距 = S5 主战场**：s=32 差 3x（launch 地板/固定开销）；s=128 差 2.6x（GM 流量）。

**#48**｜**闸门第二条放宽 + 闸门/多核/路由三个误诊澄清（2026-08-31，用户驱动）**：
用户发现 bench 全表 <1.0x 且 MULTI_CORE 开关"似乎无效"。三轮判别实验定案：
①**多核正常**——s=32 列（唯一 SplitB 生效列）单核→多核 7.7x（b1024）；
②**表 4/5 格子走的旧路径**——H24/D128 下闸门第二条（N2G×S1×S2×2≤128KB）把
s=64（192KB）及以上全部拒掉，DISABLE A/B 实证 s=64 全旧路径（17.5 vs 16.7ms 同值）
而 s=32 SplitB 活跃（4.57 vs 15.99ms = 3.5x 胜）；
③**闸门第二条是假约束**——它是参考 TilingB::IsCapable 为 BATCH_LESS_THAN_L1（L1
驻留 batch）设的预算，我们的 Pingpong 引擎逐 tile 装载（L1 用量与 S 无关），不吃它。
**修复**：删 shape_supported 的 N2G 条件（保留 max(Sq,Sk)≤128 + D≤128 硬约束）→
H24/s64、H8/s128、H24/s128 等进入 SplitB 覆盖。**[编译验证 + bench]** 待跑——
重点看 H24/s64 列（当前旧路径 17.5ms → 期望 SplitB 后大幅改善）。
另：Sq>128 分块结构已在但未验证（`curQSBlockNum>1` 的尾块几何/散射从未测过），
列为后续项（同 Sk>128 的 B 面 n 分块 + softmax 跨块状态机 = "第二形态"单独立项）。


# 测量方法学教训 —— "host 3ms" 幻影全案（devlog #60-#65，2026-09-02）

> 本文档记录一次为期两天的测量误导排查的**方法论**沉淀（结论已入 devlog2.md
> #60-#65；本文供后续任何性能数字争议时对照）。核心事件：bench2.py 报 fa 3.6-4.2ms、
> kernel 1.49ms，减法推得 "host ~2.7ms"，与直接测量（0.4ms）冲突——最终定位为
> bench 自身的 Python 解包 bug。

## 0. 全案一分钟版

| 环节 | 结论 |
|---|---|
| #60（09-01） | 四口径交叉验证：host = 0.23-0.39ms |
| #63（09-02） | 用户 bench 反推 ~2.7ms，与 #60 正面冲突 → 重开调查 |
| #64 | 裁决实验 async=0.4ms（#60 平反），但 bench 稳定复现 3.6-4.2 |
| #65 | 九轮二分 → **bench2.py `out, *_ =` 对裸张量解包**：沿 batch 维迭代创建 B=1024 个 view（1-3ms/次），且返回的 out 一直是 batch-0 错切片 |
| 修复 | 两 bench 的 fa_fwd 改为直接调用；E2E 真值：MHA 1.840 / GQA 1.457 |

## 1. 减法推导的隐含假设

`host = E2E − kernel` 成立要求 E2E 里**只有** host 和 kernel 两个成分。本次
E2E 里混入了第三成分（Python 解包），且恰好只在 bench 进程里出现。
**原则：任何用减法得到的成分，必须有独立的直接测量证实。**

## 2. 三口径计时（perf/profile/host_decompose.py）

| 口径 | 做法 | 含义 |
|---|---|---|
| async | fn() 返回即计时（每轮 sync 排空） | **host 真实开销** |
| event | record/fn/record/sync（bench 同款） | 设备时间线，含 enqueue 延迟空泡 |
| thru | 连续 N 次后一次 sync | host 可被流水隐藏时的吞吐下限 |

判读矩阵：
- async 大 ⇒ host 真瓶颈
- async 小 + event 大 ⇒ 看 thru：thru≈kernel ⇒ 时间线空泡；thru 也大 ⇒ kernel 真慢
- back-to-back 时 fn 返回变慢（本例 1.6ms ≈ kernel 时长）是分配器等流事件背压，
  **不是** host 计算量增加

## 3. 解包陷阱（本次根因）

- `flash_attn_func`（return_attn_probs=False）返回**裸张量**，不是 tuple
  （`return out if not return_softmax else (...)`）。
- `out, *_ = tensor` → Python 沿 batch 维迭代：out = batch-0 切片（**错值**），
  `_` = B−1 个 view 列表；B=1024 时 1-3ms/次，随 B 线性。
- `torch_npu.npu_fusion_attention` 返回 tuple，同写法无害——所以该坑只坑我们。
- 防御：`out = flash_attn_func(...)`；要解包先 `isinstance(r, tuple)`。
  已存记忆 fa-returns-bare-tensor。

## 4. 二分纪律（九轮的模板，可复用）

1. **先复刻再二分**：`--bench-like` 复刻目标进程状态，确认问题可复现
   （不可复现的"问题"先怀疑环境）。
2. **一次一个变量**；正交变量用 2×2 组合（本次：预触发×函数层）。
3. **排除环境**：npu-smi 查占用 + 同条件多次运行看稳定性。
4. **对解释域之外的假说直接实验证伪**（gc.disable() 一次毙掉 GC 假说），
   不要停留在推理。
5. **分段计时**：把可疑语句拆成多段分别计时——本次关键一步是"拆开写全快"，
   由此锁定"语句形式"而非"操作内容"。
6. **机制证实**：最终用 type(r) / len(_) / shape 把机制钉死，不停留在相关性。

## 5. 结论必须绑定测量口径

"3ms host" 假说两天内两次被推翻又复活，根源是每次引用的数字来自不同口径
（旧 bench vs 新 bench vs event vs async vs prof TaskDuration）。引用任何性能
数字时注明：**口径（async/event/thru/prof）、进程状态、device、构建**。

## 6. 附：占比 ≠ 瓶颈（同案并行教训，#59/#62）

torch 基线自己 FIXP 96%、scalar 98% 却比我们快——管道占比是"忙"不是"堵"。
判定瓶颈优先看 stall/wait 类指标与 A/B 实验，不看不参与阻塞的后台管道占比。

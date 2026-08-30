# SplitB 错误跟踪表（t1 起，随修复滚动更新）

> 用途：逐项跟踪 pytest / 冒烟发现的所有错误类别（用户要求，2026-08-25 设立）。
> 维护纪律：新错误先归类入表（或开新 Bug 行），修复后更新"状态"并链接 devlog 条目。
> 复现命令基型：
> `~/.conda/envs/FA2/bin/python debug/test_splitb_stage_full.py --batch B --heads H --kv-heads Hkv --sq SQ --sk SK --dim D --iters N [--multi-core] [--dump]`

## 总览

| Bug | 一句话 | 触发条件 | 状态 |
|---|---|---|---|
| ① | 多头 tile LSE tv 区重叠 | qnbt≥2、B≥3 | ✅ 已修（#44.49/#44.50，LSE_TV_FLOAT_OFFSET=512） |
| ② | S=Q(t+k)·K(t) 之 Sk≥96 案 | Sk>64、Hkv=1 时 41% | ✅ 同④根因已修（#44.51；残 17-34 个 ULP） |
| ③a | AIV 尾块（R%8≠0）整块坏 | Sq%16≠0 | ✅ 已修（#44.53e：LoadStats blockLen 整数截断，一行 RoundUp） |
| ③b | S=Q(t+k)·K(t) 之 Sk=40/80/112 案 | Sk∈{40,80,112} | ✅ 同④根因已修（#44.51；Sk=80 残罕见竞争） |
| ④ | 纯时序竞争、O 随机行垃圾(~9.8)+LSE 错 | Sq64 类 tile × ≥8 tile/批；B128 ~40%/launch；多核 20× 放大 | ✅ 已修（#44.51；残 640 个 1-ULP 确定性） |
| ⑤ | 确定性 ULP 残差（量化语义） | 大量形状（640/10/17-34 个元素，1-2 ULP，逐位不变） | ⏳ 待定夺（疑 CAST_NONE vs RINT） |
| ⑥ | Sk=80 罕见竞争残余 | 1/5 轮 69 错 max 0.44（b1 s0 h6） | ⏳ 待查 |

## Bug④/②/③b 统一根因（2026-08-26 #44.51，已修）

- 签名：**S(t) = Q(t+k)·K(t)，k∈{1,2}**，受害 tile 全部行精确匹配"Q 取自 t+k 头"
  （b0 t5: ref×7/6 = 0.0000 残差；b1 t5: ref×8/6），受害集合随时序漂移。
- 根因：BlockMmadQK l1A 无跨调用保护；loadQGM(MTE2 写) vs 上 tile copyL1ToL0A(MTE1 读)
  零事件。FAInfer 原版靠 KV 长循环天然隔离（qk_matmul.hpp 与 v3 逐字节一致=继承隐患）；
  SplitB tile 循环紧挨暴露。多核 B128 64×64 的 120× 放大 = 20 核同竞争。
- 修复：mha_fwd_splitb.cpp 段1 loadQGM 前 `Set/Wait<MTE1_MTE2>(EVENT_ID5)`（仿 l1B 惯例）。
- 修后战果：Sk=40: 4096→0；Bug④ 单/多核逐位一致 LSE 0 错 + O 640 个 1-ULP；
  Bug② 41%→LSE 全对 + O 17-34 个 1-2 ULP。

## Bug③a 当前签名（下一目标）

- **每个 AIV 行范围 R（AIV0=⌊Sq/2⌋，AIV1=Sq−⌊Sq/2⌋）的非整 8 尾块 [8⌊R/8⌋, R) 整块损坏**：
  | Sq | AIV0(R, 坏区) | AIV1(R, 坏区) |
  |---|---|---|
  | 31 | 15 → s8-14 全坏 | 16 → 净 |
  | 33 | 16 → 净 | 17 → s32 |
  | 17 | 8 → 净 | 9 → s16 |
  | 16/32/48 | 整除 → 全净 | 整除 → 全净 |
- O 与 LSE 同坏（112/115 同坏）→ softmax stats 级（max/sum）或 S 读坏，非 divout。
- 确定性（同 shape 两轮逐位同）→ 逻辑 bug 非；疑 splitb_softmax.hpp 行尾块路径
  （RowMax/RowSum 的 TAIL 掩码 / S 的 GM→UB 部分块搬运）。
- 另有满块内罕见散点错（Sq=33 的 s12/24/26 等）疑与 Bug⑥ 同族残余竞争。
- NaN/inf 出现在 Sq=17/31（尾块含未初始化数据 → sum=0/−inf 类）。

## 量化语义残差（⑤，待定夺）

- 三处确定性小残差：B128 64×64 的 640 个、Sk=112 的 10 个、GQA 128² 的 17-34 个，
  全部 1-2 fp16 ULP、逐位不变、跨 iter/核数一致。
- 假设：divout ③ Cast fp16 分支用 CAST_NONE（截断），ref 的 `.half()` 是 RNE → 边界值
  差 1 ULP；bf16 分支用了 CAST_RINT。待对照 FAInfer rescale_o 原版取整模式后统一。

## pytest 样例 ↔ Bug 对照（t1/t2 基线，待修后全量重跑）

| t2 失败样例 | 归属 |
|---|---|
| case2 (B128 64×64)、multi_core、kvc4、case12 | ④（已修） |
| case5 (H4Hkv1 128²)、case7/8 (GQA/96 类) | ②/③b（已修） |
| case33×47 类（若重跑仍败） | ③a（未修） |

> 注：t1/t2 采集自被覆盖的旧 .so；#44.51 修复后需全量重跑 pytest 刷新本表。

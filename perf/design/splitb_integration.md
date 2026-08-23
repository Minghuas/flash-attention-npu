# SplitB 模板集成设计方案（P2，v3 = catlass 路线）

> 状态：**v3 待评审**（2026-08-16，D7 路线切换后重写）
> 版本史：v1 两阶段简化（否决）→ v2 AscendC matmul 高阶 API 照搬（编译过、运行撞墙，
> 代码归档 [../archive/ascendc-matmul-paradigm-v2/](../archive/ascendc-matmul-paradigm-v2/)）→
> **v3 catlass 路线**。
> 阻塞根因与两套范式实证：[../README.md](../README.md) "当前问题记录"。
> 输入：[深度解读](../analysis/reference_splitb_deep_dive.md)·[P1.5 实测](../results/ANALYSIS.md)

---

## 1. 设计原则（不变项，继承 D5/D6 修订）

1. **算法层照搬参考**：核间只切 B 轴；S2 不切分（单遍 softmax、无 rescale 状态机）；
   任务粒度 = batch × 全部头；workspace 中转 + ping/pong；触发闸门严格照搬
   （`alignedS2≤128 && N2G×S1×S2×dtype≤128KB`）；小 B 不回落
2. **机制层用我方已验证范式**（D7 新增）：catlass `BlockMmadQK/PV` + `__DAV_C220_*` 显式
   双段 + `CrossCoreFlag` 同步——与现有 FAInfer、构建约定（auto-infer=false）、launch 机制
   完全一致，零新增基础设施约定
3. 独立 kernel 文件；底层模块（matmul 块/epilogue/flag）能复用则复用；dispatch 复用
   `FwdLaunchArgs`；每步可编译可测

## 2. 机制映射表（参考实现 → v3 catlass 对应物）

| 参考机制（bn2gs1s2_b.h） | v3 对应 | 差异说明 |
|---|---|---|
| `matmul::Matmul` + `IterateBatch(A批=N2×G, B批=N2)` | `BlockMmadQK/PV` **逐头调用**（复用 [qk_matmul.hpp](../../../csrc/ascend910/flash_attn_npu/qk_matmul.hpp)/[pv_matmul.hpp](../../../csrc/ascend910/flash_attn_npu/pv_matmul.hpp)，`loadQGM(gQ, layout, rowNum, qNBlockSize=1, qHeads)` 支持 BSND 头寻址） | 失 `BATCH_LESS_THAN_L1` L1 驻留：MHA 零损失（每头数据本就独立）；GQA 下 K/V 每组重载 G 次（带宽代价，无算法影响）。作后续优化项 |
| KFC 自动 cube/vec 分发 | `__DAV_C220_CUBE__`/`__DAV_C220_VEC__` 显式段 | 与现有构建/launch 一致 |
| matmul Wait + MTE 事件 | `CrossCoreFlag`：qkReady/softmaxReady/pvReady（复用 `Arch::CrossCoreFlag` + host 传 `fftsAddr`，照抄 [flash_api.cpp:734-736](../../../csrc/ascend910/flash_attn_npu/flash_api.cpp#L734) `rtGetC2cCtrlAddr`） | FAInfer 同款三 flag |
| `SoftmaxFlashV2` 单遍 | **新写 `splitb_softmax.hpp`（一次性 softmax：scale→[softcap]→[mask]→max→sub→exp→sum→cast→P 写 ws→stats 写 GM）**。不复用 online_softmax.hpp（2026-08-18 用户 FIXME #5 确认封装形态但 stats 须走 GM）：其 gm/gl 只在 UB 传递，四段批下后一 tile 的 softmax 会覆写前一 tile 的 stats——devlog #34。**封装形态仿 FAInfer**：init/operator() 内部按 qNBlockSize/subBlockNum 双 AIV 拆行 + SubCoreCompute | 行归约/事件/UB 偏移照抄 FAInfer 验证范式 |
| `Bmm2ResultDiv` 一次性除法 | **新写 `splitb_divout.hpp`（单遍 divout：OTmp→GM stats 读入→/sum→cast→O 散射→LSE）**。不复用 rescale_o.hpp：同理 stats 走 UB（devlog #34）；封装形态仿 rescale_o 的 operator()/SubCoreCompute（O 三分段散射、LSE 头主序逐 token gather/scatter 参数照抄） | |
| attenMask GM + 压缩模式偏移 | **FAInfer 原生 mask 机制**（2048×2048 triu 表 + `triUp/triDown` 窗口 + SWA 窗口参数，epilogue 内建三种 MASK_TYPE 变体） | 比 v2 的"mask 翻译层"更简：我方语义原生表达，无需翻译 |
| host `MatmulApiTiling`（TCubeTiling） | **删除**（catlass 自管 L1/L0，无需 cube tiling） | tiling 结构去 2 个 TCubeTiling 字段 |
| aiv 切片基数（每核 2 vector 子核独立任务） | aic 基数（`GetBlockIdx()/GetSubBlockNum()`，FAInfer 同款） | 切片数减半；host 公式 `usedAiv`→`usedAic`。后续独立优化项 |
| 3 槽 boIdx 流水（extraInfo[3] stagger） | 分层：**步 3 先串行**（每 batch 四段 + flag 链）→ 步 5 视性能数据加 `(PRE_LAUNCH+1)` 槽轮转 | 机制不变，深度分步 |

## 3. Kernel 结构规范（`mha_fwd_splitb.cpp` 重写）

> 2026-08-18 补充（devlog #35）：文件形态对齐 FAInfer 主体——namespace SplitB 内
> `SplitBKernel` 模板类（构造/operator()(FAIKernelParams const&)/runMainLoop/成员区）
> + 尾部模板入口 `FAInferSplitB` 组装类型并调用；runMainLoop 为单 batch 粒度（差异见 §3.2）。

### 3.1 模板与类型

```cpp
// 局部枚举（不 include mha_fwd_kvcache.cpp——FaiKenel 在其中定义，不能侵入）
namespace SplitB {
enum class MaskType : uint32_t { NO_MASK = 0, MASK_CAUSAL = 1, MASK_SWA = 2 };

template <typename DType, MaskType MASK_TYPE, bool HAS_SOFTCAP>
__global__ __aicore__ void FAInferSplitB(
    uint64_t fftsAddr, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask,
    GM_ADDR o, GM_ADDR lse, GM_ADDR workspace, GM_ADDR tiling);
}
```

kernel 内类型定义**照抄 FAInfer**（[mha_fwd_kvcache.cpp:1041-1101](../../../csrc/ascend910/flash_attn_npu/mha_fwd_kvcache.cpp#L1041)）：
`BlockMmadQK`（L1Tile `GemmShape<128,128,128>`）、`BlockMmadPV`（`GemmShape<128,128,256>`）、
`EpilogueOnlineSoftmax`（`EpilogueAtlasA2OnlineSoftmaxT<OUT_ONLY, float, HAS_SOFTCAP>`）、
`EpilogueRescaleO`、`Arch::CrossCoreFlag×3`、`Arch::Resource`——全部原样复用，
include `qk_matmul.hpp`/`pv_matmul.hpp`/`online_softmax.hpp`/`rescale_o.hpp`/`kernel_common.hpp`。

### 3.2 执行流（每核；2026-08-18 修正为参考的四段批结构，devlog #32；#38 段序重排 + flag 配对纠错）

```
coreIdx = GetBlockIdx()/GetSubBlockNum()                    // aic 基数（FAInfer 同款）
SetSyncBaseAddr(fftsAddr)；tiling GM→栈拷贝；init + 事件预置（照抄 FAInfer）
batchStart/End = coreIdx × splitFactorSize 的批区间

for boIdx in [batchStart, batchEnd):                       // 单层循环（核间 B 切分，照搬参考 Process）
  batchBuf = boIdx % 2                                     // ping/pong 按 boIdx 奇偶
  ┌ 段1 QK（CUBE）────────────────────────────────────────┐
  │  for (qSBlockIdx × qNBlockIdx): rowNum=qSBlockSize×qNBlockSize │ // 打包多头（FAInfer tile 模型）
  │    loadQGM + blockMmadQK(Q_tile → S ws tile 块)          │ // devlog #34：不再逐头循环
  │  SetFlag(qkReady)                                      │ // 每 batch 一次（广播双 AIV）
  └────────────────────────────────────────────────────────┘
  ┌ 段2 Softmax（VEC）────────────────────────────────────┐
  │  WaitFlag(qkReady)                                     │ // 每 batch 一次
  │  for tile: smEpilogue(gP,gS,gStats,...)（双 AIV 都调用，│
  │            operator() 内部按 qNBlockSize/subBlockNum 拆行）│
  │  SetFlag(softmaxReady)                                 │ // ⭐每 batch 一次（全部 tile 完成后）
  └────────────────────────────────────────────────────────┘
  ┌ 段3 PV（CUBE）────────────────────────────────────────┐
  │  WaitFlag(softmaxReady)                                │ // 批级一次（#39：同步责任上收到批级）
  │  for tile: blockMmadPV(P→OTmp)（WAIT_SOFTMAX=false    │ // policy 编译期关闭逐调用等待
  │           ——MmadAtlasA2FAIPVT 第三模板参数）            │ //   （#40；FAInfer 缺省 true 不变）
  │  SetFlag(pvReady)                                      │ // 每 batch 一次
  └────────────────────────────────────────────────────────┘
  ┌ 段4 DivO（VEC）───────────────────────────────────────┐
  │  WaitFlag(pvReady)；for tile: divoutEpilogue(O/sum+LSE) │ // 同款双 AIV 内部拆行
  └────────────────────────────────────────────────────────┘
```

要点（与参考的对应关系）：
- **单层 boIdx 循环 + 四段 per batch** = 参考 Process() 的结构（BMM1→Vec1→BMM2→Vec2）；
  差异仅在"BMM1/BMM2 一次 batch matmul"改为 catlass 段内 tile 循环（D7 机制适配）；
  tile 几何照抄 FAInfer runMainLoop（GetQNBlockTile/GetQSBlockTile，rowNum ≤ Q_TILE_CEIL）
- **段序即文本序即计算序**（QK→Softmax→PV→DivO，每段一个 #ifdef 区分执行核；#38）：
  CUBE 依序执行 段1→段3，VEC 依序执行 段2→段4
- **flag 全批粒度（#39 定案，#40 机制修正）**：qk/pv/softmax 均"每批 set 一次 ↔
  wait 一次"——与参考的批级事件语义一致；PV 的逐调用内部等待通过 dispatch policy
  编译期关闭：`MmadAtlasA2FAIPVT<PAGED, UNIT, WAIT_SOFTMAX=false>`（第三模板参数，
  缺省 true = FAInfer 原行为，不同实参 = 不同类型，零哨兵零运行时分支）
- **ping/pong 按 boIdx 奇偶**（照搬参考 workspace 复用），每批 tile 块布局：
  每 tile [S/P 原地区（P 为 fp16 视图，half 索引=2×float 索引）| OTmp | stats(2×128)]

### 3.2.1 workspace 复用分析（2026-08-18 review 补记，devlog #34）

**批内 tile 之间不能复用——这是批级 flag 的结构不变式，不是疏漏**：qkReady/pvReady
每批只 set 一次——段2 的 tile 0 要等段1 的 tile N-1 算完才能开始（softmax 无从
得知某个 tile 的 S 已就绪），段4 要等全部 PV 完成；因此批内所有 tile 的中间结果必须
并存：`perBatchF = nTilesPerBatch × perTileF` 由此而来，与 host 公式严格一致。
（softmaxReady 虽已改 tile 级配对——#38，但其消费者 PV 的输出 OTmp 仍要全批并存到
段4，故不改变 workspace 尺寸。）

已有的两层复用：
1. **P 原地覆写 S**（tile 内）：概率矩阵不另占空间（S 区按 fp32 计一份，P 为其 fp16 视图）
2. **ping/pong 批槽 ×2**（boIdx 奇偶）：每核串行处理 splitFactor 个 batch 全部复用这 2 槽；
   且 2 槽天然支撑**跨批 1 深度流水**——CUBE 的 QK(bo+1) 写 buf1 与 VEC 收尾 bo（读 buf0）
   并行。安全性：QK(bo+2) 重写 buf0 前，CUBE 必已等过 softmax(bo+1)，而 VEC 的 DIV(bo)
   在 set softmax(bo+1) 之前完成。1 槽会失去此流水，2 槽刚好

与参考的差距（S5 项）：参考是 **tile 级 3 槽轮转**（`BMM1(bo+2)∥Vec1(bo+1)∥BMM2(bo)`
三段同时各占一槽，flag 按任务粒度握手），workspace 从"批内全量"缩成"每核 3 个固定槽"
（FAInfer 同款：`curStackTileMod = stackSeqCount % (PRE_LAUNCH+1)`，固定 3×128×512 槽，
与 KV tile 数无关）。移植需把批级 flag 改任务级 flag + 槽轮转状态机——S5 性能步骤一并
解决 workspace 占用（当前触发闸门下封顶 ~60MB/20核：批内 ≤~1.5MB × 2 槽 × 核数）。

- 参考的 3 槽跨 batch 流水（BMM1(bo+2)∥Vec1(bo+1)∥BMM2(bo)）是性能项，S5 性能步再补；
  当前为串行批（正确性优先）
- VEC 分摊：每 tile 双 AIV 都调用 epilogue，内部拆行（FAInfer 范式，替换 #23/#32 的外层任务分摊）
### 3.3 与 FAInfer 的关系（复用边界）

| 复用（include + 实例化，零改动） | 新写 |
|---|---|
| BlockMmadQK/PV、EpilogueOnlineSoftmax、EpilogueRescaleO、EpilogueInitOut、CrossCoreFlag、Resource、kernel_common 常量、2048 triu mask 表机制 | kernel 入口 + boIdx×h 双层循环 + 任务定位算术（coreIdx×splitFactor）+ SplitBTilingData 读取 |

> 不 include `mha_fwd_kvcache.cpp`（侵入禁区）；`FaiKenel::MaskType` 在其中定义，
> 故 v3 在 SplitB 命名空间定义本地同值枚举。

## 4. Host 规范（`splitb_host.cpp` 改造）

继承归档版全部框架，三处变更：
1. **删** `MatmulApiTiling` 生成段（~50 行）与 tiling 结构的 `TCubeTiling` 字段
2. **核间切分 aic 基数**：`usedCores = min(B, GetCoreNumAic())`，`splitFactorSize = ceil(B/usedCores)`，
   `blockDim = usedCores`（launch 与 FAInfer 一致；CalcTschBlockDim 删除）
3. **补 fftsAddr**：`rtGetC2cCtrlAddr(&fftsAddr, &fftsLen)`（照抄 flash_api.cpp），传入
   `FwdLaunchArgs.fftsAddr`

workspace 公式照搬参考（T=float 口径：mm1区/stage1区/mm2区 各 ×2 ping-pong × coreNum，
512B 对齐），但槽位语义对齐 3.2（步 3 实现时校准 per-slot 尺寸 = rowNum×align(S2) 等）。

## 5. Tiling 结构（`splitb_tilingdata.h` 小改）

删 `TCubeTiling bmm1/bmm2TilingData`、`SoftMaxTiling`（FAInfer epilogue 不需要 host
softmax tiling——其 softmax 用 UB 内建参数；若步 3 验证发现需要再补）。其余字段保留。

## 6. Dispatch（`fwd_splitb_dispatch.hpp/_impl.hpp` 小改）

模板轴改 FAInfer 风格：`<DType, MASK_TYPE(NO/CAUSAL/SWA), HAS_SOFTCAP>`（去 HAS_ATTEN
布尔，三分 mask 枚举对齐现有 launch 树）；`FwdLaunchArgs` 复用不变（fftsAddr 字段现被真正
使用）；autogen TU 不变（重新生成一次即可）。

## 7. 测试与验收（继承 v2 §7）

- 正确性：`tests/test_flash_attn_npu_splitb.py` golden 网格（S∈{16..128}×B×{NO,CAUSAL,SWA}×
  {softcap}×{MHA,GQA}×{fp16,bf16}×{D64,128}）；`FORCE_SPLITB` 测非触发形状；现有测试全量回归
- 性能（bench repeat=100）：触发形状 ≥10x 旧路径（阶段目标）；与 baseline 同量级（目标）；
  seqlen≥256 无回归
- 每轮结果记录 `perf/results/`

## 8. 执行步骤（每步可编译可测，完成后勾选）

| # | 内容 | 出口判据 |
|---|---|---|
| S1 ✅ | 归档范式 A 代码 + 本设计 v3 | 评审通过 |
| S2 ✅ | host 改造（删 matmul tiling/aic 基数/fftsAddr）+ tiling 结构瘦身 + kernel 改 DAV 段空骨架（flag 链路）+ dispatch 模板轴调整 | ✅ 编译链接过；✅ FORCE_SPLITB 冒烟 6 模板分支（NO/causal/SWA/softcap/bf16/B=1024 满核）无崩溃无挂起——**机制底座（DAV+CrossCoreFlag+fftsAddr）全链路打通**；✅ 默认路径回归 192/193 绿（唯一失败为 NPU 计算精度既有问题，与 SplitB 无关，用户核查 perf/test/log1.txt）。期间修正：tiling getter 为 [host] 函数→设备侧直访公有字段；mha_fwd 早退返回 4 值对齐 |
| S3 ✅ | kernel 主体：tile 打包循环 + BlockMmadQK/PV + `splitb_softmax.hpp`/`splitb_divout.hpp`（FAInfer 式封装，stats 走 GM）；NO_MASK 全特性 | **2026-08-19 t10.log 12/12 PASS**（MHA b1-b20/尾块/multi-qS块/GQA×2/Sk32/D64/bf16；fp16 err 1-2e-4，bf16 1.2e-3）。遗留：多核路径待 FORCE_SPLITB 无 DEBUG 跑验证（DEBUG 强制单核）；b128/b1024 补回（devlog #43） |
| S4 | feature 全量：CAUSAL/SWA/softcap/GQA/bf16/D64/LSE | §7 golden 网格全绿 |
| S5 | 性能验证：sweet spot bench（b∈{128,512,1024}×s∈{64,128}×{causal,non-causal}，repeat=100）vs 旧路径 vs baseline；数据入 perf/results/ | ≥10x 旧路径；定位与 baseline 差距来源（若流水/槽位轮转有收益空间，实施并复测） |
| S6 | 默认开启（env 翻转）+ 全量回归 + 文档收尾（README 阶段表、设计文档勾选、记忆更新） | review |

## 9. 风险

| # | 风险 | 缓解 |
|---|---|---|
| R1 | online_softmax.hpp/rescale_o.hpp 的单 tile 退化路径与小 tile UB 布局不兼容（其 UB offset 常量按 128×512 预算设计） | S3 首项即验证；不兼容则裁剪精简版（保留算法，重排 UB offset——工作量可控，算法不变） |
| R2 | GQA K/V 重载带宽损失 | MHA 主场景零影响；触发闸门下 GQA 组数有限；量化后作后续优化项 |
| R3 | flag 串行每 (boIdx,h) 开销高于参考的按批摊销 | S5 数据说话；若显著，加 (PRE_LAUNCH+1) 槽轮转（机制现成） |
| R4 | 与并行分支合并冲突 | 新文件为主，flash_api.cpp 改动局部 |

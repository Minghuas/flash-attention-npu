# 归档：范式 A（AscendC matmul 高阶 API）SplitB 实现

> 归档日期：2026-08-16｜对应设计：[splitb_integration.md v2](../../design/splitb_integration.md)（D6 方案 A）
> 状态：**编译/链接通过（R1 ✅），运行时冒烟失败（aicore 507015），路线切换至 catlass（D7）**

## 这是什么

P3 步 1 按 v2 设计（照搬 CANN 参考实现的 matmul 栈）写就的 SplitB 骨架全套 6 文件。
机制特征：`matmul::Matmul` + `IterateBatch` + `BATCH_LESS_THAN_L1` + `REGIST_MATMUL_OBJ` +
`matmul_tiling::MatmulApiTiling`（host 侧 cube tiling 生成）——即 CANN KFC 隐式混合核范式。

## 为什么归档（阻塞根因，详见 [perf/README.md](../../README.md) "当前问题记录"）

CANN 存在两套**不互通**的混合核编程范式：
- 范式 1（显式双段）：`__DAV_C220_CUBE__/VEC__` 宏段 + 手写 cube 侧（catlass/intrinsics）+
  CrossCoreFlag 同步 + `--cce-auto-infer-kernel-type=false`（**我方构建全局约定**）
- 范式 2（KFC 隐式）：matmul 高阶 API 单线代码 + `KERNEL_TASK_TYPE_DEFAULT(MIX_AIC_1_2)` +
  **auto-infer 必须开启** + KFC workspace

范式 A 代码属范式 2，与我方构建体系（范式 1）冲突：kernel 类型错误 → `GetSysWorkSpacePtr`
拿到垃圾 → `REGIST_MATMUL_OBJ` 注册写坏内存 → 507015。补声明则因 auto-infer=false 编译失败。
（全 ops-transformer 无两范式混用先例；实证见 README。）

## 归档价值

1. **host tiling 公式翻译完整可用**（TilingB 全公式：基本块/核间切分/workspace）——v3 直接
   继承，仅删 `MatmulApiTiling` 段、切片基数 aiv→aic、补 fftsAddr
2. 触发条件 / dispatch 骨架 / autogen 接线完整可用——v3 继承（模板轴调整）
3. 若未来 CANN 工具链或我方构建允许 per-TU auto-infer + KFC workspace，此路线可复活
   （保留 `BATCH_LESS_THAN_L1` 完整语义）；`REGIST_MATMUL_OBJ` 的 workspace 是显式参数，
   可传自分配区作兜底

## 与 v3（catlass 路线）的差异

| 维度 | 本归档（范式 A） | v3（catlass，范式 1） |
|---|---|---|
| matmul | matmul::Matmul + IterateBatch（batch 维进 API） | BlockMmadQK/PV 逐头调用（复用 qk/pv_matmul.hpp） |
| L1 语义 | BATCH_LESS_THAN_L1 batch 驻留 | 每头独立 LoadL1（MHA 零损失；GQA K/V 每组重载） |
| 核类型 | KERNEL_TASK_TYPE_DEFAULT + auto-infer | DAV 宏段自证（与现有构建一致，零新增约定） |
| 同步 | KFC 内建 | CrossCoreFlag + fftsAddr（FAInfer 同款） |
| softmax | SoftmaxFlashV2（照搬） | online_softmax.hpp 单 KV-tile 退化路径（待验证） |
| 切片基数 | aiv（每核双任务） | aic（后续独立优化项） |

## 文件清单（与 csrc/ascend910/flash_attn_npu/ 下同名文件归档时快照一致）

| 文件 | v3 中的去向 |
|---|---|
| `mha_fwd_splitb.cpp` | **重写**（catlass 范式） |
| `splitb_host.cpp` | 大改：删 MatmulApiTiling 段、aic 基数、fftsAddr |
| `splitb_tilingdata.h` | 小改：删 TCubeTiling×2 字段 |
| `fwd_splitb_dispatch.hpp/_impl.hpp` | 小改：模板轴改 FAInfer 风格 mask 枚举 |
| `splitb_host.hpp` | 基本不动 |

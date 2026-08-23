# SplitB kernel 设计复审（跳出补丁循环）

> 日期：2026-08-18｜触发：用户指出连续多轮"测试→症状→补丁"循环，要求重新审视设计。
> 基线事实（t6-1/t6-2）：B=4 全 32 任务完成但 max_err=nan；B=1 挂（aicore timeout）。

## 与 FAInfer 的偏差清单（风险排序）

| # | 偏差 | FAInfer（生产验证） | 我的 SplitB | 风险 |
|---|---|---|---|---|
| R1 | **设备 printf 用量** | 零设备 printf | 20+ 处（含 pv_matmul.hpp 12 处探针） | **头号嫌疑**：printf 占 UB/同步资源，与我的固定偏移 UB 布局（0~57KB）可能冲突 → NaN/间歇挂死（症状模式吻合：间歇、时好时坏） |
| R2 | 流水结构 | 3 槽跨任务流水（QK(i+2)∥softmax(i+1)∥PV(i)） | 串行四段 + 每任务 flag 双 set | 已推理布尔语义自洽，但串行时序从未生产验证 |
| R3 | P 写回 pad 列裁剪 | epilogue 直写 GM 无 pad | DataCopyPad 3 参 + srcStride 跳列（#10/#11 缝补） | 中 |
| R4 | L0C→GM 后无 MTE3 完成等待即 set flag | 同款（PIPE_FIX 序） | 同款 | 低 |

## 复审结论

诊断探针（printf/转储）本身在污染观测——#24（stage 掩码反向）的教训在更大尺度重演：
每次"加探针→读症状→加补丁"都在改变被测系统。**决断：拆除全部设备 printf 回干净基线，
此后数值定位只用 STAGE 掩码（不改变执行路径）+ STAGE=3 转储（专用模式）**。

## 决断行动序列

1. ✅ 拆除全部设备 printf（kernel 10 处 + PV 12 处探针已删）
2. ⏳ 干净基线测试：DEBUG+STAGE=5 单核 → 看 max_err 是否仍有 NaN
3. 若仍 NaN → STAGE=3 转储判别（QK 输出 vs torch scores——此判别因早前挂死从未成功
   执行过，是缺失的关键证据）
4. 若 QK 正确 → 逐段二分 softmax（stats 转储）/PV/divout

## 遗留的纪律性教训（并入 devlog 经验法则）

- 调试探针必须有独立缓冲或专用门控，不得改主执行路径/污染主输出（#24/#30）
- 设备 printf 在 kernel 内不是"免费"的——生产代码零 printf 是有原因的

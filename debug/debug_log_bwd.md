# bwd softcap+alibi 调试日志（2026-08-03）

## 根因（已定位）：softcap Div 的 `/64` 在 N<64 时 repeat=0 → Div 跳过

**诊断方法**：one-hot dO（dO 只在 q=512 为 1）→ dV[k]=P[512,k]，逐 k 读出 bwd 重算的 P[512,:]。
结果：P[512,k] 对 k=0..511 全对（≈0），**唯独 P[512,512] 错（diff 0.387）**。

**根因**：尾-尾 tile（q=512 尾 Q-tile × k=512 尾 K-tile），s1ExtendSubGraph=1, s2ExtendAlign=16，N=16。
softcap 的 `Div<float,false>(..., (N)/64, ...)` → 16/64=0 → **Div repeat=0，完全跳过**。
score[512,512] = exp(-2t)+1-softcap（未除以 2*softcap），严重错误。
而 alibi[512,512]=0（近对角），使 k=512 成为 q=512 最关注的 key，P[512,512] 大，
单点错误被放大 → dV/dQ/dK 超容差。tiny-slope 时 P[512,512] 不突出，错误在容差内 → 通过。

**为何 softcap-only 不暴露**：无 alibi 时 P[512,512] 不特别大，Div-skip 的单点错误在容差内。
softcap+alibi 时 alibi 压低其他 key、凸显 k=512，放大该单点错误。

## 修复：Div repeat 用 ceil(N/64) 而非 floor(N/64)
```
old: (s1ExtendSubGraph*s2ExtendAlign)/64       // N<64 时 =0，Div 跳过
new: (s1ExtendSubGraph*s2ExtendAlign + 63)/64  // N<64 时 =1，Div 正常
```
两处都改（Epilogue1 行704、Epilogue2 行1433）。N≥64 时 ceil=floor，无变化。
多处理的 padding 元素（N 到 64 之间）后续被 alibi(columnNum=s2Extend)/softmax(s2Extend) 忽略，无害。

## 已排除的假设（红鲱鱼）
- columnNum padding、bwdWorkUb buffer 位置、Sq>Sk 对齐、softcap 因子(post/pre alibi savedS)、
  TMP 别名、operation order —— 均非 dV 根因。
- DumpTensor 在 CANN 8.5.0 不存在（FA2 用的 9.0.0 有）。

## 待验证
编译 Div ceil 修复后，跑 softcap+alibi 矩阵，预期全 ✅。

## 验证结果（2026-08-03）
- BSND softcap+alibi：**全部通过** ✓（Div ceil 修复有效）。
- softcap repeatTimes 检查：完整正确（Div 是唯一用 repeat 的，已修；其它用元素计数）。
- 3 个 varlen 失败（test_fa_varlen_bwd, GQA H=5/kvH=1 或 H=2/kvH=1, causal）：
  **预存 varlen GQA bwd bug**，与 softcap/alibi 无关（data_type1 softcap=0 无alibi 也崩）。
  GQA 5:1 崩溃(507057)，GQA 2:1 数值错。
  **处理决定：暂不排查**——当前代码非最新版，其它开发者可能已修复 varlen GQA 问题。
  待代码更新到最新版后重新跑测试，若仍失败再排查。

# Vec2 专项解读：归一化与输出的设计

> 对象：[flash_attention_score_bn2gs1s2_b.h](../../ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h) 的 `ProcessVec2()` / `Bmm2ResultDiv()` / `Bmm2DataCopyOut()`（源码已加【解读】注释）
> 配套：[Vec1 专项](reference_vec1_tiling.md)（含 §6.5 "为什么推迟归一"的问答）
> 日期：2026-08-16

---

## 1. 问题设定：Vec2 的输入与输出

进入 Vec2 时，本核负责的 boIdx（batch）在 GM workspace 里已有两份半成品：

```
mm2Res 区（fp32=T）:  O' = P·V   [N2×G, S1, D]   ← BMM2 输出，未归一
softmaxSum 区（fp32）: rowsum    [B, N2×G, S1, 8]（每行 8 元素 padding）
softmaxMax 区（fp32）:  rowmax   同上布局          ← LSE 用，Vec2 不消费
```

Vec2 要做的（对每头每行）：

```
O[i][:] = O'[i][:] / rowsum[i]     ← 行广播除（"rescale O"在 S2-不切分下的退化形态）
→ cast INPUT_T → 按输入布局步长写 attentionOut GM
```

**它存在的理由**（详见 Vec1 文档 §6.5）：① P 以 fp16 存 GM，未归一使最大元素恒为
1.0、值域贴 fp16 良好区间，归一推迟到 fp32 的 O 上做一次；② 模板家族（S2 切分的那
两个兄弟）算法上必须推迟；③ 除法次数 S1×D 而非 S1×S2。

## 2. 为什么 O' 必须经 GM 中转（cube→vector 的数据交换模式）

BMM2 的累加结果在 **L0C**（cube 子核的私有存储），vector 子核无法直接访问——L0C 没有
到 UB 的直接通路（只有 L0C→GM 的 FIX 管线搬出）。所以 cube 的每个 matmul 结果都走
**L0C →（FIX）→ GM workspace →（MTE2）→ UB** 的中转，vector 再从 GM 读回。这与我方
FAInfer 的 S/P/OTmp workspace 中转是同一模式（我们 v3 的 OTmp 同样如此）——是 A2
上 cube/vector 协作的固定代价，模板设计的 ping/pong 流水正是为了把这笔中转藏进重叠。

## 3. tiling：行块由 D 决定（与 Vec1 的关键差异）

Vec2 的循环流程速记（与 [Vec1 文档](reference_vec1_tiling.md) §2 同格式对照）：

**三个维度的切分**（处理对象是 O' = P·V，形状 `[N2×G, S1, D]`）：

- **D（列）**：不切分，但读宽对齐：读 `rows × dSizeAlign16`（凑 16 对齐满足 DataCopy
  长度约束，pad 列为垃圾但写出按 `dSize` 精确、无害）；`dOuterSize = 1`
- **S1（行）**：UB 预算 32KB，容纳 8192 个 FP32，有：
  `s1Vec2BaseSize = align16(8192 / alignedD) × (2/inputDtypeBytes)`（fp16/bf16 系数=1），
  再 `min(alignedS1)`；`s1Vec2OuterSize = ceil(S1 / s1Vec2BaseSize)`
- **N2×G**：每个头串行处理

**形成双重循环**：

```cpp
for biN2GoIdx = 0 .. N2×G-1:                    // 头（外层）—— 与 Vec1 同
    for s1oIdx = 0 .. s1Vec2OuterSize-1:        // S1Vec2 行块（内层，块大小由 D 决定）
        处理 tile = [s1Vec2BaseSize × D]  (≤ 8K 元素 / 32KB fp32)
            尾块行数 = s1Vec2BaseTailSize
            // 每块四步：⓪等上块MTE3排空 → ①O'块读回 → ②rowsum读回
            //          → ③行广播除 O'[i][:]/=rowsum[i] → ④cast+按布局写出
每 batch 总块数 = (N2×G) × s1Vec2OuterSize
```

**与 Vec1 的两个关键差异**：① 列维是 **D** 不是 S2——同一 S1 被两者切成**不同的块数**
（例 S2=16、D=128、S1=512：Vec1 按 256 行切 2 块，Vec2 按 64 行切 8 块），互不同步、
各自独立扫，靠 GM 的 OTmp/stats 交接；② 每块做的是**归一+写出**而非 softmax。

---

Vec1 的 tile 是 `[行 × S2]`（softmax 归约矩阵），Vec2 的 tile 是 `[行 × D]`（O 矩阵），
**行块大小因此不同**：

```
s1Vec2BaseSize = align16(8192 / alignedD) × (2/inputDtypeBytes)   ← 同守 8K 元素 / 32KB fp32 预算
                 再 min(alignedS1)
s1Vec2OuterSize = ceil(S1 / s1Vec2BaseSize)     ← 独立于 s1OuterSize！
```

数值走查：

| shape                | Vec1 块（s1BaseSize，由 S2 定） | Vec2 块（s1Vec2BaseSize，由 D 定） |
| -------------------- | ------------------------------- | ---------------------------------- |
| S2=64, D=128, S1=64  | 64                              | 64                                 |
| S2=16, D=128, S1=512 | 256（撞上限）                   | 64 →**s1Vec2Outer=8**       |
| S2=128, D=64, S1=64  | 64                              | 128                                |

两者**互不同步、各自独立扫**：Vec1 按 `(头, S1块)` 扫 S/P，Vec2 按 `(头, S1Vec2块)` 扫
O'——中间靠 GM 的 OTmp/stats 交接，不存在块级对应关系。这允许两级各自取最优块大小。

双层循环结构与 Vec1 同构：`for biN2GoIdx（头）× for s1oIdx（S1Vec2 行块）`。

## 4. 一个 tile 的执行流

```
进入 (biN2GoIdx, s1oIdx)：
 ⓪ SetFlag/WaitFlag(MTE3_MTE2)      —— 上块的 O 写出（MTE3）已排空，读入 buffer 可复用
 ① O' 块 GM→UB（stage1Ping/Pong 32KB，按 taskId 奇偶与 cube 写侧对齐）
    · D 是 16 对齐：DataCopy 直拷 rows×dSizeAlign16（读宽凑 16 对齐，pad 列垃圾无妨）
    · D 非 16 对齐：BMM2 输出是 NZ 格式 → NzToNd() 用 vcopy 转置回 ND（§5）
 ② rowsum GM→UB（pseTBuf 复用；读 rows×8 元素——含每行 padding）
 ③ Bmm2ResultDiv：O'[i][:] /= rowsum[i]（§6 指令级解读）
 ④ Bmm2DataCopyOut：Cast INPUT_T（vecOut 复用）→ DataCopyPad 按布局写 GM（§7）
循环尾：SetFlag/WaitFlag(MTE3_MTE2) 排空最后一块
```

## 5. NZ→ND 转置路径（D 非 16 对齐的边角）

matmul 的 C 输出在 D 非 16 对齐时以 **NZ 格式**落 GM（cube L0C 搬出的天然分形格式：
`[ceil(D/16), 行, 16]` 列分块排布）。`NzToNd()` 用 vcopy 指令把 `[D/16, rows×16+8]`
转回 `[rows, D]`——代码里那些 `offsetJ = 128×rows+64` 的手算偏移就是 NZ 分形的块间距。
**迁移提示**：我们 v1 的触发条件实际覆盖 D∈{64,128}（16 对齐），此路径可不移植；
参考的 `dSizeAlign16 == dSize` 分支选择（`bmm2` vs `bmm2Nz`）也据此简化。

## 6. Bmm2ResultDiv 指令级解读（行广播除）

```cpp
BinaryRepeatParams{
    src0BlkStride = 1,                    // O 行内元素连续
    src0RepStride = dSizeAlign16 / 8,      // 行推进（列数/8 个 block）
    src1BlkStride = 0,                    // ★ sum 行内广播：本 repeat 内恒取同一 sum 标量
    src1RepStride = 1,                    // ★ 行间推进：每行换下一个 sum（8 元素/行 padding 对齐 block）
    dstRepStride  = dSizeAlign16 / 8 };
Div(dst, src0, src1, count=64, repeatTimes=行数, params);
```

要点：

- `count=64`（repeatMaxSize = 256B/4B）：每条指令处理 64 个元素 = **一行内的一段**；
- `repeatTimes=行数`：**行串行语义**——repeat 步长负责行推进（对照
  [perf/devlog.md](../devlog.md) #18，我们曾在此错传 行数/8）；
- 内层按 `dSizeAlign16/64` 段循环 + 尾段 SetVecMask；
- 外层 `行数/255` 批（repeatTimes 是 uint8）；
- sum 的 GM 每行 8 元素 padding 在此兑现价值：`src1RepStride=1`（1 个 block）即取到
  下一行的 sum，无需跨步寻址；
- 家族兼容分支：`T=half` 的其它模板实例先 Cast sum 再除（B 模板 T=float 走直除主路径）。

## 7. Bmm2DataCopyOut：布局适配的写出

cast 后的 O 行（INPUT_T）从 UB 写 GM，三种输入布局的步长（`blockCount=行数， blockLen=D×sizeof(INPUT_T)`，`srcStride=0` 因 cast 输出行连续）：

| 布局                   | O 平面排布                          | 行步长 dstStride       |
| ---------------------- | ----------------------------------- | ---------------------- |
| **BNSD**（默认） | `[B, N2G, S1, D]`：头内行连续     | 0（直拷）              |
| **BSH/BSND**     | `[B, S1, N2G, D]`：token 内头打包 | `(N2G−1)×D×2B`    |
| **SBH**          | `[S1, B×N2G, D]`：token 主序     | `(B×N2G−1)×D×2B` |

**uint16 溢出 fallback**：`DataCopyParams.dstStride` 是 uint16（≤65535）。SBH 大
B×N2G 时步长超限 → 退化为逐行 `blockCount=1` 循环拷贝。**迁移提示**：我方 v1 固定
BSND，恒走 `(H−1)×D×2B` 一条步长（H×D×2 ≤ 65535 在 H≤128、D≤128 内安全，无 fallback
需要）。

## 8. UB 的分时复用（本模板的紧凑手法）

Vec2 没有新分配 buffer，全部**分时复用** Vec1 的区段：

| 区段                          | Vec1 期间               | Vec2 期间                                            |
| ----------------------------- | ----------------------- | ---------------------------------------------------- |
| stage1Ping/PongBuf（32KB×2） | S 块读入 + softmax 计算 | O' 块读入（taskId 奇偶 ping/pong，与 cube 写侧对应） |
| pseTBuf（16KB）               | PSE 偏置输入            | rowsum 读入（fp32）/ cast 输出（fp16 视图）          |
| vecOut（16KB）                | P 的 fp16 cast 输出     | rowsum 读入（pong 侧）/ 输出 cast                    |

合法性的来源：Vec1 与 Vec2 在流水线上隔着 BMM2（bo_{t-2} vs bo_{t-1}），且各自块首的
MTE3_MTE2 事件保证了上一用户的写出已排空。**UB 紧张时的设计范式，我方 v1 的 64KB
区段同理（softmax 与 divout 共用）**。

## 9. 与我方 splitb_divout.hpp 的映射对照

| 参考 Vec2                       | 我方 splitb_divout.hpp（v3）                         | 说明                                                                                                      |
| ------------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `for biN2GoIdx`（头）         | `for h`（kernel 主循环）                           | 同                                                                                                        |
| `for s1oIdx`（s1Vec2 行块）   | `for chunk`（64 行/块，divout 内部）               | 我们行块固定 64（=D=128 的参考值），未按 D 参数化——D=64 时参考用 128，我们仍是 64，正确性无差、性能微差 |
| O' GM→UB（DataCopy/NzToNd）    | 同（仅 ND 直拷路径）                                 | NZ 路径不移植（§5）                                                                                      |
| rowsum GM→UB（每行 8 padding） | stats 读入（同 8 padding 布局）                      | 同                                                                                                        |
| Bmm2ResultDiv                   | Div 广播除（同参数模式，行批 ≤16 分批）             | count 语义已对齐（devlog#18/#20）                                                                         |
| Bmm2DataCopyOut 三布局          | 仅 BSND 步长                                         | v1 固定 BSND                                                                                              |
| LSE                             | 参考：softmaxSum/Max 已是算子正式输出，Vec2 不算 LSE | 我们：divout 内 Ln+Add 合成 LSE 写 GM（我方接口契约要求 lse 返回）                                        |
| uint16 溢出 fallback            | 无                                                   | H×D×2 ≤ 65535 恒成立                                                                                   |

## 10. 细节清单（迁移对照用）

1. `s1Vec2BaseTailSize`：S1Vec2 尾块行数（与 Vec1 的 tail 独立计算）
2. 读宽 `dSizeAlign16` 凑 16 对齐（DataCopy 长度对齐要求），pad 列为垃圾但
   `blockLen=dSize` 精确写出、无害
3. stats 的每行 8 元素 padding 同时服务 Div 的 `src1RepStride=1` 和 DataCopy 对齐
4. taskId 奇偶的 mm2Res ping/pong：与 BMM2 写侧（IterateBmm2）严格配对
5. 每块首尾的 MTE3_MTE2 Set/Wait：读入 buffer 复用保护（上块写出排空）
6. `Vec2 不消费 rowmax`——max 只为 LSE 存在；我们的 divout 因要写 LSE 才读它

---

## 附：数据流总图（一个 batch 的 O 路径）

```
BMM2(cube): P(fp16,GM) × V(GM) ──L0C──FIX──► GM mm2Res: O' = P·V (fp32, 未归一)
                                               │  ← ping/pong by taskId
Vec2(vector): ①GM→UB(32KB tile [行×D])  ②rowsum GM→UB
              ③Div: O'[i][:] /= rowsum[i]        （fp32 上的一次除法/行）
              ④Cast fp16 → DataCopyPad → attentionOut GM（BSND 步长 (H−1)×D×2B）
LSE 路径:     rowmax(GM) + log(rowsum(GM)) ——（参考:host 合成 / 我方:divout 内合成）
```

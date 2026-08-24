# SplitB 开发问题日志（devlog）

> 用途：记录开发过程中遇到的**所有** bug、报错、踩坑——根因分析与修复方案，
> 避免遗漏、避免重犯。每解决一个问题即追加一条（用户要求，2026-08-16）。
> 格式：#序号｜分类｜现象｜根因｜修复｜预防
> 关联：设计文档 [design/splitb_integration.md](design/splitb_integration.md)；
> 归档（范式 A）[archive/ascendc-matmul-paradigm-v2/](archive/ascendc-matmul-paradigm-v2/)

---

## 一、范式 A 阶段（AscendC matmul 高阶 API，已归档）

**#1**｜编译｜`MatmulType/TPosition/CubeFormat/LayoutMode` 未声明
根因：CANN 9.0.0 中高阶 API 类型都在 `AscendC` 命名空间；参考实现入口
[flash_attention_score.cpp:35](../../ops-transformer/attention/flash_attention_score/op_kernel/flash_attention_score.cpp#L35)
有 `using namespace AscendC;`，头文件依赖它。
修复：kernel 文件 include 后补同款 using。
预防：照搬别人代码时，入口文件的 using 指令也是"代码"的一部分。

**#2**｜编译｜`reinterpret_cast` from `__gm__ uint8_t*` 不允许
根因：tikcpp 禁止 `__gm__` 地址空间指针 cast 到普通指针。参考实现的 tiling 是普通指针
因为 GE 框架的 `GET_TILING_DATA_WITH_STRUCT` 已做 GM→栈拷贝，我们没有框架。
修复：kernel 入口逐字节拷 tiling 到栈再按普通指针用。
预防：`__gm__` 指针只能解引用读，不能转普通指针。

**#3**｜链接｜`undefined symbol: launch_fwd_splitb`
根因：inline 函数定义在被 autogen TU 独占的 impl 头里，该 TU 不调用它 → 符号未发射。
修复：inline 定义移到轻量头 [fwd_dispatch.hpp](../../csrc/ascend910/flash_attn_npu/fwd_dispatch.hpp)，
调用方 TU 发射符号，impl 模板链接期解析（照 `launch_fwd` 成熟模式）。
预防：inline 函数放在**调用方能 include** 的头里。

**#4**｜运行时｜冒烟 aicore 507015（范式撞墙，触发路线切换 D7）
根因：CANN 两套混合核范式不互通（详见 [README.md](README.md) "当前问题记录"）。
范式 2（matmul 高阶 API）依赖 auto-infer + KFC，与我们全局 auto-infer=false 的范式 1
构建冲突 → kernel 类型错 → GetSysWorkSpacePtr 垃圾 → 注册写坏内存。
处置：归档范式 A，切 catlass（范式 1）。附：`KERNEL_TASK_TYPE_DEFAULT` 宏与 builtin
被 `__CCE_ENABLE_AUTO_INFER__` 守卫，auto-infer=false 时不存在。

---

## 二、S2 阶段（catlass 骨架）

**#5**｜编译｜tiling getter 在 kernel 里不可调（`[host]` function）
根因：我给 tiling 结构写的 getter 无 `__aicore__` 标注 → bisheng 归为 host 函数。
FAInfer 的 kernel 直访公有字段（`fATilingData->batch`），不用 getter。
修复：kernel 侧直访公有字段；getter 仅 host 侧用。
预防：设备代码访问的结构体，要么字段公有无 getter，要么 getter 标 `__aicore__`。

**#6**｜运行时（python）｜`flash_attn_func` 返回值解包错误
根因：接口返回形态随分支演化（dropout/ret-probs 合入后 fwd 返回 4 值；`flash_attn_func`
包装层默认只返回 `out` 单值）。我的 C++ 早退分支返回 2 值 ≠ 函数尾 4 值；测试脚本又解包错。
修复：C++ 分支返回 `{out, softmaxlse, p, rng_state}` 对齐函数尾；脚本按 `out = ...` 收单值。
预防：在函数中加早退分支时，核对整个函数所有 return 的形态一致。

**#7**｜测试方法｜环境变量 static 缓存不随 `importlib.reload` 重置
根因：C++ `env_enabled()` 的 `static bool` 是进程级缓存，reload Python 模块不影响 .so。
处置：删掉脚本里"同进程切换新旧路径"的对比，旧路径正确性由回归测试背书。
预防：同进程内无法切换 C++ 静态 env 开关，对比测试要分进程跑。

---

## 三、S3 阶段（kernel 主体）

**#8**｜编译｜`SBCeilDiv` 未声明（我 sed 全局替换把自己的定义也改名成 `SBSBCeilDiv`）
修复：改回正确名。预防：全局替换后 grep 校验定义处未被误伤。

**#9**｜编译｜`constexpr` 助手函数设备侧不可调
根因：未标 `__aicore__` 的函数被 bisheng 归为 `[host]`，设备代码禁止调用。
修复：加 `__aicore__ inline`（去 constexpr）。
预防：设备头文件里所有函数都标 `__aicore__ inline`（照 online_softmax.hpp 惯例）。

**#10**｜编译｜`DataCopyPad` 无匹配重载（UB→GM 方向）
根因：UB→GM 只有 **3 参**重载（dst, src, DataCopyParams）；`DataCopyPadParams`
（padding 语义）仅存在于 GM→UB 方向。参考实现是旧版 CANN，4 参能过。
修复：LSE/O/P 三处 UB→GM 写出去掉 padParams；P 写回补 `srcStride=(colsPad-cols)*2`
跳过 cast 输出的 pad 列。
预防：按方向查重载（CANN 9.0.0 的 kernel_operator_data_copy_intf_impl.h）。

**#11**｜编译｜`DataCopyPad` 模板 T 冲突（float vs half）
根因：UB→GM 重载要求两端同元素类型；P 写回用了 gS（fp32 视图）。
修复：softmax 增加 gP（fp16 视图）参数专司 P 写回。

**#12**｜编译｜catlass 引用参数不能绑临时值
根因：`loadQGM(uint32_t &singleGroupHeads, uint32_t &qHeads)` 与 PV operator 的
`uint32_t &nIdx/&nLoop/&blockSize` 是非常量引用；FAInfer 调用处都用具名变量。
修复：补 `singleHead/qHeadsP/pvNIdx/pvNLoop/pvBlockSize` 具名变量。

**#13**｜编译｜`Cast` 五参调用无匹配
根因：CANN 9.0.0 的 Cast 是六参 `(dst, src, roundMode, mask, repeatTime, params)`，
fp32→fp16 步长 `(1,1,4,8)`（dst 64 元素=4 block / src 64 元素=8 block）。
修复：照 [rescale_o.hpp:401-413](../../csrc/ascend910/flash_attn_npu/rescale_o.hpp#L401-L413)
可编译模式改写（含 bf16→CAST_RINT / fp16→CAST_NONE 分支镜像）。
预防：写向量指令前先在本仓库找同款已编译过的调用照抄。

**#14**｜编译｜`printf` 需 `AscendC::printf`
（用户发现）设备代码里用 `AscendC::printf`。

**#15**｜**运行时：VEC 握手不对称死锁（第一例挂死）**
现象：首个用例即挂；stage 1（纯握手骨架）通过、stage 2 也挂（当时误判）。
根因：我把 flag 握手（WaitFlag/SetFlag）包进了 h-owner 判断（`if (h >= hSubStart…)`），
两个 vector 子核对 flag 的**参与不对称**。CrossCoreFlag 是无任务标签的握手信号，
非 owner 子核在等"它的"任务的 flag 时可能消费到别人任务的 flag → 握手序列错乱 → 死锁。
判别证据：S2 骨架两子核对称参与全部握手 → 不挂。
修复：握手移出 owner 判断——所有子核对称执行（双 wait qk、双 set softmax、双 wait pv，
FAInfer 同款），计算工作仅 owner 做。见 kernel VEC 段注释。
预防：**flag 握手必须所有消费者对称参与；只有计算可以按 owner 分摊。**

**#16**｜测试方法｜设备 printf 在 kernel 挂死时不刷新
根因：AscendC::printf 缓冲到 kernel 结束才输出，挂死时全部不可见。
处置：改用**分阶段执行掩码**（`FLASH_ATTN_SPLITB_STAGE=1..5`，tiling 字段 + env 控制
kernel 只执行到第 N 段，stub 分支显式补 wait/set 保持握手平衡）+ host 侧 printf 链路 +
debug 时强制单核。stage 二分定位到 QK 内部。
预防：定位设备挂死用"执行掩码二分"，别依赖设备 printf。

**#17**｜**运行时：漏 BlockMmad.init + 硬件事件预置（stage 2 真根因）**
现象：stage 2（仅 QK 真算）挂死于 loadQGM/QK 内部。
根因：两个遗漏——① `blockMmadQK/PV.init(resource, nDyn, kDyn, …)` 未调用：L1/L0
buffer（l1ATensor/l1BTensor/l0A/l0B/l0C）从未从 resource 划分绑定（空指针），
`copyGmToL1A` 首写即挂；② FAInfer kernel 开头的**硬件事件预置块**（CUBE 侧 18 个
`SetFlag<M_MTE1/FIX_M/MTE1_MTE2 EVENT_ID0-7>` + VEC 侧 14 个 `MTE3_V/MTE3_MTE2/V_MTE2`）
未抄：catlass 块内部 ping-pong 事件的首次 WaitFlag 依赖预置的"已释放"初态。
修复：照抄 [mha_fwd_kvcache.cpp:180-235](../../csrc/ascend910/flash_attn_npu/mha_fwd_kvcache.cpp#L180-L235)
全部初始化（含 kDynNum/nDynNum/L1 布局计算公式）。
预防：**照搬调用序列时，kernel 入口的一次性初始化（init + 事件预置）是最易漏的部分**——
它们在主循环之外，注意力容易只放在循环体。照搬 checklist 应含：类型定义/对象构造/
**init**/**事件预置**/主循环/收尾。

**#18**｜**运行时：向量指令 count 语义错误（数值错，×2 处同类）**
现象：跑通但 err 0.27~2.0；恰似"前 8 行对、其余错"。
根因：**行串行指令**（repeat 步长负责行推进的那类）的 count=**行数**，我错传 `行数/8`：
- softmax 广播减 `Sub`（对照 FAInfer CalcExp :689 传 rowNumCurLoop）
- divout 广播除 `Div`（对照参考 Bmm2ResultDiv：repeatTimes=行数）
→ 每指令只处理 1/8 的行。
修复：两处 count 改回 `rowsCurRound`。
预防：**两种 count 语义并存，按指令类型区分**——
① 元素级指令（Muls/Exp/Cast）：count = CeilDiv(总元素, 64)；
② 行串行指令（广播 Sub/Div/行数组 Max/Add 等，特征是 repeatParams 带行步长）：count = 行数。
写每条指令前对照 FAInfer/参考的同款调用核对 count。

**#19**｜编译｜host debug printf 块位置过早（coreNum 等未定义）
修复：移到 workspace 计算之后。预防：插入打印时确认变量作用域。

**#20**｜运行时/方法｜count=64"挂死"系环境误判；行批 ≤16 保留为安全措施
现象：#18 修复（count 8→64）后首例挂死，曾判定 count=64 触发广播 repeat 上限。
**更正（2026-08-16 用户提供新证据）**：挂死实为 NPU 卡上残余进程占用（#21），
换卡后同二进制可跑。count=64 是否真有 ISA 问题**未被证实**。
处置：行批 ≤16 的分批实现**保留**（与 count=行数数学等价、严格落在 FAInfer
生产验证区间 ≤16，无代价纯安全）；经验法则 #4 的"≤16"降级为"建议"而非"必须"。
真正的教训：**报"回归"前先排除环境因素（换卡/清进程复测）**。

**#21**｜方法/环境｜**NPU 卡残余进程导致假性挂死（两次误导排障方向）**
现象：kernel"挂死"，换 device 卡号后同代码同二进制正常执行。
根因：此前挂死的测试进程残留在卡上，占用了设备/流，后续进程同步等待被卡死。
处置：挂死排障前先换卡或清理残留进程（ps 查 python/npucntl）复测一次再下结论。
本条已两次干扰判断（S3 首例"挂死"、#20"count 回归"均受其影响）。
预防：**任何"挂死"结论前：换 device 复测 + 查残留进程。**测试脚本可固定
`torch.npu.set_device(N)` 选空闲卡。

**#24**｜**工具 bug：stage 掩码语义写反——stage=0 一直跑空骨架（引发连环误判）**
现象：debug 输出出现 "PV-stub got-softmax" 而 stage=0 应为全真实。
根因：门控写成 `if (dbgStage >= N) 真算 else stub`——0 落进 else，QK/softmax/PV/divout
全被跳过。自 stage 设施引入后，**所有默认运行从未真正计算过**。
后果：①"err=0.2790/2.0691 逐位不变"（#22 悬案破案）= 未初始化 O buffer 的确定性垃圾值，
非计算错误；②此前所有数值分析作废；③ 骨架模式两子核仍发空队列 set → 竞态仍在
（B=4 骨架完成而 B=1 骨架挂的随机性即此）。
修复：① 门控改 `dbgStage == 0 || dbgStage >= N`（0/5=全真实，1=骨架，2/3/4 逐级）；
② 骨架模式 set 前补 32B 最小真 MTE3 写（消除 stage 工具自身的竞态）。
预防：**新增调试设施后，第一件事是用打印验证其默认行为**（stage=0 时打一行
"FULL"），调试工具自身的 bug 会污染全部后续观测。

**#23**｜**运行时：VEC 段重构为行对半分摊（间歇死锁根治方案）**
现象：同二进制间歇性首例挂死（换卡可复现/不复现），B=128 稳定挂——典型竞态。
根因分析：原设计"h-owner 分摊 + 所有子核对称 set flag"里，非 owner 子核对
softmaxReady 发**空 MTE3 队列**的 PIPE_MTE3 set（无任何真工作）——FAInfer 从不出现
此模式（其双子核各做行半的真实 softmax 后才 set）。flag 语义边角（空管道 set 与
wait 配对时序）引入竞态 → 间歇死锁 + 间歇数值错（PV 可能在 P 写完前通过 wait）。
修复：VEC 段重构——每任务双子核按行对半分摊真实工作（softmax/divout 逐行独立，
拆分零合并代价），两者都做真 MTE3 再 set；顺带 vec 吞吐翻倍。行区间=rowHalf
(align8(ceil(rows/subNum))) 起 rowNum 行。**经验法则 #2 细化：不仅握手要对称，
set flag 前的"真工作"也要对称——照搬范式时 flag 事件的完整结构（谁 set、set 前
有什么工作、几次 set）都必须一致。**

**#22**｜运行时（数值，追踪中）｜**疑似存在上游主导数值 bug（QK 写 S 阶段）**
现象：count 修复前后 B=4 err=0.2790 / B=1 err=2.0691 **逐位相同**——输入有种子、
kernel 确定性，不同 count 语义不可能同输出 → 要么二进制未更新（待复测），
要么错误在 Sub/Div 之前已定（QK 写 S / softmax 读 S）。
最大嫌疑：S/P 的 GM 行距用了 colsPad(=64)，FAInfer 固定 512（stackSeqTilePad
恒为 MAX_KV_STACK_LEN）；若 catlass LayoutC/L0C→GM 写出内部假设 pad ≥ L1TileN(128)，
pad=64 造成 S 写出地址错乱。
下一步：复测干净二进制；若 err 仍同 → 加 S 转储探针（stage3 后把首 S tile 拷到
O buffer，python 侧对比 torch scores）在 QK 写出边界二分数值错误。

---

**#25**｜测试脚本｜S 转储对比索引错误（scores 维序）
现象：dump_probe 报 "size of tensor a (32) must match b (8)"。
根因：torch_ref 返回的 scores 是 [B,H,S,S]（BNSD 语义），我按 [B,S,H,S] 索引了。
修复：`ref = scores[0, 0, :n, :Sk]`。
预防：多维 tensor 索引前打印 shape 核对（尤其 BNSD/BSND 混用时）。

**✅ 里程碑（2026-08-18）**：stage 掩码修复（#24）后，**stage=3（QK+softmax 真算，
单核）完整跑通**——8 任务 × 双子核全部正常、kernel 干净退出。QK 链路（含 #17 init）
与 softmax（行分摊 #23 + 行批 #18/#20）作为整体首次验证可运行。挂死收窄到：
**stage=0 全真实 + 多核（blockDim=4）** 组合。

---

**#26**｜运行时（数值）｜softmax 的 GM→UB 拷贝后缺 MTE2→V 事件同步
根因：① 处 `DataCopyPad`（MTE2 管道）后仅 `PipeBarrier<PIPE_V>` 就开始向量计算——
PipeBarrier 不跨管道，S 数据可能未就绪（数据竞态→数值错）。FAInfer 的 GetBmm1Result
在 GM→UB 后用 `SetFlag/WaitFlag<MTE2_V>` 对同步。
修复：① 后补 `SetFlag<HardEvent::MTE2_V>(3); WaitFlag<...>(3);`（相邻自配对，无需预置）。
预防：**跨管道数据依赖（MTE2→V / V→MTE3）必须用 Set/Wait 事件对，PipeBarrier 只管
本管道内序**。分阶段排查发现 PV 挂死后插桩时的伴生修复。

**#27**｜调试方法｜pv_matmul.hpp 事件级探针（SPLITB-DBG 标记，诊断后移除）
现象：stage4/5 单核下 cube 卡死在 blockMmadPV 内部（softmax flag 双子核均已 set）。
静态分析：PV 内部 FIX_M/M_MTE1/MTE1_MTE2 乒乓与 QK 交替共享 EVENT_ID，静态推演
全部自持平衡，无法定位。
处置：pv_matmul.hpp 的每个 Wait/Set/关键 copy 前插 `AscendC::printf`（12 处，
SPLITB-DBG 注释标记，**诊断完成后必须移除**——它污染 FAInfer 旧路径输出）。
首个 wait 无输出的探针 = 挂死点。

---

**✅ 里程碑（2026-08-18）**：MTE2_V 事件同步修复（#26）后，**stage=4（QK+softmax+PV 真算）
B=4 全 32 任务 PV DONE、kernel 完整退出、sync 正常返回**。PV 探针显示事件链全部通过
（EVT4→softmax→FIX_M→l1P 乒乓→l0AB→L0B→L0C→DONE）。err=0.2790 此时为垃圾值属预期
（stage4 无 divout，O 未写）。

**#28**｜运行时（追踪中）｜**同进程第二次 kernel launch 挂死（B=4 完成→B=1 零输出挂）**
现象：B=4（32 任务）完整跑完后，同进程第二个用例 B=1 launch 后设备侧零输出（连首条
printf 都无——挂死时缓冲不刷新，无法区分 preamble/任务循环）。
初步假设：跨 launch 的核状态残留（CrossCoreFlag 通道/硬件事件/ffts 状态未随 kernel 结束
复位；上一 kernel 的 flag 终态影响下一次握手）。注意 FAInfer 的旧路径同进程反复 launch
无此问题——差异必在我们 kernel 的收尾状态或初始化假设。
判别实验（各自独立进程）：① 仅 B=1 stage=4 → 过=二 launch 污染实锤；挂=B=1 特有。
② 仅 B=4 连跑两次同 shape → 第二次挂 = 与 shape 无关的二 launch 污染。
工具：测试脚本新增 FLASH_ATTN_SPLITB_CASES 用例过滤器。

---

**#29**｜**运行时：plain DataCopy 的 blockLen 单位错误（字节 vs 32B block）→ MTE 写越界崩溃**
现象：stage=5（+divout）2 个任务后 aicore 异常："The write address of the MTE instruction
is out of range"（vec/fixp error，MTE3/2 侧）。
根因：`DataCopyParams.blockLen` 在 **plain DataCopy** 中单位是 **32B block**，不是字节
（FAInfer online_softmax.hpp:630 传 `rows/8` 佐证）；**DataCopyPad** 的 blockLen 才是字节。
我三处按字节传：① divout OTmp 读入 blockLen=512（=512 blocks=16KB/行！64 行写 1MB 进
32KB UB → UB 越界崩）② softmax stats 写 256→8KB ③ divout stats 读同。
修复：三处除以 32（rowsCurRound/8、colsPad/8、rowNumRound/8）。
预防：**DataCopy=block 单位（值=字节/32），DataCopyPad=字节单位**——两套拷贝 API 的
params 语义不同，混用是重灾区；写拷贝先确认用的是哪个 API。

**#28 补充（2026-08-18 判别实验）**：B=1 单独跑 stage=4 仍挂（零设备输出）——
**非二 launch 污染**，是 stage≥4 的任务内问题且与 B 无关（B=4 能过的差异在时序运气）。
紧接 stage=5 B=4 崩溃暴露 #29（MTE 越界）——#29 的 UB 越界写同样可解释 stage=4 的
间歇挂死（越界写踩坏 UB 同步结构/printf 缓冲 → 表现随机）。#29 修复后需复测
stage=4/5 全矩阵确认。

---

**✅ 里程碑（2026-08-18）**：#29 blockLen 修复后 **stage=5 全真实（单核）完整跑通
32 任务，无挂无崩，python 拿到真实输出**——四段计算链路（QK→softmax→PV→divout）
作为整体首次端到端执行成功。

**#30**｜测试方法｜golden 全 NaN = S 转储探针污染 O（测试工件，非 kernel bug）
现象：stage=5 跑通后 max_err=nan（fp16/bf16 皆同）。
根因：dump 探针把 b0h0 的 O 区当转储目标（写 32 行原始分数），且该任务 divout 被跳过
→ O[b0][h0] = 原始分数 32 行 + 未初始化垃圾 32 行（垃圾位型含 NaN 模式）→
max over 全张量 = NaN。两用例同现 NaN 与 b0h0 污染完全吻合。
修复：dump 仅在 STAGE=3（dump 专用模式）激活；STAGE=5 全真实保持 O 干净。
预防：**调试探针不得污染主输出缓冲；探针目标要用独立缓冲或专用 stage 门控**。

---

**#31**｜方法/纪律｜**跳出"测试→症状→补丁"循环：设计复审 + 拆除诊断探针**
背景：用户指出连续多轮补丁循环（#15..#30），要求重新审视设计。
复审结论（详见 perf/analysis/design_review_kernel.md）：头号嫌疑=**设备 printf 自身**——
20+ 处探针（含 pv_matmul.hpp 12 处）占用 UB/同步资源，与我的固定偏移 UB 布局可能冲突，
NaN+间歇挂死的症状模式完全吻合；FAInfer 生产代码零设备 printf 是有原因的。
行动：拆除全部设备 printf（kernel 10 处 + PV 12 处），回干净基线；此后数值定位只用
STAGE 掩码（不改执行路径）+ STAGE=3 转储专用模式。
纪律：**诊断探针不得改主执行路径/污染主输出；每加一轮探针都在改变被测系统**（#24/#30/#31 同源）。

---

**#32**｜**设计纠错：三重循环→四段批结构（用户审查发现，重要）**
现象：用户指出 kernel 采用 boIdx×h×s1o 三重循环，与参考 Process() 的四段批结构不符，
也与设计文档 §1"任务粒度=batch×全部头"原则自相矛盾。
根因：catlass 无 batch matmul，我把头循环错误提升为外层（flag 按任务而非按 batch），
实质上退回了"通用 tiling 任务粒度"——SplitB 要消灭的东西。
修复：重构为照搬参考的单层 boIdx 循环 + 每 batch 四段：
  段1 QK 全部头（段内头循环）→ 每 batch 一次 SetFlag(qkReady)
  段2 softmax 本 vec 分摊任务（(h,s1o) 扁平序号对半）→ 双 vec 各一次 SetFlag(softmaxReady)
  段3 PV 全部头（批次级 WaitFlag(softmaxReady) 一次；每调用前自设满足 PV 内部等待）→ SetFlag(pvReady)
  段4 divout 本 vec 分摊任务
  ping/pong 按 boIdx 奇偶；workspace 改 per-batch 布局（S/OTmp/stats 三区 ×2 批），
  host 公式同步（nTask×(s1AreaF+oAreaF+statsPerTask)×2）。
预防：**实现结构与设计文档原则、参考源码三者必须逐条对齐**——设计评审时对照
参考的循环嵌套层级画结构图，不能只抄函数名。
附：行对半分摊（#23）改回任务对半分摊（整任务单 vec 处理），空分摊 vec 补最小 MTE3 写。

---

**#33**｜**数值 bug：Div 行广播的 src1 步长与 stats 布局错位（读专项文档发现）**
现象：用户指示阅读 perf/analysis 的四份专项文档（bmm1/bmm2/vec1/vec2）后交叉核对发现。
根因：参考的 sum 在 GM 是**每行 8 元素 padding**（[rows,8]，Vec2 文档 §6：src1RepStride=1
block 恰推进一行）；我方 stats 为**连续存储**（1 float/行，softmax ⑥ DataCopy 直写），
divout 却沿用参考的 src1RepStride=1 → 第 r 行除 sum[8r]，配对全错且越界读 stale UB。
修复：divout 改为**逐行标量 Div**（rows 次 `Div(..., sumVal, segs=colsPad/64)`）——
语义等价、不依赖 padding 布局；块广播 Div 待性能优化时与 stats 8-padding 一并对齐。
教训：**文档中的设计决策（如 8-padding）与实现必须逐条核对**——"同一思想"≠"同一布局"。
附：专项文档交叉核对确认：BMM1/BMM2 catlass 路径① ✓；P 未归一契约 ✓；LSE 我方合成 ✓。

---

**#34**｜**方案 B 重构：FAInfer tile 模型 + epilogue 封装 + stats 走 GM（用户 FIXME 审查驱动）**
背景：用户在 mha_fwd_splitb.cpp 中标注 6 处 FIXME，要求参照 mha_fwd_kvcache.cpp 重构：
- QK/PV 不得逐头双层循环（Q shape=(s1Inner,D) 逐行算效率低）——须用 FAInfer 的
  `rowNum = qSBlockSize × qNBlockSize` 打包多头（GetQNBlockTile：`(128/S1)/2*2` 封顶
  groupSize；恒偶 → 双 AIV 对半），blockMmad 内部处理打包行
- softmax/divout 须 FAInfer 式封装：epilogue 的 operator() 内部按 `qNBlockSize/subBlockNum`
  拆行给两个 AIV、SubCoreCompute 逐行块计算（替代 kernel 外层 myStart/myEnd 任务分摊）
- 构建四个子模块文件；能复用则复用
决策（用户拍板）：**方案 B**——保持参考的四段批结构与流水逻辑（核间 B 轴切分、每批
QK→SM→PV→RS 四段、flag 每批每段一次、ping/pong 按 boIdx 奇偶）。QK/PV 两段直接复用
qk_matmul.hpp/pv_matmul.hpp（tile 循环在 kernel 段内、单 tile matmul 机制在模块内）；
softmax/divout **重新实现**为 FAInfer 式封装——因为 online_softmax.hpp/rescale_o.hpp 的
stats（gm/gl）只在 UB 传递，四段批下 tile k+1 的 softmax 会覆写 tile k 的 stats，
必须加 GM 往返（参考实现同此做法）。
实施：
1. `splitb_softmax.hpp` 重写为 SplitBSoftmax<DType, HAS_SOFTCAP> 类：init（UB 偏移照抄
   FAInfer：LS 2×32KB ping/pong、LP 2×16KB、TV@160KB、LM/LL/SOFTCAP@168KB+）+ operator()
   （双 AIV 拆行、行块 ping/pong 预取、SubCoreCompute：rowmax→exp→cast→P→stats→GM）；
   stats 布局：tile 块内 [0,128)=max、[128,256)=sum，行距 128=Q_TILE_CEIL
2. `splitb_divout.hpp` 重写为 SplitBDivOut<DType> 类：GO@128KB 直读 OTmp、LoadStats 读
   GM stats（MTE2_V 自配对）、go/gl 广播除、O/LSE 打包行散射还原 BSND 头主序（CopyOToGm
   三分段 + 多头 LSE 逐 token gather/scatter，参数照抄 rescale_o.hpp）
3. `mha_fwd_splitb.cpp`：tile 几何 helper（照抄 FAInfer :513-541）+ 四段内 tile 循环 +
   事件预置扩展到 FAInfer 全量 + 尾部事件 drain（FAInfer :372-410，此前缺失）
4. `splitb_host.cpp`：workspace 改 tile 块模型（rowNumMax=128；tile 数 = CeilDiv(G,qNBlockTile)
   ×N2×CeilDiv(Sq,128)，host/kernel 独立计算同一公式，注释互指）
**附带发现的旧代码 bug**：旧 P 寻址用 `gP[sOff]`（half 索引=float 偏移）→ 字节地址只有
S 区一半，P 落在 S 区前半重叠区（多任务下 P(t) 与 S(t-1) 字节重叠，靠四段写读顺序侥幸
不炸）。新约定：**P = S 区原地 fp16 视图，half 索引 = 2×float 索引**（`gP[sOff*2]`），
四段批下安全（S 已被段2读完后段2才写 P；PV 读完 P 后下一批才覆写）。
教训：① **封装复用与否由数据流决定**——epilogue 的 stats 传递通道（UB vs GM）决定它能
否适配批级流水结构，不能只看"算法相同"；② GlobalTensor 的**元素索引单位随元素类型变**
（float 偏移 ×4B vs half 偏移 ×2B），同址多视图必须显式换算并在注释中写明。
附：causal/SWA 骨架（flag 链保活 + stub 写）已就位，S4 加 mask 重载即可；softcap 已随
HAS_SOFTCAP 模板穿透。

---

**#35**｜**文件形态对齐 FAInfer：SplitBKernel 类 + FAIKernelParams 入口（用户要求）**
现象：用户要求 mha_fwd_splitb.cpp 按 mha_fwd_kvcache.cpp 的主体形式重构——代码结构和顺序
与 FAInfer 一致，方便长期扩展；"如果实现方式差别很大，不利于后续长期维度"。
修复：照抄 FAInfer 骨架重组：
- namespace SplitB 内定义 `SplitBKernel` 模板类（模板轴 = BlockMmadQK/PV + 两个 epilogue +
  MASK_TYPE，照 FAInferKernel 的五模板轴形态）：空构造 `SplitBKernel() {}` →
  `operator()(FAIKernelParams const &params)`（tiling 经 params.tiling 读入成员）→
  `runMainLoop(boIdx, coreWsF, globalTensors)` → private 成员区（tiling 派生参数/strides/
  workspace 布局/tile 几何/resource/flag×3/四个模块对象）——顺序与 FAInfer 一一对应
- 文件尾部重开 namespace SplitB，模板入口 `FAInferSplitB`（类型组装 + FAIKernelParams
  打包 + `SplitBKernelType splitBKernel; splitBKernel(params);`）——签名不变，dispatch 不动
- 嵌套类型照 FAInfer：`TileGeom`、`GlobalTensorBundle`（10 个 GlobalTensor 引用）
- 结构性差异（仅一处，文件头注释声明）：FAInfer runMainLoop 为单 tile 粒度（任务跨核
  轮转），SplitB 为单 batch 粒度（B 轴切分 + 四段批）——tile 几何因此提为私有成员
  GetTileGeom（四段各需一次）；gmQ/gmK/gmO/gmLse 等每 tile 偏移照 FAInfer :524-535 公式
教训：**kernel 文件骨架是团队的长期接口**——类封装形态（构造/operator()/runMainLoop/
成员区）+ 入口组装两层结构与主 kernel 保持一致，后续特性（S4 mask、SWA、性能流水）
才有稳定的挂载点。

---

**#36**｜**语义 bug：误用 MAX_KV_STACK_LEN=512 于 SplitB 场景（用户 review 发现）**
现象：kernel 中 4 处使用 FAInfer 的 `MAX_KV_STACK_LEN`（512，常规 kernel 的 KV 逐栈
迭代长度），但 SplitB 的 S2 不切分（单 KV 栈即整段 S2 ≤ 128）。
排查（maxKVStackLen 全部消费点）：① getKVOffset 只与 nIdx 相乘，我方恒 nIdx=0——
正确性零影响；② resetBlockStart(kvStart=0)——0×任何=0，无影响；③ paged 分支——
死代码；④ blockStackNum 传参——PV 体内从未使用（残留参数）。**唯一实际影响：
L1 预算公式 `L1_MAX_SIZE - D×512×dtype` 给 V tile 预留 512 行空间，实际只需
colsPad(≤128) 行——D=128 时多预留 96KB/512KB**。今天 nDyn 恰被 L1_MAX_N_NUM=128
封顶未造成功能损失，但语义错误且脆弱（D 变大/预算公式调整时会无谓压小 nDyn）。
修复：SplitB 命名空间定义 `S2_STACK_LEN = Q_TILE_CEIL = 128`，替换 L1 预算、
QK/PV init 的 KVStackLen、blockStackNum 三处；常量注释写明与 512 的区别。
教训：**照搬参考公式时逐个常量问"它在我的场景里语义还成立吗"**——尤其是作为
"容量上限/预留尺寸"进入预算公式的量（512 在 FAInfer 是迭代单位，在 SplitB 是浪费）。

---

**#37**｜**代码卫生：magic number 清理 + pagedBlockSize 穿透（用户 review 三点意见）**
内容：
1. 恢复 FAInfer 的 `pagedBlockSize` 数据通路：SplitBTilingData 增 `blockSize` 字段
   （host 设 DEFAULT_BLOCK_SIZE=128），kernel 读为成员并贯穿 resetBlockStart×2 /
   QK 调用 / PV 调用 / blockStackNum——替代原先散落的 4 处裸 128
2. 上取整改用库函数：`colsPad/dPad` 的 `(x+15)/16*16` → `RoundUp(x, FaiKenel::BLOCK_SIZE)`；
   `blockStackNum` 的 `(a-1+b)/b` → `CeilDiv(S2_STACK_LEN, pagedBlockSize)`
3. QK/PV 调用的 `0, 1, 128` 字面量 → FAInfer 命名局部变量（kvSIdx/nowkvSIdx、
   kvSLoopNumTotal；PV 的 nIdx/nLoop 为引用参数须非 const 左值）
保留的裸数字：GemmShape<128,...>（FAInfer 原样字面量，保持与参考逐字符一致）；
调试 stub 写的 8×sizeof（debug 专用）。另 #36 修正补记：S2_STACK_LEN=128 不再
别名 Q_TILE_CEIL（用户指出概念不同，仅数值巧合）。
教训：**常量穿透优于散点字面量**——同一语义的数值（页大小）应有单一来源
（tiling → 成员 → 调用点），否则将来 paged 支持时要在 4 处分别改。

---

**#38**｜**同步机制纠错：删除 CUBE 自设 softmaxReady + 段序重排为计算序（用户 review）**
问题 1（用户 FIXME）：段3 CUBE 在每次 blockMmadPV 前 `Set(softmaxReady)`——语义错误：
该 flag 的含义是"VEC 已完成 softmax"，只能由 VEC 置位；CUBE 自设是掩盖 PV 内部 wait
配对不足的 hack（源自旧批级 flag 结构）。
问题 2（用户 FIXME，二次提出）：代码块按核分组（CUBE 块含段1+段3、VEC 块含段2+段4），
文本序与计算序 QK→Softmax→PV→DivO 不符，要求参照 FAInfer 按计算序排布、宏区分执行核。
机制研究（本次修正的依据）：
- 底层原语：Set→`ffts_cross_core_sync(pipe, msg)`、Wait→`wait_flag_dev(flagId)`
  （dav_c220 kernel_operator_sync_impl.h:430/439，编译器 builtin，计数语义不可见于源码）
- **ffts 未消费 set 深度上限 15**（catlass cross_core_sync.hpp 头注释：连续 set >15 次
  无 wait 会挂死；MAX_REVERSE_DEPTH/WithReverse 即为此设计）
- FAInfer 的配对形态：每 tile VEC（双 AIV 各自）Set(softmaxReady) 一次 + blockMmadPV
  内部每调用 Wait 一次——1:1 配对；双 AIV 槽都置位才放行 = P 双半区都落 GM
修复：
1. 段2 每 tile `smEpilogue()` 后 `Set<0x2,PIPE_MTE3>(softmaxReady)`（PIPE_MTE3 保证
   在本 tile P 拷贝之后置位）；段3 删除批级 Wait 与自设 hack，PV 内部 wait 直接消费
2. runMainLoop 重排为四段计算序文本块：段1 QK[CUBE]→段2 Softmax[VEC]→段3 PV[CUBE]→
   段4 DivO[VEC]（每段一个 #ifdef；CUBE 依序执行 1→3，VEC 依序执行 2→4）
3. debugStage==3（softmax 跑而 PV 门控）：CUBE else-branch 等 N 次保持配对；
   stage 1/2（softmax 不跑）：零 set 零 wait 自然平衡，删除旧 stub-set 块
4. host 守卫 `nTilePerBatch ≤ 15`（TORCH_CHECK；S6 并入触发闸门回落旧路径）——
   批内 tile 数 = VEC 领先深度上限，防 ffts 4-bit 溢出挂死
附带收益：段2/段3 天然逐 tile 流水（VEC 算 softmax(t+1) 时 CUBE 已可跑 PV(t)），
不再"整批 softmax 完才整批 PV"。
教训：① **flag 置位权属于完成工作的一侧**——消费方自设是掩盖配对缺口的自欺；
② 排查同步问题先读原语实现与库头注释（ffts 深度 15 这类硬约束藏在注释里）；
③ 代码块按计算序而非核序组织，与参考一致才可长期对照。

---

**#39**｜**同步粒度定案：softmaxReady 回归批粒度，pv_matmul 增可选 flag（用户决策）**
背景：#38 改为"段2 每 tile set + PV 内部逐调用 wait"的 1:1 配对（FAInfer 形态），
但引入 ffts 深度 15 守卫（host 拒绝 nTilePerBatch>15）。用户指出这不是优雅做法：
SplitB 的正确语义是**完整 Batch 计算完才同步一次**（参考 BMM2 即整批一次 Wait），
应改 PV 内部实现而非让调用侧迁就。
修复（用户方案）：
1. `pv_matmul.hpp` operator() 的 `softmaxFlag` 参数增缺省值 `CrossCoreFlag(0)`
   （id=0 = 无效）；体内 `if (softmaxFlag.id != 0U)` 才 Wait——传真实 flag
   （FAInfer 路径）行为不变，向后兼容
2. kernel 段2 改回"全部 tile 完成后 set 一次"（PIPE_MTE3 在全部 P/stats 拷贝后
   置位）；段3 批级 Wait 一次（门控与 softmax 同条件：stage 0/≥3），PV 逐 tile
   调用**不传 flag**（吃缺省值，内部零等待）
3. 删除 #38 的 host nTilePerBatch≤15 守卫与 kernel 内 ffts 深度注释（批粒度下
   未消费 set 深度恒 ≤1，约束不再相关）
握手平衡（每批）：qk 1 set（CUBE）↔ 2 wait（双 AIV）；softmax 2 set（双 AIV）
↔ 1 wait（CUBE 批级，softmax 未跑时两侧都不动作）；pv 1 set ↔ 2 wait。
教训：**同步粒度跟算法结构走，不跟底层算子的实现细节走**——当共享算子的内部
同步与目标结构不匹配时，给算子加"可选关闭"的门（缺省参数）比调用侧补偿
（自设/密集配对/深度守卫）干净得多；且改动必须向后兼容（缺省 = 原行为）。

---

**#40**｜**机制修正：缺省哨兵 flag → dispatch policy 编译期开关（用户质疑驱动）**
问题：#39 的 `softmaxFlag = CrossCoreFlag(0)` 缺省哨兵（id=0 视为无效跳过 wait）有
两个缺陷（用户指出）：① GetffstMsg 中 `flagId & 0xf`——id=0 硬件可编码、**并非架构
保留值**（我方与 FAInfer 只是恰好从 1 编号）；若某调用方真用 flag 0 做同步，哨兵会
静默吞掉其 wait → 数据竞争；② 运行时 `if (id != 0)` 分支不符合 catlass 惯例。
修复（catlass 惯例 = 行为开关放 dispatch policy，编译期）：
1. fa_block.h：`MmadAtlasA2FAIPVT` 增第三模板参数 `WAIT_SOFTMAX_FLAG_ = true`
   （缺省即 FAInfer 原行为，源兼容）
2. pv_matmul.hpp：特化模式与 DispatchPolicy 别名同步加第三参；wait 改
   `if constexpr (DispatchPolicy::WAIT_SOFTMAX_FLAG)`；**删除缺省参数哨兵**，
   softmaxFlag 恢复为必传参数
3. splitb：`MmadAtlasA2FAIPVT<false, false, false>` 类型级声明"批粒度同步"；
   PV 调用照常传 softmaxReady（类型门控下不等待）
优点：零哨兵、零运行时分支、误用不可能（要关闭必须显式写第三参）、FAInfer 的
`<PagedCacheFlag, false>` 实例化路径完全不变（不同模板实参 = 不同类型）。
教训：**"无效值哨兵"要求该值在域内真正不可用**——选哨兵前先查编码函数确认值域
（本例 0 可编码即不可作哨兵）；表达"行为开关"用类型系统（模板参数 + if constexpr），
不侵入运行时数据通路。

---

**#41**｜**调试脚手架盘点与加固：stage 门控具名化（用户质疑驱动）**
用户问题：代码中插入大量原功能没有的调试代码，是否会造成麻烦、有无把握。
盘点结论（四类）：
① stage 门控 ×5（纯调试）：每批一次标量比较，性能可忽略；stage=0 时全走真实分支，
  生产行为等价。风险不在性能在**耦合**——softmaxReady 的 set(段2)/wait(段3) 必须共享
  同一谓词，原为两处独立复写的条件式，改一处即引入 stage=3 死锁
② mask 骨架 stub（S4 占位）：仅在 MASK_TYPE≠NO_MASK 的模板实例中存在，S3 的
  NO_MASK 二进制不含此分支
③ epilogue 0 行子核 stub（**生产代码**，非调试）：rowNum=1 等极端形状 subBlock1
  分到 0 行时的 #23 竞态防护，须保留
④ STAGE=3 dump 探针（纯调试，**设计欠债**）：原始 S 转储进真实输出 gO（当时无专用
  dump 区）；有门控但 env 忘删会静默污染 O
加固：①的 5 处条件式改为 4 个具名开关（qkRun/smRun/pvRun/divRun）单点定义；
smRun 同时引用于段2 Set 与段3 Wait——配对不变式从注释约定升级为结构保证。
拆除计划：S3 golden 通过后删①④（④若仍需 dump 先迁 workspace 尾部专用区），
保留②（S4 实现时替换）与③（生产防护）；已列入 S6 收尾清单。
教训：**bring-up 脚手架要"单点定义 + 预定拆除日"**——散点复写的调试条件式与
生产不变式（flag 配对）耦合是最大的隐患源；debug 探针不写生产输出张量。

---

**#42**｜**调试脚手架全量拆除，改用 printf 探针（用户指令；t7 挂死驱动）**
背景：t7.log 显示 kernel launch 后挂死（stage=5 全真实路径，门控全开——挂死与门控
无关，但用户要求：删除所有 dump 等侵入式调试、只留 AscendC::printf 看调用链路，
降低代码复杂度便于掌握）。
拆除内容：
1. **stage 门控体系全删**：debugStage tiling 字段 + host FLASH_ATTN_SPLITB_STAGE
   env + kernel 5 处门控（#41 的 qkRun/smRun/pvRun/divRun）——四段无条件执行
2. **dump 探针全删**：DumpS 方法、gDump 参数与调用、段2 的 dumpCase；测试脚本
   dump_probe 与 FLASH_ATTN_SPLITB_DUMP 分发同步删除
3. **mask 骨架 stub 删**（用户判定为调试占位）：段2 else 只留 TODO(S4) 注释；
   S4 实现前 mask 型模板不可用（dispatch 不应路由）
保留：debugFlag（env FLASH_ATTN_SPLITB_DEBUG）→ host 单核强制 + 设备 printf；
epilogue 0 行子核 stub（**生产防护**非调试，rowNum=1 形状的 #23 竞态，用户 FIXME
"这种做法是否正确"——答：正确且必要，devlog #23 实测）。
新增 printf 探针（debugFlag 门控，5 个点）：kernel 入口（核号/批区间/tile 数）+
四段各一段首探针（打印点在同步 wait 之后）——**最后一条输出即把挂死定位到下一
个 wait**。printf 于生产为零开销路径（flag=0 单标量测试）。
教训：调试设施分层——**printf 链路（低成本、可常驻）≠ 执行掩码/dump（高侵入、
必须限期拆除）**；bring-up 中后期若仍在用掩码二分，说明该换 printf/单元化了。

---

**#43**｜**🎉 S3 首次 golden 通过（MHA fp16）；golden 自身 GQA bug（用户发现）**
突破：t9.log 前 3 用例（S3-core/b1/b8，MHA fp16 D=128）**3/3 PASS，max_err=0.0001**——
#34 重构 + #39 批粒度 flag + #42 简化后的方案首次全链路数值正确。printf 链路同时
确认：CUBE（blk0）与双 AIV（blk0/blk1，均归一化 c0）四段全部走通、逐批推进无挂死；
GQA 用例（gqa2）kernel 链路同样完整跑完。
问题（用户发现）：全量测试在 gqa2 崩于 golden——test_splitb_s3.py torch_ref 未处理
GQA（qf H=8 与 kf Hkv=4 直接 matmul 维度不匹配）。**kernel 无责**。
修复：golden 按 q 头 h → kv 头 h//G 映射 repeat_interleave 展开 K/V（与 kernel 的
kvNIdx=qNBlockIdx/qNBlockNumPerGroup、qNStartIdx=kvNIdx*G+... 映射严格一致）。
教训：**golden 先于用例覆盖做对**——参考实现的支持范围（GQA/bf16/非对齐长度）必须
先于全量网格验证，否则 kernel 无责的崩溃会污染判读。
**收官（t10.log）**：golden 修复后全量 **12/12 PASS**（max_err：fp16 各用例 1-2e-4、
bf16 1.2e-3）——覆盖 MHA b1-b20、尾块 Sq=48、**多 qS 块 Sq=200**（curQSBlockNum=2）、
Sk=32、**GQA gqa2/gqa8**（多头打包 qNBlockTile=2、rowNum=128 路径）、D=64、bf16。
**S3 判据达成**。遗留盲区：① FLASH_ATTN_SPLITB_DEBUG=1 会强制单核——上述通过均为
单核；多核（coreNum>1 的 workspace 分段 + 跨核 ping/pong）待 `FLASH_ATTN_FORCE_SPLITB=1`
（无 DEBUG env）跑一轮验证；② b128/b1024 用例被临时裁剪，S4/S5 补回。

---

**#44**｜**⚠ 已知问题（延后）：-O2 编译下失败（用户提供信息）**
现象：此前所有通过均在 setup.py `DEBUG_MODE=TRUE`（**-O0 -g3**）下取得（含 t11 多核
12/12、t12 补回 b128/b1024 后全过）。`DEBUG_MODE=FALSE`（-O2）后：未跑完 12 用例，
已跑用例 **FAIL（结果不对）**，若干用例后**卡死**。
机理（用户指出）：-O2 对跨流水线（AIV/AIC/Scalar/MTE）同步要求更严格——指令调度更
激进，-O0 下靠时序巧合成立的隐式依赖会暴露。-O0 通过≠同步模型正确。
决策（用户拍板 + 补充）：**先走 S4**（-O0 下完善功能与正确性），-O2 调试延后统一做；
但先做**一轮失败特征采集归档**（t13_o2_noprobe.log / t13_o2_probe.log 两轮对照：
首个 FAIL 的形状与 err 量级、挂死卡点探针、带/无探针差异）——S4 前代码库最小，
归因窗口最佳；特征指导 S4 的 mask 路径写法（避免克隆错误同步模式）。
候选嫌疑（采集后分诊用）：① flag 置位早于数据真正落 GM 的竞态（时序敏感）；
② epilogue 内 V pipe 相邻算子缺 PipeBarrier 的隐式依赖；③ UB 区跨段复用的生命周期；
④ 尾部 drain 事件不完整。
教训：**-O0 只验证数学正确性，同步正确性必须以 -O2 为准**；性能数据（S5）也只在
-O2 下有意义——-O2 调试在关键路径上，不可跳过。

**#44.1**｜**t13 特征采集结果（2026-08-19，两轮对照，b1024 已注释 → 13 用例）**
现象修正：-O2 下**不再卡死**——13 用例全部执行完成，仅部分结果 FAIL。**卡死与 b1024
强相关**（注释后消失），待 b128 类大 B 单独复验。
结果表（noprobe / probe 两轮，extract 脚本 debug/extract_results.py --diff）：

| 用例 | noprobe | probe | 备注 |
|---|---|---|---|
| core / d64 / bf16 | 2.0401 / 2.0092 / 2.0342 | 同左 | 两轮**完全相同**（确定性） |
| sk32 | 5808.2065 | 同左 | 垃圾级（读错区域） |
| b1/b8/b12/b20/b128/tail48 | 1.97~3.52 | 漂移 | 带/无探针 err 不同 → **时序竞态** |
| gqa8 | PASS(2e-4) | **FAIL(0.0668)** | **探针翻转** → 边缘竞态实锤 |
| sq200 / gqa2 | PASS | PASS | 两轮稳定 |

**核心相关（最强判别）**：1 tile/批（qS×qN=1×1，全部 Hkv=8 Sq≤64 用例）→ 全 FAIL；
≥2 tile/批（gqa2/gqa8=2N、sq200=2S1）→ PASS（探针扰动下 gqa8 边缘翻转）。rowNum
（64 vs 128）与批数均不判别（sq200 rowNum=64 却 PASS）。
**err 量级** ~2-3.5 ≈ ΣV（未归一化输出量级）：与"DivO 拿到 sum≈1/max≈0 或读到错数据"
自洽；sk32 例外（5808，另查）。
**探针序列**（probe 日志 37-94/1409-1466 对照）：失败与通过用例的 VEC 标记**完全一致**
——每批 S2-SM(b)→S4-DO(b) 顺序正确 → **排除 VEC 阶段乱序**，问题在数据可见性/槽位
复用类竞态。
**代码对照**：qkReady/pvReady=PIPE_FIX、softmaxReady=PIPE_MTE3 与 FAInfer
（mha_fwd_kvcache.cpp:675/849/904）**逐字相同**，而 FAInfer -O2 生产正常 → 排除 flag
模式本身（候选嫌疑①降权）。候选收窄到 SplitB 特有机制：② 批循环 ping/pong 槽位
（FAInfer 无批循环）；③ stats 走 GM 的 MTE3→MTE2 可见性；④ 多 tile 循环内 epilogue
事件 flag（EVENT_ID 0/1/4/6）跨 tile 配对。
**下一步实验（进行中）**：FLASH_ATTN_SPLITB_ERRMAP=1（test_splitb_s3.py 新增）打印
argmax 位置 + per-batch/per-head 误差 + 最差 (b,h) 行模式（AIV 半区判别）——区分
批局部（槽位）/头局部（打包散射）/行局部（AIV 拆分）/全局（数据竞态）。

**#44.2**｜**根因候选（强）：softmax 的 CopyStatsToGm 缺 V→MTE3 同步（2026-08-19）**
排查思路修正（用户提醒）：-O2 暴露的不只跨核 flag 链，还有**同核内细粒度同步**
（V↔MTE3、V↔MTE2、AIV↔Scalar）。逐条审查 splitb_softmax.hpp SubCoreCompute 的
pipe 级依赖，发现：
```
④ DownCastP (V) → ⑤ SetFlag<V_MTE3> → ⑥ CalcLocalRowSum (V 写 llUb) → ⑦ SetFlag<V_MTE2>
→ ⑧ WaitFlag<V_MTE3>（消费⑤，与⑥无关）→ ⑨ CopyPUbToGm (MTE3) → ⑩ SetFlag<MTE3_V>
→ ⑪ CopyStatsToGm (MTE3 读 lmUb/llUb)
```
⑤ 只覆盖 ④（Cast）；⑥ 在 ⑤ 之后才发射；⑧ 不等 ⑥；⑨⑪ MTE3 in-order 连续发射
→ **⑪ 读 llUb 时 ⑥（BlockReduceSum 长指令）可能未完成** → stats sum 读旧值 →
DivO 按错 sum 归一化 → O ≈ ΣV（err ~2-3.5，与 t13 观测吻合）。-O0 指令间隙大靠
时序掩盖；-O2 发射紧凑 → 暴露。
佐证：SplitBDivOut 的 LSE 路径（245-246）有同款 SetFlag<V_MTE3>+Wait 自配对（正确）；
Softmax 漏了。FAInfer stats 留 UB 不写 GM，无此模式 → 解释"FAInfer 同款 flag 却正常"。
修复（已提交代码）：SetFlag<V_MTE3> 从 DownCastP 后**移到 CalcLocalRowSum 后**（一行
移动，set/wait 配对守恒）：⑧ 等 ⑥ 完成 ⟹ P 拷贝（lpUb=④）与 stats 拷贝（llUb=⑥）
两处依赖同时满足。
验证：-O2 重编译 + 13 用例全量（t15）。若残留失败则继续查：AIV↔Scalar 隐式依赖
（GetTileGeom 地址计算提升）、多 tile 循环内 epilogue 事件 flag 跨 tile 配对。

**#44.3**｜**t15 复验结果 + b1 推论（2026-08-19，14 用例含 b1024）**
修复**部分生效**：
- sk32：5808.2（垃圾）→ **1.5614**（ΣV 量级）——独立的垃圾读被消除
- 速度：-O2 慢/卡死 → **几秒跑完 14 用例**（用户观察）——stats 竞态使 MTE3↔V 互锁、
  flag 链路连锁空转；修复后管线通畅（佐证 #44.2 修复真实有效）
- 但 1 tile/批 11 用例仍 FAIL（err 2.0-3.5，b1024=5.18）；sq200/gqa2/gqa8 仍 PASS
**b1 决定性推论**：B=1 单核单批单 tile 无任何复用/并发，-O2 确定性 FAIL（2.9198，
多轮逐位相同）→ **不是同步竞态，是 -O2 编译语义差异**（未初始化读/指令重排），
且与"1 tile/批"相关必然在单次执行路径内（排除批循环槽位方向）。
排除链（静态）：tile 几何 host/kernel 一致（GetQSBlockTile 恒 128）；S/P/O/stats
跨批槽位复用全部由 flag 链串行化（推导见上）；DivO 读 O 竞态（pvReady=PIPE_FIX）
与"ΣV 量级"不符（旧 O 是正确量级）；SM 读 S 竞态同理。→ 剩：stats 的 sum 被读为
≈1（未归一化）或 P/O 数据量级错——需错误位置数据判别（t16 errmap）。
下一步（t16）：FLASH_ATTN_SPLITB_ERRMAP=1 聚焦 b1/core/sk32/gqa2/sq200 六用例，
看 per-batch/per-head/行半区模式。b1 单批：per_batch 只有 1 值，若全错 → 全局；
per_head 判别打包散射；行模式判别 AIV。

**#44.4**｜**t16 脚本 bug（2026-08-19）**：errmap 打印行 `err[b0, h0, am[1], am[3]]`
索引错位——err 为 [B,Sq,H,D]，h0（am[2]）写进 Sq 槽、am[1]（Sq 索引）写进 H 槽 →
首用例即 IndexError（am[1]=36 > H=8）中断整轮。已修为 `err[b0, am[1], h0, am[3]]`
（:58）。教训：argmax→unravel_index 的元组维度顺序 [B,Sq,H,D] 与 err 访问必须一致，
与 per_batch/per_head/rows 的索引要一次核对齐。该轮 S3-core 已正常打印
splitb_max_err=2.0401（与 t15 ΣV 量级一致），未产生 errmap 数据，需重跑。

**#44.5**｜**t16 errmap 数据 + 指纹检验（2026-08-19）**：
errmap 7 用例：FAIL=core/b1/sk32/bf16（h0=0.000，h1-h6=0.07-0.17，h7=1.5-2.9，
rows 64/64 坏且 AIV0/AIV1 各半全坏）；PASS=sq200/gqa2/gqa8（全 0.000）。
h7 错误全行全 d 均匀 → 输出行退化为常数行（O[s,:]=Σ_j p̃_j·V_j 与 s 无关）。
CPU 指纹检验（debug/sv_fingerprint_check.py，与 t16 同种子）逐一证伪候选：
|ΣV−ref|=12.2 远大于观测 2.04（P≡1 假设 ✗）；ΣV/64≈0.08（✗）；"0.5·ref+0.5·ΣrawS·V"
在 max 误差值上巧合吻合（1.99/2.88/1.58/1.99 vs 观测 2.04/2.92/1.56/2.03），但在
**观测 argmax 位置**处 0.5mix 误差仅 0.56/0.05/0.34（比值 3.7/63/4.6）→ 证伪。
结论：观测 argmax 处（如 core b=1,s=36,d=86）ref=−0.216、ΣV=−12.3、ΣrawS·V=−1.33
均不解释 2.04 → 停止猜测候选公式，转向分阶段实测（用户建议采纳）。
分阶段方案：① 全 1 输入判别（无需重编译，输出值唯一标识损坏模式：正确 1.0、
P≡1&div≈1→64、div 减半→2、P=rawS→11.31/724、ΣV→64）；② DumpTensor 探针
（tiling dumpFlag + 四段 h7-tile dump，desc=111/211/221/222/311/411/412）。

**#44.6**｜**h7 根因定位与修复（2026-08-19）：divout 的 tvUb 双用途竞争**：
t17 全 1 判别：h0-h6 全对、h7 恒 4.1367（V=1 时）；V=s+1 时 134.375，同因子 4.1346
→ h7 的 O 被 **4.1367 常数因子**放大（P̃/sum = 1/15.47）。t18 DumpTensor 铁证：
S=128 ✓、P=1.0 ✓、max=11.3137 ✓、sum=64 ✓、OTmp=64 ✓ → 前三段全对，**段4 divout
除错**。除数 = 64/4.136719 = **15.472 = ln(64)+11.3137 = LSE 值**！
机制：divout 的 tvUb 双重用途——② Div 的除数广播 Brcb(tvUb←gl=64) 与 ⑤ LSE 的
Brcb(tvUb←lseUb=15.472) 写同一区。② Brcb（MTE2）→ Div（V）之间只有
PipeBarrier<PIPE_V>（只管 V），-O2 跨 pipe 调度使 Div 读 tvUb 时已被 ⑤ Brcb 覆盖
→ 除数 = LSE 值。h7 为最后 tile，调度窗口恰命中；带 dump 时 h2 半区也中（调度漂移）。
修复（splitb_divout.hpp）：① ② Brcb 后加 SetFlag<MTE2_V>(EVENT_ID2)+Wait 自配对
（Brcb 写 tvUb 完成后 Div 才读）；② ⑤ Brcb 前加 SetFlag<V_MTE2>(EVENT_ID4)+Wait
自配对（Ln/Add 完成后才读 lseUb）；③ ⑤ Brcb 目标改独立区 tvUbTensor[64]（与 ②
分离，gather 同步偏移，UB 容量核验 ✓）。
顺带修机制甲（h1-h6 stats 竞态，#44.2 遗留）：SetFlag<MTE3_V> 从 P 拷贝后移到
CopyStatsToGm 后，覆盖 P+stats 双写（t16 h1-h6 中等误差源；全 1 输入被 rowmax
处处相同掩盖）。
待验证：重编译 → t18 全 1（h7 应 1.0）→ -O2 全量 14 用例。

**#44.7**｜**概率性残留错误：坏行 AIV0 后半 s16-31，判刀 W 定位（2026-08-19）**：
机制乙修复后全量仍随机 FAIL（7/14±，三次跑 FAIL 集合漂移，err 0.06-0.11）。特征：
坏行恒 AIV0 后半（core s16-31、tail48 s12-23）；fp16 单核 10/10 PASS、bf16 单核
~1/5、fp16 多核随机；b1 恒 PASS；全 1 输入掩盖。判刀 W（tile/行/列三维区分的
结构化输入：Q=(h+1)(s+1)/512、K=(j+1)/64、V=j+1）100% 复现且模式确定。
DumpTensor 单核（t26）：稳定 b2 h0 s63 err=6.25——tile0（h0）的 stats 拷贝流
**末行**被污染。

**#44.8**｜**机制丙根因与修复（2026-08-19）：RowMax 写 lmUb 与上一 tile stats 拷贝的竞态**：
SubCoreCompute 原顺序 CalcLocalRowMax → CalcExp → WaitFlag<MTE3_V> → DownCastP：
RowMax（V 写 lmUb）在 MTE3_V wait **之前**执行，而上一 tile 的 CopyStatsToGm
（MTE3 读 lmUb/llUb）靠该事件保护 → 下一 tile 的 RowMax 可早于上一 tile 的 stats
拷贝完成 → 上一 tile stats 被污染（拷贝流末尾行 s63 概率最高，t26 实证）。
#44.6 只移了 set 未移 wait 消费点，竞态仍在（t16 h1-h6 中等误差同源）。
FAInfer 无此模式：其 stats 驻留 UB（dm），无 MTE3 读 lmUb。
修复：WaitFlag<MTE3_V> 移到 CalcLocalRowMax 前（CalcExp 读的 lsUb 槽由 V_MTE2
链保护，MTE3 不碰，提前无副作用；lpUb/llUb 保护不变）。
待验证：重编译 → t26 判刀 W（应 20/20）→ -O2 全量 14 用例。

**#44.9**｜**机制丁根因与修复（2026-08-20）：双 AIV 共用 softmaxReady flag id 的
"任一通过"竞态**：
用户指示批判式重读权威 online_softmax.hpp 后定位。#23 原始记录实证跨核 flag 为
**"任一置位即通过"语义**（空管道 set → PV 在 P 写完前通过 wait）。SplitB 段2 末尾
双 AIV 各发一条 softmaxReady（同 flagId=2）→ CUBE 在 AIV0 完成后即开始 PV，
**AIV1 的 P 拷贝（8 tile MTE3 队列）尚在飞行** → PV 读 AIV1 半区未写完的 P →
AIV1 行错（t26/t27 b2 h0 s63 实证：s63 = AIV1 末行）。
现象全解：b1 恒 PASS（单批无相位累积，窗口小）；B=4 单核第 3 批撞（相位漂移）；
多核撞率高（GM 带宽竞争放大 AIV0/AIV1 完成差）；-O0 不撞（工作慢、完成差小）；
全 1 输入掩盖（P 处处相同）。FAInfer 不暴露：逐 tile set/wait，完成差仅 1 tile 窗口。
#44.2/#44.6/#44.8 的修复均非根因（但各自修了真实的潜在窗口，保留）。
修复：AIV0/AIV1 分用不同 flag id（2/4，避开系统保留 8/9/10）——AIV0 set
softmaxReady、AIV1 set softmaxReadyAiv1，CUBE 依序 wait 两个。kernel_common.hpp
加 SOFTMAX_READY_AIV1_ID=4。
待验证：重编译 → t27 判刀 W（应 20/20）→ -O2 全量 14 用例。

**#44.10**｜**Scalar/管道发射乱序加固（2026-08-20）**：
用户指导方向落地：CANN 头文件实证 set_flag/wait_flag/pipe_barrier/DataCopy 的发射
全部是 CCE_SCALAR（Scalar 单元）；Brcb/SetVectorMask/数学指令在 V（begin_pipe(V)/
PIPE_ID(PIPE_V) 实证）。-O2 下 Scalar 发射与 V/MTE2/MTE3 的执行可乱序——事件 set
可能在源管道工作完成前发射置位（判刀 W 残留错误的候选根因；错误位置随编译时序
漂移的特征吻合）。修复：所有关键 SetFlag 前加对应管道 PipeBarrier（Scalar 等管道
排空后 set 才发射）：
- splitb_softmax.hpp：RowSum 后 SetFlag<V_MTE3/V_MTE2> 前 PipeBarrier<PIPE_V>；
  双拷贝后 SetFlag<MTE3_V> 前 PipeBarrier<PIPE_MTE3>；CopySGmToUb 后
  SetFlag<MTE2_V> 前 PipeBarrier<PIPE_MTE2>
- splitb_divout.hpp：LoadStats/LoadO 后 MTE2_V set 前 PipeBarrier<PIPE_MTE2>；
  Cast 后 V_MTE3 set 前 PipeBarrier<PIPE_V>；CopyOToGm 后 MTE3_MTE2(6) set 前
  PipeBarrier<PIPE_MTE3>
待验证：重编译 → 判刀 W（B=2 20 轮）→ -O2 全量 14 用例。

---

**#25**｜**运行时（-O2 竞态，LSE）｜divout LSE 块 Brcb 与 SetFlag<V_MTE3>(EVENT_ID4) 间缺 PipeBarrier<PIPE_V>**
现象：与 FAInfer rescale_o.hpp:535-541 逐行对照发现（Brcb → PipeBarrier<PIPE_V> →
SetFlag<V_MTE3> 三步，SplitB 漏了中间一步）。devlog #44.10 同族竞态（Scalar set_flag
早于 V 发射）：-O2 下 MTE3 的 LSE gather 可能读到 Brcb 写入前的旧 tvUb → LSE 出错。
修复：Brcb 后补 `AscendC::PipeBarrier<PIPE_V>();`（用户已改）。
附带事实更正：查 CANN 实现（kernel_operator_vec_brcb_impl.h），**Brcb 是 V 管道指令**
（vbrcb intrinsic、if ASCEND_IS_AIV）——代码里"Brcb 跨 pipe"的注释前提是错的，
相关 SetFlag/WaitFlag 对是无害空栅栏，不要按错误前提继续推理。

**#26**｜**运行时（潜伏竞态）｜softmax 0 行子核防护只有注释没有实现**
现象：文件头注释说"补最小真 MTE3 写再返回（devlog #23）"，实际是裸 return——divout 有
完整 stub（DataCopyPad + SetFlag<MTE3_MTE2>(6)），softmax 没有。
后果：Sq=1 类 shape 下，0 行 AIV 不 set MTE3_V(0)/V_MTE2(0) → 下一 tile 消费陈旧计数
+ set/wait 不配平 → #23 同族竞态。
修复（方案）：stub 写点选**本 tile stats 区末 32B**（gStats[2*ROW_NUM_MAX-8]；0 行 ⇒
rowNum≤1 ⇒ 行 124-127 永不读，安全），写后 PipeBarrier<PIPE_MTE3> + SetFlag<MTE3_V>(0)
+ SetFlag<V_MTE2>(0)。注意 divout 的 stub 写 gOutput 头部在 Sq=1 场景会污染另一 AIV
的 O 区，建议同改 stats 尾。
预防：**照搬的"防护分支"必须与注释一致**——注释描述过的防护在重构中可能只留下注释。

**#27**｜**运行时（数值，追踪中）｜stats sum 确定性小偏差（tile6 s48/s50）**
现象：dump 探针（t32_divout_pre，output2.txt）显示 QK 逐位正确、OTmp 仅 fp16 精度差，
但 **tile6 的 sum s48/s50 有 0.0244/0.0227 确定性偏差**（8.9249 vs 8.9493；b0/b1 相同）——
fp32 行和 0.3% 偏差不可能来自舍入。S 探针只覆盖 tile0/2/3，未覆盖 tile6，来源未定位
（QK 写 S 错 vs RowSum 错）。
待办：S dump 扩到 tile6 + ERRMap 完整跑一遍，看最终 out 坏点是否对应 s48/s50 行。

**#44.11**｜**PIPE_ALL 注入二分 + t31-t33 探针演进（2026-08-21）**：
#44.10 加固后判刀 W 仍 0/20 FAIL → 用户提议 PipeBarrier<PIPE_ALL>（全流水最严同步）
注入二分定位竞态窗口：
- 阶段1（三处段边界注入 PIPE_ALL：qkReady/softmaxReady/pvReady set 前）→ 仍 0/20
  FAIL → 竞态在**段内**（跨核 flag 覆盖不到）；
- 阶段2（t31-t33）：sum dump 扩全部 8 tile × 2 batch（desc=224+boIdx*10+tile，
  PipeBarrier<PIPE_MTE3> 消除假象）→ 全对；段4 前 stats 同址再 dump（desc=424…）→
  全对；divout 内三点链 UB dump（521 glUb / 531 Div 后 / 541 Cast 后）→ b0 tile0 全对。
- 关键认识：① t28g 曾见的 S"垃圾"/OTmp"行错位"是 **dump 时机假象**（Fixpipe 拷贝
  未完成即读）——段末 PipeBarrier<PIPE_FIX/PIPE_MTE3> 后假象全消；② DumpTensor
  ≤1MB/核，段4 的 O dump 曾在缓冲压力下整条丢失（desc=412 缺失之谜）；
  ③ 错误位置随编译/时序漂移（b2→b1→h1/h2/h3），探针定点永远慢一拍。
教训：**时序漂移型错误不要用定点探针追**——每次编译都改变时序，位置必漂；
正确做法是每次编译抓全量（#44.12 方案）。

**#44.12**｜**分阶段全量验证方案（2026-08-21，用户主导调试）**：
前情：t28-t33 已证 GM 级中间数据（S/P/stats/OTmp）在好时机全对、破坏点在 divout
内部，但错误位置随编译时序漂移、探针反复微调编译成本过高。用户决策：换小样例 +
整区 dump + 基准同格式打印的"阶段解耦验证"。
方案三件套：
1. **测例**（可辨识/可手算）：B=2 Sq=32 Sk=32 H=2 D=128 fp16；
   Q=(b+1)(h+1)(s+1)/256、K=(j+1)/32、V=(j+1)+(d+1)/128 →
   S_raw=(b+1)(h+1)(s+1)(j+1)/64（小整数乘积，一眼可验）；
   tile 几何 = 2 tile 每头一个（qn=1, rowNum=32）。
2. **kernel 整区 dump**（mha_fwd_splitb.cpp，旧探针全清；每 dump 前 [SB-DUMP]
   printf 声明来源/布局）：desc=100+b 段1末 S 整区（float 视图，P 将覆盖故仅此时
   可读）；desc=200+b 段2末同区 half 视图（P）；desc=300 kernel 末（drain 后）
   workspace 整区（OTmp+stats 终态）；desc=400 O 全量；desc=450 LSE 全量。
   预算：S/P 各 2×162KB + WS 0.66MB + O/LSE ≈766KB < 1MB 每核（dump 模式单核 +
   B≤2 时槽不覆写）。
3. **基准脚本** debug/test_splitb_stage_full.py：--print-ref 打印基准各阶段；
   默认子进程跑 kernel 捕获 dump → 逐段比对（量化语义模拟：输入 fp16、P fp16 量化、
   sum 用 fp32 exp、OTmp=fp16P@fp16V）→ 每阶段 ✓/✗ 汇总 + 错误精确定位到
   b/tile/s/j/d；--log 离线解析。**自测全绿**（正向全相符 + 注入 S[b1,h1,s7,j9]+1
   被精确定位）。脚本内置 tile 几何复刻（GetQNBlockTile/GetTileGeom 的 Python 版）。
教训：反复"改探针→编译→跑"的迭代对时序漂移型 bug 收束极慢；一次编译抓全量、
比对自动化才是正路。同时 dump 语义关键：DumpTensor 只支持连续区间，workspace 是
tile 块状布局（[S|OTmp|stats]×tile）不连续，不能按 [H,Sq,Sk] 逻辑视图直接 dump。

**#44.15**｜**DumpTensor 1MB 预算超限：整区方案数据全丢 + 捕获层教训（2026-08-21）**：
#44.12 整区 dump 方案首跑实测：单轮总预算 ≈1.03MB（S 331K + P 331K + WS 331K +
O 32K + LSE 0.5K）**恰好超 1MB** → 输出只保留了最后一条（desc=450 LSE），其余
全部丢弃（缓冲滚动行为）。且 `contextlib.redirect_stdout`（Python 对象级替换）
**抓不到 device dump**（C 层走 fd 1）→ 解析 0 条。
修复：① kernel dump 改逐 tile 有效区紧凑版（S=100+b*10+tile 4KB/条、P=200+…2KB、
OTmp=310+…16KB、stats=330+…256B，每轮 ≈122KB ≪1MB；每条 printf 带
core/b/tile/qStart/行列数，来源明确）；② 脚本捕获改 os.dup2 fd 级重定向。
顺带收获：**新样例（B2/S32/H2/D128 fp16）max_err=0.0000 两轮 PASS**——错误未复现，
小 shape 是否规避了竞态窗口待观察（若持续 PASS，可逐步放大 shape 找触发边界）。
教训：① DumpTensor 预算 = 每核每次 kernel 调用 1MB，**含头部开销，预留余量**；
② device 输出捕获必须 fd 级（redirect_stdout 无效）；③ 整区 dump 只适合紧凑布局，
块状 workspace 的有效数据占比低时逐区紧凑 dump 更划算。

**#44.47**｜**【多核首测全通过】B=4→1024 × 20 核全 pass；B=128 挂死销案（2026-08-24）**：
t60 系列（用户跑，`--multi-core --debug`，无 dump）：B=4（×5 重复）/10/30/60/100/128/
512/1024 全部 `[OUT] max_err=… PASS`。证据：host 打印 coreNum=20 splitF=52（20 核 ×
每核 52 批），每核 enter/exit 标记完整；max_err=0.0156 = fp16 在 [16,32) 的 1 ULP
（量化噪声级）。用户已自行给脚本加 **O 张量级校验**（[OUT] 行）——多核/无 dump 形态下
的正确验证方式。**记录在案的"B=128 多核挂死"未复现 → 销案**（推测被 #44.40 系列事件
修复顺带解决）。目标工作负载（B=1024, S=32）多核正确性达成。
剩余测试矩阵：bf16 / GQA（H≠Hkv）/ 多头 tile（G≥2 使 qNBlockTile>1，如 heads=8
kv-heads=4）/ Sq=128 / Sk≠Sq / D=128 / 官方 pytest 套件 + 触发闸门路由确认。

**#44.46**｜**多核开关独立化：MULTI_CORE env 与调试开关解耦 + dump 全核化（2026-08-24）**：
用户指出：此前所有测试（t5x 全系列）都被强制单核——链路＝脚本无条件设
FLASH_ATTN_SPLITB_DUMP → host `usedCoreNum=(dbg||dump||smOnly)?1:min(B,aicNum)`（调试
早期按用户要求加的单核防串扰），**多核路径从未执行过**。按用户设计改为：
- host：独立 `FLASH_ATTN_SPLITB_MULTI_CORE` env 控制 usedCoreNum（未设→单核默认；
  已设→min(B,aicNum)），与 debug/dump/smOnly 完全解耦——少核也可以 dump。
- kernel：dump 门放开到全核（CUBE 侧 `dumpFlag`；VEC 侧 `dumpFlag && AIV0-only` 防双份）；
  desc 按全局 boIdx 编号跨核唯一，多核 dump 记录不撞号；DumpTensor 预算按核独立。
- 脚本：`--multi-core` 开关（设上述 env，dump 照常）。追加（用户要求）：`--debug`/
  `--dump` 参数化——FLASH_ATTN_SPLITB_DUMP **不再默认开启**（此前脚本无条件设置），
  FLASH_ATTN_SPLITB_DEBUG 补上显式开关；不带 --dump 的运行提示"--log 比对将无数据"。
待验证：多核首测（B=2/4 → 2/4 核）七项全绿——注意多核 dump 记录的跨核交错是否影响
parser 的记录组装（printf 行与数据行配对）；若解析乱，需按 desc 关联而非行序。

**#44.45**｜**【清理收尾】裸 printf/PIPE_ALL/drain 修复 + host 静默化（2026-08-24）**：
用户清理阶段第二步（DumpTensor 七项数据代码**全部保留**——用户明确要求，后续排障还要用）：
- 删 softmax.hpp 未门控 printf：[P-COPY]（#44.22 遗留）与 [SOFTMAX_DEBUG]（含 TODO 注释）。
- 删 kernel 两处 PIPE_ALL"DBG 注入"死注释行（用户已注释，#44.11 遗留；两处 PIPE_ALL 本体
  均已不在执行路径——**注意：这意味着当前验证过的形态本就无注入屏障**）。
- **drain 补 MTE3_V(3)**：softmax AIV1 的 pingpong1 链（evId=1+2×1=3，#44.24）——原
  drain 照抄 FAInfer 只有 0/1/2/4，AIV1 末 tile 的 MTE3 拷贝（P/stats）未被等待即出核，
  back-to-back launch 有脏写风险。
- host printf 静默化：编号梯子 333/444/555/777/888 删除，222/999/1000/9999 收敛到
  dbgEnv 门控（定义提前至函数头，默认完全静默——此前每次 forward 打 10 条）。
- 保留：dumpFlag 门控的 DumpTensor 块（七项数据）、debugFlag 门控的 [SB] printf、
  softmaxOnly 机制（S4 排障用）、用户已注释的探针块原样不动。
待验证：重编译后 -O0/-O2 × B=2/4 回归（printf 删除改变时序，必须回归）。

**#44.44**｜**【清理】workspace 规范化：P 区独立基址 + 布局量命名对齐 FAInfer（2026-08-24）**：
用户清理阶段第一步（历史遗留：链式时代 gS/gP 同基址 + 寻址掺混合偏移）。改动：
- **布局重排**：每核两段连续 [tile 区（2 批 × T tile 块）| P 区（2 批 × T 槽）]——
  原"#44.35 每批内交错 [tiles|P 槽]"废止。总量不变（perCoreElems = 2×T×(perTileElems+pSlotElems)）。
- **独立基址**：gS/gOTmp/gStats → 本核 tile 区首；**gP → 本核 P 区首**（独立指针）。
  视图含 coreWsOffset，runMainLoop 的 batchBase = batchBuf×perBatchTileElems（不再掺
  coreWsOffset，签名删参）；GetPHalfIdx(batchBuf, tileIdx) 一行寻址（原实现掺
  batchBase+tile 区偏移，是链式遗留的不优雅）。
- **命名规约**（用户指出 F 后缀含义不明）：废除 *F 后缀，改 **\*Elems = float 元素数**
  （对齐 FAInfer 元素计数语义 MAX_UB_S_ELEM_NUM）：sTileElems/pSlotElems/oTmpTileElems/
  perTileElems/perBatchTileElems/tileAreaElems/perCoreElems；coreWsF→coreWsOffset。
  kernel+host 同步重命名；FAInfer 区段对照（S=mm1Out/P=smOnlineOut/OTmp=mm2Out）记于
  成员声明注释。
待验证：重编译后 -O2 B=2/4 七项全绿（布局等总量重排，功能不应变化）。
追加（用户要求）：**GetPHalfIdx 整个删除**——P 偏移与 gS 同款就地计算
（`pOff = (batchBuf×T + tileIdx)×pSlotElems×2`，紧邻 sOff/oOff 声明；dump 点内联同式），
顺带修正 PV 处过时的"P 为 S 区原地视图"注释（链式时代残留）。

**#44.43**｜**【S3 完成】ScaleS 恢复后全矩阵回归全绿（2026-08-24）**：
用户复测（t57）：-O2 B=2/4（含 B=6 抽查）七项全对——S3（NO_MASK/fp16/B≤4）全流程
正确收官。本阶段（#44.12-#44.43）三大根因：①P/S GM 复用跨 AIV 竞争（#44.23/#44.37，
解耦=FAInfer 哲学 #44.39）；②divout 跨 tile 事件两重缺陷（#44.40）；③dump desc 撞号
假象（#44.41）。下一步 S4：causal/SWA mask（softmax mask 重载 + host 分发）+ softcap
验证；S5 性能；S6 默认启用+清探针（含 kernel 末尾 drain 缺 MTE3_V(3)、B≥8 desc 撞号）。

**#44.42**｜**ScaleS 恢复（2026-08-24）**：
splitb_softmax.hpp :527 的 ScaleS 解注释（调试期曾禁用对齐脚本，#44 早期）；脚本
s_sc = s_raw × SCALE（=1/√D）恢复。一致性核验：scale 在 softmax 的 UB 拷贝内做
（Muls in-place），GM 的 S 保持 raw QK 输出 → S dump(100 系) 对脚本 s_raw 的比对
不变；P/max/sum/OTmp/O/LSE 的 ref 自动经 s_sc 生效。待跑全矩阵回归：
-O0/-O2 × B=2/4/6（fp16；bf16 走主测试套件）。

**#44.41**｜**【里程碑】#44.40 修复验证通过：全流程七项 -O0/-O2 × B=2/4 全绿（2026-08-24）**：
t56（-O2）：B=2 七项全对（S/P/max/sum/OTmp/O/LSE）；B=4 仅 OTmp(b2/b3 t0) 报错——
**desc 撞号假象**：OTmp 家族 310+b×10 在 b≥2 与 stats 家族 330+b×10 撞号（b2 t0=330
=stats b0 t0），parser 把 stats 记录当 OTmp 比对（"错误值"0.25/0.5/0.75…正是 max
数组小线性值）。决定性反证：O/LSE 全对（O=OTmp/sum 从真实 GM OTmp 算出）。
修复：OTmp desc 改 **600+b×10+tile**（kernel + parser 同步；B≤9 无冲突。遗留：B≥8
时 stats b7 t0=400 与 O b0=400 仍会撞，届时再迁 O/LSE 家族）。另：OTmp dump 的
"data is not enough" 为 shape(2048) vs count(1024) 固有不匹配告警，B=2 亦有，无碍。
**状态：S3（NO_MASK/fp16/B≤4/-O0/-O2）全流程正确。** 下一步：恢复 ScaleS（kernel
注释行 + 脚本 s_sc）→ 全矩阵回归 → S4 mask/softcap → S5 性能 → S6 默认启用+清探针。

**#44.40**｜**【-O2 LSE bug 定位+修复】divout 跨 tile 事件两重缺陷（2026-08-24）**：
t55（解耦布局，B=2）：-O0 全对（t55_b2_o0.log）；-O2 仅 LSE b1 h0 s16-31 错（恰为 AIV1
行区间；O/stats 全对）。**数值定源**：16 个错误值与 tile1(h1) 的原始 sum[16..32) **逐位
全等**（1.528096=sm1[16] 等）——写出的不是算错的 LSE，而是未做 Ln/Add 的 h1 原始 sum。
LSE 计算路径（divout ⑤）：Ln 就地读 glUb 写 lseUb（**lseUb≡glUb 同址**）→ Add gmUb →
DataCopyPad(gLse)；跨 tile 顺序：t 的 LoadStats（MTE2 写 gl/gm）→ SubCoreCompute →
末尾 O/LSE MTE3 拷贝 → 下一 tile LoadStats 覆写。**两重缺陷**：
1. **Wait 位置错**：跨 tile 保护 Wait<MTE3_MTE2>(6) 原在 SubCoreCompute 入口——晚于
   LoadStats 执行，gl/gm 覆写仅靠末尾 PipeBarrier<MTE3>（-O2 下不足）→ t1 的 stats
   覆写抢在 t0 LSE 拷贝读 lseUb 之前 → 原始 sum 当 LSE 写出（数值实证的机理）。
2. **事件无子核分域**（#44.24 同款，divout 用固定 ID0/1/2/4/6）：另一 AIV 的同 ID Set
   可越权放行本核 wait——修 1 之后此缺陷立即成为新窗口。
修复（splitb_divout.hpp）：① Wait<MTE3_MTE2> 移至 operator() 的 LoadStats 之前（首轮
由 kernel init 预置 MTE3_MTE2 ID0/2 满足）；② 全部事件改 evId = 2×GetSubBlockIdx()
（AIV0→0、AIV1→2；类型标志位空间独立，闭合对与跨 tile 共用安全，softmax 先例）。
0 行 stub 路径的 set 同步分域。待验证：-O2/-O0 × B=2/4 全绿。

**#44.39**｜**【勘误+参考背书】FAInfer 的 gP 从不复用 S 区——解耦即参考设计（2026-08-24）**：
用户问 FAInfer 怎么处理 gP/gS。查证 mha_fwd_kvcache.cpp:160-169 + flash_api.cpp:556-565：
workspace 分段 [LseFD|OFD|gS|gP|gOTmp|gOUpdate]，**每 stage 产物独立区段**；每核
128×512×流水深度 3，S 系数 4B、P 系数 2B（fp16 减半直接体现在区段尺寸，无对齐应对）。
P 布局紧凑连续（online_softmax.hpp CopyPUbToGm 行距=colsPad，无洞）；PV LayoutP 同款。
FAInfer 的复用是**跨 tile 槽位轮转**（PRELANCH_NUM=3，QK(t+2) 覆写 S(t) 槽，flag 排序），
读写从不落在同批同字节窗口——用内存买断这类问题。
- **勘误 #44.23 叙事**："照搬 FAInfer 的 in-place 覆写"为误记——in-place 是 SplitB 移植时
  自创的省 GM 优化，FAInfer 无此设计，故参考实现从未踩过跨 AIV 竞争坑。
- **决策背书**：当前解耦布局 = FAInfer 哲学（独立 P 区/连续/零洞/零同步）；链式回归的
  优先级降级为"仅 S5 实测 GM 成瓶颈才考虑"（stride-2 方案 #44.38 留档备查）。

**#44.38**｜**回归链式的正确修法：跨距对齐（stride-2）链式，零跨 AIV 同步（2026-08-24）**：
用户否决跨 AIV 屏障方案（性能），追问无同步的修复。解耦布局 B=4 复测亦全对（t52_full_t4_2）。
设计：**P[t≥1] 写 S[t-1] 死区时行距用 2×colsPad（对齐 S 的 fp32 行栅格）**——P 行 r 压在
S 行 r 的前半字节。则 AIV_s 的 P 半区（行 [sR/2,(s+1)R/2)）只落进自己刚读完的 S 行区：
- 安全性构造性成立：tile t 的 P-copy MTE3 在本 AIV 标量流中晚于 tile t-1 compute 的
  WaitFlag<MTE2_V>（＝自己的 S(t-1) MTE2 已完成）→ 自己的写追不上自己的读；对方 AIV
  从不读我的行区（行对半拆分字节不交）。**零新增同步、零运行时开销**。
- 对比事故布局：原链式 P 紧凑排（stride=colsPad）→ P 行 16-31 落 S 行 8-15（对方读区）。
- 内存：回到 +1 slot/批（大 T 显著优于解耦的 +T slots；T=2 时两者仅差半个 slot）。
- 改动清单：CopyPUbToGm 加 dst 行距参数；PV LayoutP stride 同步（P[0] 保 half-slot
  紧凑则按 tileIdx 分支，或统一 stride-2 且 P[0] 开 full slot）；dump@200 按行距读；
  host perBatchF 回 T×perTileF+pScratchF。`[需 NPU 验证]` LayoutP ld=2×colsPad 的
  MTE1 fractal 装载。验证矩阵：-O0/-O2 × B=2/4 × sm-only/full + PRE-S 三时点一致。
补充（用户问 PV 影响）：stride-2 下 P 为**两个连续半块**（偏移 0 与 A/2，各 R/2 行），
仅拼接处有洞；正确性/数值零影响（ld>n 常态，LayoutS 同款）；代价 = P 装载 DMA 事务
约 ×2（P 为最小操作数，S5 实测）。**不可能性结论**："P 连续 + 零跨 AIV 同步 + 复用
死区"三者不可兼得——读区按行对半（R/2 行×4B），P 半区仅 R/2 行×2B，两个 P 半区总落在
同一读区内，下半区字节上必入对方读区（按 tile 奇偶换行区分配亦然，已证）。空洞是
零同步复用的固有代价；若 S5 不可接受则回屏障或维持解耦。

**#44.37**｜**【根因定位】链式 P 的跨 AIV 写读竞争——字节级+位级双重实锤（2026-08-24）**：
用户追问"为何解耦后 P 对、之前错在哪"。代码推演（splitb_softmax.hpp :463-481 行分摊 +
[SOFTMAX_DEBUG] 实测 rowNumTile=64→每 AIV 单次 16 行 2KB MTE2）：
- **布局算术**：P[t≥1] 以 fp16 写入 S[t-1] 区，半行距映射 P 行 r ↔ S 行 r/2（P 行偏移
  r×colsPad×2B、S 行偏移 k×colsPad×4B）。AIV1 的 P(t1) 行 16-31 落在字节 [1024,2048)
  = **S(t0) 行 8-15 = AIV0 那次 2KB MTE2 读的后半程**——与错误位置（b1 t0 s8-15、
  仅 tile0、行 0-7 恒安全=映射 AIV0 自写半区、tile1 的 P 恒好=无人覆写）**精确重合**。
- **位级指纹**：fp16(1.0)=0x3C00 作 fp32 高半字 → 0x3C000000=2⁻⁷=0.0078125 ≈ 实测
  max 0.007825；P(t1) 行内峰值≈1 打满高半字 → 整行读出≈0.0078 均匀值 → exp(x−max)≈1
  → sum≈31.77 → P≈0.992188——**三个指纹数全部由"P 字节被当 S 读"定量导出**。
- **排除法**：UB 残留类机制不依赖 GM 布局，解耦不可能修复——但实际修复了 → 只剩
  GM 字节覆写。stats 同坏 → 计算侧（非事后覆写）；三时点 P 不变 → 无后续写（#44.34）。
- **被证伪的前提**：链式安全论证"softmax 顺序执行保证 S[t-1] 读先于 P[t] 写"只在
  单 AIV 自链成立（#44.24 evId 域）；P[t] 须等两个 AIV 的读完成，跨 AIV 写读顺序不存在。
  sm-only 对称起步天然大余量；full 模式 VEC 流水态（DO(b0) 残留/队列占用）制造偏斜命中
  （b1-only 吻合"首个 DO 之后"；B=4 时 b2/b3 干净为待解细节）。
- **回归链式的正确修法**：P[t] 写 S[t-1] 前加跨 AIV 屏障（两 sub-block 互等对方
  MTE2_V），只还原布局不够。遗留：①控制组（去 PRE 探针重跑 -O0）封 attribution；
  ②-O2 专属 LSE bug（b1 h0 s16-31）另案。

**#44.36**｜**【主 bug 修复确认】P/S 解耦后 -O0/-O2 双双全对；残留 -O2 专属 LSE bug（2026-08-24）**：
t54 解耦布局两轮（用户跑）：**-O0**（t54_decouple.log）P/max/sum/O/LSE 全对——LSE 经
stats(340) 反推 ln(sum)+max 与 dump(451)/ref 三方逐值一致（脚本汇总曾误读，-O0/-O2 日志
张冠李戴过）；**-O2**（t54_decouple2.log）P/max/sum/O 全对，主指纹（b1 t0 s8-15）消失。
- 主修复归因基本成立（-O0/-O2 两套时序都过 → 非探针时序扰动），**待控制组定案**：
  注释 PRE 探针（860 系）重编译跑 -O0，仍全对 → 铁案。
- **新暴露第二 bug（-O2 专属）**：LSE 仅 b1 h0 s16-31 错（16/128，值 1.528→1.157，
  与 t53 链式时 s16-31 的错误值完全相同——t53 时代即存在，被 s8-15 大指纹掩盖）。
  O 与 GM stats 全对 → divout 的 sum 正确，错在 LSE 计算/写出路径（AIV1 行区间，
  疑 -O2 的 ln 向量化/Stats 读回 UB 时序）。下一步：读 splitb_divout.hpp LSE 路径。
- 归因口径：隔离的是**地址区间**（gP/gS 同一 workspace 两视图），非独立指针 →
  链式"P[t]→S[t-1] 死区"的写以硬件可见方式干扰了同批 S 读，机制待查（回归链式前必修）。

**#44.35**｜**P/S GM 全解耦（临时调试布局）+ 三时点 gS 探针（2026-08-24）**：
用户决策：为剔除"P 借 S 死区"变量，P 改完全独立 GM 空间——`perBatchF = T×(perTileF+pScratchF)`，
P[t]→批尾第 t 槽（GetPHalfIdx 单点改）；S 自 QK 写出后**永无人覆写**（任意时点 dump gS
都是干净观测）。**修好 full 流水 bug 后必须回归 #44.23 链式**（省 (T-1)×pScratchF/批）：
还原点在代码内搜 `#44.35`（workspace 布局注释 + GetPHalfIdx 两处）+ memory
splitb-p-gm-layout.md；softmax 的 MTE2_V 事件链未动，回归零改动可用。
三时点 gS 探针（每 tile dump rowNum×colsPad fp32）：**PRE=860+b\*10+tile**（段2 wait
qkReady 后、SM 前——"SM 将读到的 S"的直接观测）、**PPV=810**（段3 后金丝雀）、
**FIN=840**（段4 后金丝雀）。判读：①860 坏（b1 t0 s8-15≈0/垃圾）→ S 在 [QK 完成, SM 读]
窗口被写坏（full 语境）；②860 对而 P@200 仍坏 → GM 没问题，错在 SM 的 GM→UB 搬运/事件链
（转 splitb_softmax.hpp 读路径）；③三时点首个异物即凶手段；④加探针后 P 变全对 →
读侧时序竞争实锤（观测者效应本身是证据）。

**#44.34**｜**t53 三时点 P dump：三点数值一致 + 指纹自洽定案（2026-08-24）**：
200（SM 后）/810（PV 后）/840（DO 后）三个时点的 P **数值完全一致**，恒为 b1 t0
s8-15 错 → PV/divout 无任何追加写，P 一次写错后保持不变（#44.33 的 Fixpipe 越界猜想
对 P 不成立）。**指纹自洽**：坏行 max=0.0078、sum=31.766≈32×0.9927、P=0.992188 恰为
"该行 S≈全 0"输入的正确 softmax 输出；LSE=3.466227=ln(31.766)+0.0078 严丝合缝 →
**不是 P 被写坏，而是 SM 拿着近似全零的 S 行算出了自洽垃圾**——错误在读到的 S 本身
（GM 里的 S 就错，或 GM→UB 搬运/事件同步错）。结论：后续探针目标从 gP 改为 gS。

**#44.33**｜**用户七组对照实验：错误与段3/4 执行强相关，恒在 b1 t0（2026-08-23）**：
①②③ softmax-only B=2/4/6：S/P/max/sum 全对（任意批数都干净）。
④⑦ full B=2/4：S✓，P✗ 恒在 **b1 t0**（0.992188 老指纹）；B=4 时 b2/b3 干净。
⑤ 混合（b0 full、b1 sm-only）：错误仍恒在 b1 t0——**b1 自己没跑段3/4 仍坏**。
⑥ 混合（b0 sm-only、b1 full）：错误仍恒在 b1 t0。
用户假设：GM 空间冲突（类似 #44.23 S/P 复用），段3/4 某中间数据地址与段1/2 混用。
公式层核验：链式 P[t]→S[t-1] 与 OTmp[t-1]/stats[t-1] 子区不重叠（同 tile 块内
不同偏移）；P[0]→批尾 scratch 独占。**公式层无冲突 → 嫌疑转向硬件级写入粒度**：
copyL0CToGm（Fixpipe）按 L0C 全 tile 对齐写——若实际写入行数/宽度超过
rowNum×dPad 的软件假设（如按 fractal 16×16 或 128 行对齐补齐），OTmp/stats 写
可能越出 tile 块进入相邻区（批尾 scratch 恰在最后 tile 块之后！）。
待验证：PV 的 OTmp 实际写入范围（dump tile 块后 256B 边界外区域在 full 前后对照）。

**#44.32**｜**t52 全流程 vs t51 softmax-only 对照定案（2026-08-23，供用户分析）**：
同二进制、同输入、唯一变量 = 段3/4 是否运行：
- softmax-only（t51_2）：S✓ P✓ max✓ sum✓（双批全部 tile）
- 全流程（t52）：S✓ **P✗（b1 t0 s8-15 = 0.992188）** max✗（同位置 = 0.007825 恒定）
  sum✗（= 31.766）→ OTmp/O/LSE 连带错
关键事实（P dump 代码位置 mha_fwd_splitb.cpp:466-487）：**P dump 在段2 tile 循环
之后**（PipeBarrier<MTE3> 后读 GM），即 t0 与 t1 的 smEpilogue 都完成后才 dump。
P(b1,t0) 在 dump 时已坏 → t0 的 smEpilogue 写入的 P 就是坏的（t1 的链式写去
S[0] 死区、不碰 t0 的 scratch 槽，排除"后被覆盖"）。
指纹解读：max s8-15 = 0.007825 **8 行恒定**（若为 lmUb 残留应逐行不同）+
P = exp(−0.00784) → RowMax 第 2 级归约对第 2 个 repeat-8 组产出同一异常值，
或 S 载入 rows 8-15 读到的旧数据恰有此最大值。
**全流程 vs softmax-only 的结构差异（当前代码实证，qkReady 已被磁盘编辑回退为
批级：set 在段1 循环后 :436 / wait 在段2 循环前 :441）**：
① VEC 侧 SM(b1) 之前紧邻 DO(b0)（softmax-only 无）——DO(b0) 用过 tvUb/gmUb/glUb
（与 SM 的 tvUb/lmUb/llUb 同区或邻近）+ MTE3_MTE2(6)/MTE2_V(0)/V_MTE3(0) 事件；
② CUBE 侧 QK(b1) 之前紧邻 PV(b0)（时序推迟，QK(b1) 的 Fixpipe 写 S 与 VEC 的
MTE2 读更近）；③ pvReady/softmaxReady 交叉流量。
候选机制（按当前证据排序）：
庚-1 S 跨核可见性窗口：QK(b1) Fixpipe 落盘（FIX flag 后）→ VEC MTE2 读同一 GM，
FIX 完成语义是否含"对 VEC 可见"未证；rows 0-7 可见 / 8-15 未见的粒度吻合。
庚-2 DO(b0) 的 UB/事件残留干扰 SM(b1) 第一个调用的 RowMax 第 2 组。
建议用户的判别实验（不必改 kernel）：A. 段2 的 qkReady wait 后加
PipeBarrier<PIPE_MTE2>+PipeBarrier<PIPE_V> 延迟读；B. dump S 于 smEpilogue 调用
前一刻（stage2 循环内）对照段1 dump——直接看 S 在读时刻的可见性。

**#44.31**｜**比对器修复：Block 子串静默丢记录 + 自适应长度（2026-08-23）**：
① t51 desc=331 丢失根因：`[SOFTMAX_DEBUG] ... rowActualThisSubBlock` 行含子串
"SubBlock"→"Block"，命中 parse_log 第一分支被当块头行——**未闭合记录被静默丢弃**。
修复：该分支先 append 再重置。t51 复验：S/P/max/sum 全 ✓（OTmp/O/LSE 因段3/4
未跑为 ✗，正常）——与用户人工核对一致，#44.30 结论获得工具背书。
② check() 改自适应长度（dump 可短于 ref=AIV0 半行版或等于 ref=全行版，比对公共
前缀）——用户已把 S/P dump 改全行（1024），比对器不再依赖固定长度假设。
③ 教训：**解析器对"含关键字子串的无关行"的静默分支是丢失记录的温床**——
每个丢弃分支都必须先闭合在途记录。

**#44.30**｜**【判别完成】softmax-only 全对 → bug 锁定段3/4 交互（2026-08-23）**：
t51（softmaxOnly 真正生效：段1 QK + 段2 softmax，段3 消费 softmaxReady 后 continue、
段4 整段跳过、dump 仅 S/P/stats）结果：
- **S：全对**（4 tile）✓
- **P：全对**（4 tile）✓ —— 链式 P 方案在全流程 kernel 内验证通过
- **max/sum：全对**（用户人工核对全部 batch/head 均与 ref 一致；比对器显示的
  "b0 tile1 全 0"系 dump 记录 331 被解析器漏掉——大单行 + ShapeInfo 行交互，
  已改为显式 WARN 标记缺失而非填 0 假错）
**结论：QK→softmax 链路（含链式 P、S 跨核可见性）完全正确；b1t0 s8-15 指纹
的根因在段3（PV）或段4（divout）与 softmax 的交互。**
对照 t50（全流程）：P/stats 在 softmax-only 下对、全流程下坏（b1t0 s8-15）
→ **PV 读到了未写完/被覆盖的 P**。头号嫌疑（机制己）：段3 批级
WaitFlag(softmaxReady)（一次）+ 段2 逐 tile 双 AIV 各 set（每批 4 次 set）——
"任一置位即通过"语义下 PV 可在某 AIV 的某 tile P 拷贝未完成时即开始读 P。
下一步：段3 改逐 tile wait（循环内每次 PV 调用前 wait 一次，消耗 4 set/batch
对齐 4 wait/batch），或恢复 PV 内部 WAIT_SOFTMAX=true 的逐调用等待。

**#44.29**｜**编译：注释断行裸 `#ifdef` 被预处理吞（2026-08-23）**：
段4 门控注释跨行时第二行行首落了裸 `#ifdef`（首行 `//` 注释未延续）→ 预处理
当真指令 → "unterminated conditional directive"。修复：注释改写不跨指令词。
教训（写设备代码注释的硬规则）：**多行注释每行都必须以 `//` 开头；绝不出现
行首裸 `#if/#ifdef/#endif` 字样**（哪怕在语义上是注释内容）。

**#44.28**｜**softmaxOnly 段3 门控用户方案（消费后 continue）+ 条件反转修正
（2026-08-23）**：
用户方案：段3 处 `WaitFlag(softmaxReady)` 后 **softmaxOnly 则 continue**——先消费
flag 再跳批。优于我方"整段包裹不消费"设计：softmaxReady 每批 set 数=wait 数，
**任意 B 无累积风险**（我方累积 set 有 MAX_REVERSE_DEPTH=15 上限）。
用户初版写成 `if (!softmaxOnly) continue`（反转：全流程跳 PV / softmaxOnly 跑 PV）
——已修正为 `if (softmaxOnly) continue`。
softmaxOnly 模式最终 dump 集：段1 的 S + 段2 的 P + dump 块的 stats（OTmp/O/LSE
被门控跳过）。

**#44.27**｜**softmaxOnly 门控结构修正（2026-08-23，用户指出）**：
初版把 `!softmaxOnly` 塞进段3/段4 的 **for 循环条件**——但段3 循环外还有
批级 `CrossCoreWaitFlag(softmaxReady)` 与 `CrossCoreSetFlag(pvReady)`、段4 循环外
有 `CrossCoreWaitFlag(pvReady)`，这些**不被循环条件覆盖**：softmaxOnly 下 CUBE 仍
消费 softmaxReady、仍置位 pvReady（无 PV 却发"PV 完成"），VEC 仍等 pvReady——
靠错位 set/wait 互相抵消才未挂死，语义完全错误。
修复：段3/段4 各用显式 `if (!softmaxOnly) { 整段 }` 包裹（含批级 wait/set/循环/
printf）；段4 的门在 dump 块**之前**闭合（stats/P dump 两种模式都要执行）。
注：段2 在 softmaxOnly 下仍 set softmaxReady（无消费者累积；B=2×2tile×2AIV=8
< MAX_REVERSE_DEPTH=15，调试样例安全；更大 B 需再门控 set）。
VEC 分支模拟括号平衡 ✓、全文件平衡 ✓。

**#44.26**｜**playground harness 双 bug 修复后全绿 + softmaxOnly dump 门控（2026-08-23）**：
修复复现器两个 harness bug：①P-scratch 与下一批 tile 区重叠（原 (r*B+b+1)*PER_BATCH
布局——b0 的 P[0] 砸 b1 的 S[t0]；全流程 kernel 无此问题，scratch 在 perBatchF 内部）；
②双 AIV 批门控错误（blockIdx==0||bo==0 使 b1 只有 AIV0 处理）。
**修复后双 AIV + 双批 + 链式 P 全绿**（2 轮 × 2 批 × 2 tile，P/max/sum 全 0 错）。
**至此 softmax 类本身（含链式 P）在单 AIV / 双 AIV / 单批 / 双批 全部形态验证通过**。
推论：全流程 b1t0 s8-15 指纹的输入侧差异只剩：S 由 QK Fixpipe 写（playground 是
host 预写）+ 跨核逐 tile flag + PV/divout 交互——由 softmaxOnly 判别。
kernel softmaxOnly 补 dump 门控（段3/4 未运行不 dump OTmp/O/LSE，仅 stats；用户指出）。
教训：**复现器的 harness bug 会制造"修复无效"的假象**——两个 bug 都在"被测组合"上，
此前双批数据全废；harness 布局必须与被测 kernel 布局逐项核对（scratch 位置尤其）。

**#44.25**｜**t50 后复盘：softmaxOnly 链曾被磁盘编辑回退 + 事件分域无效（2026-08-23）**：
t50 复验发现：①softmaxOnly 三件套（tiling 字段/host env/kernel 门控）已被磁盘编辑
回退 → t49/t50 的"softmax-only"实为全流程（[OUT] 提示只是 python 侧），隔离实验
从未生效；②事件分域（evId=pp+2×subIdx，已编译在内）后全流程指纹依旧（b1t0 s8-15）
→ HardEvent 共享理论不充分或非此机制。已重新补回 softmaxOnly 三件套。
复现器追加数据：双 AIV + 链式 P 下首批（b0）两 tile 完美——与全流程 b0 全对一致；
复现器 harness 自身出现 scratch 槽与下批区域重叠的布局 bug（线性布局未给每批留
scratch），暂停复现器路线。
当前事实基线（全部实证）：S✓；b0 全对；**b1 t0 s8-15（AIV0 第 2 个 repeat-8 组）
坏**（max=0.007825≈exp 残留、P=0.992、sum=31.77）；链式 P 与事件分域均未治愈；
单 AIV 形态不复现。下一步：softmaxOnly 真正生效后重跑判别。

**#44.24**｜**t48 全流程复验：P 链式修复后 S✓ 但 b1 tile0 s8-15 指纹仍在（2026-08-23）**：
正式修复编译后全流程（B2/S32/H2/D64）：S 全对 ✓、P/max/sum 在 **b1 tile0 s8-15**
仍 0.992188/0.007825/31.766（与 t44 修复前逐位相同）→ 链式修复解决了复现器暴露的
P-copy 丢失机制，但全流程 kernel（**双 AIV 并发**形态）还有一个同指纹的独立 bug。
工具修复：parser 现跳过 ShapeInfo 前置元信息行（shape>dumpSize 时打印在数据前，
曾被误判为记录终止符 → S/P/OTmp 假"空记录"）。
指纹再确认：S 对 + AIV0 半区的第 2 个 repeat-8 组（s8-15）坏 → 腐败发生在
S-load(MTE2)→RowMax 之间：AIV0 的 lsUb s8-15 在 RowMax 时未就绪。
**头号新嫌疑（双 AIV 专属）**：AIV0/AIV1 并发跑同一事件链（MTE2_V/V_MTE3 等
EVENT_ID0 自配对）——若 HardEvent 为核级共享（非子核独立），AIV1 的 Set 可提前
释放 AIV0 的 Wait → AIV0 的 MTE2 半程即被读（后落地的 s8-15 读旧值）。FAInfer
同结构生产可用 → 或事件确为子核独立、或其时序不撞；我方 b1t0（第 3 次调用、
流水热身后）稳定撞。
下一步实验：①softmax-only 模式（已内建）看剥离后指纹是否仍在；②类内事件 ID
改子核分域（id += 2×subIdx）对照。
**①已执行（t49）**：softmax-only 指纹仍在（P b1t0 s8-15 = 0.992188）→ bug 锁定
softmax 段内、双 AIV 并发形态（S✓、单子核复现器✓）。
**②已实施**：splitb_softmax.hpp 全部 5 处自配对链（MTE2_V/V_MTE2/MTE3_V/V_MTE3）
的 id 改 evId = pingpongFlag + 2×GetSubBlockIdx()（AIV0 用 0/1、AIV1 用 2/3）；
mha_fwd_splitb.cpp 预置补 MTE3_V(EVENT_ID3)（V_MTE2 0-3 原有 ✓）。待编译验证。

**#44.23**｜**【根因定案】S/P 的 GM in-place 复用在批粒度流水下非法（2026-08-22，用户定性）**：
FAInfer 照搬来的设计"P 写回 S 区原地（fp16 视图覆写 fp32 S）"在其 **tile 级流水**中合法
（QK 写 tile S → softmax 立即消费并原地写 P，生命周期紧凑、由逐 tile qkReady 衔接）。
但 SplitB 是**批粒度流水**（段1 写全部 tile 的 S → 段2 循环逐 tile softmax）：
softmax 循环内 MTE2 读 GM[S区] 与 MTE3 写 GM[同一批字节]（P 原地）在循环迭代间
产生同地址跨 pipe 读写冲突 → 确定性 P 写入丢失（-O0 同指纹，非 MTE 硬件问题、
非时序竞态——是空间设计错误）。
实证链（#44.20-22 剥离复现器）：源 lpUb 对、dst 地址对、参数对、stats（独立区域）
落地、P（in-place 区）消失。
**修复方向（用户决策）：GM 排布重设计——P 独立区域，不再与 S 复用。**
**✅ 复现器验证通过（2026-08-23）**：playground 改 P 独立区后，被处理 batch 的 P 两半
全部 0 错（此前 h1 ~506 错）——根因实锤。剩余 stats"错"仅为 host 读回未同步新偏移。
**链式省空间方案验证（2026-08-23）**：P[t]→S[t-1] 死区（softmax[t] 启动时 S[t-1]
读已证明完成）+ t=0 用批尾 scratch 槽（+1 P 槽/批而非 +T 槽，S 区 +50%→+6%@8tile）。
复现器 r0 b0 两 tile **P/max/sum 全 0 错** ✓（r1 有残留待查，疑 host 读回/b1 未处理伪影）。
r1 残留=未处理 b1 伪影（r1 b0 两 tile 亦全 0 错，链式方案完整验证 ✓）。
**正式修复已完成（2026-08-23）**：mha_fwd_splitb.cpp——布局注释重写（P 链式独立区）、
pScratchF 成员、perBatchF 加批尾 P-scratch、新增 GetPHalfIdx(batchBase, tileIdx)
统一寻址（t=0→scratch、t≥1→S[t-1]，返回 half 索引）、段2 softmax/段3 PV/P-dump 三处
gP 基址全部改走 GetPHalfIdx；splitb_host.cpp——pScratchF 常量 + perBatchF 公式同步。
大括号/预处理平衡 ✓。已存长期记忆（memory/splitb-p-gm-layout.md，含 softmax tiling
改动前的重验警告）。
待办：① mha_fwd_splitb.cpp 三视图/段2 段3 gP 基址 +
splitb_host.cpp workspace 公式（新 perTileF = s1AreaF + pAreaF + oAreaF + stats）；
③全量回归。
新每 tile 块布局（float 计）：[S 区 128×colsPad | P 区 128×colsPad halfs（=64×colsPad
floats）| OTmp 区 | stats 区]；perTileF 相应加大。消费点同步：splitb_host.cpp workspace
公式、kernel 三视图与段2/段3 的 gP 基址、dump 探针。

**#44.23**｜**编译：aicore 代码禁用 fflush/stdout（2026-08-24）**：
mha_fwd_splitb.cpp 加调试 printf 时顺手写了 `fflush(stdout)`（host 习惯），编译报
`call to [host] function from [aicore] function` + `global variable 'stdout' is
not allowed in aicore function`。根因：设备侧不存在 libc，`fflush`/`stdout` 均
host-only；`AscendC::printf` 输出走调试通道，**kernel 结束 host 同步时自动刷出**，
设备代码里本就无需（也无法）flush（代码 809 行注释已写明此约定）。修复：删 14 处
` fflush(stdout);`，printf 语句全保留。注意 host 侧（splitb_host.cpp、
fwd_splitb_dispatch_impl.hpp 等）的 fflush 合法不动。经验法则 #6 的延伸：不只 printf
要换 `AscendC::printf`，**任何紧跟它的 host 侧 flush 习惯也要去掉**。

**#44.22**｜**关键判别：-O0 -g3 下指纹完全相同（2026-08-22，用户提议）**：
复现器以 -O0 -g3 编译运行：**逐位相同的失败**（2744 处、同 h1-P 丢失指纹）。
→ **这不是 -O2 优化/时序/竞态问题，是确定性的逻辑 bug**（与编译模式无关）。
诊断方向修正：排查 CopyPUbToGm 的 DataCopy 参数/地址计算/布局传参在第二次调用
（base+512 floats、in-place S 区）下的确定性错误路径，而非同步问题。

**#44.21**｜**softmax bug 收敛至"P 拷贝在偶数次调用丢失"（2026-08-22，剥离复现器）**：
零并发串行复现（__vector__ 单子核 + half 循环，3 次运行逐位一致）确定性指纹：
- **每次 tile 的第 1 次调用（行 0-15）完美**；
- **第 2 次调用（base+512 floats，行 16-31）：stats（max/sum）正确落地，但
  CopyPUbToGm 的 P 拷贝从未写入**（P 区 = 原始 S 字节，fp16 视图呈 [0,~1.4] 交替）。
- 同 base 对照实验：第 2 次调用的 P 写到了 base（值错）→ 链路执行但内容/地址错。
- UB 内部 dump（DumpUb）会扰动时序（Heisenberg），不可用于此 bug 的内部定位。
- b1"全坏"为 bCount=1 串行 DBG 未处理所致（修正此前误读）。
结论：bug 在 CopyPUbToGm（lpUb→gP 的 MTE3 DataCopy）在连续第二次调用时的行为。
下一步：①只对 P 拷贝换 DataCopyPad（带显式参数）或改用 3 参重载对照；②检查
lpUb 的 LP_UB_TENSOR_OFFSET=64KB 与 lsUb 槽 0-64KB 的布局在第二次调用时是否
存在 MTE2（载入行 16-31 到 lsUb[0]）与 MTE3（读 lpUb[0]）的跨 pipe 事件缺口
——类内 V_MTE3 自配对只保证 V 完成，未保证"上一调用的 MTE3 已离开 lpUb"。

**#44.20**｜**softmax 独立复现器建成：bug 锁定在 softmax 类的跨调用状态（2026-08-22，
用户主导剥离方案）**：
工具 debug/test_softmax_playground.cpp（独立 ACL 程序，~秒级编译，无 torch/无 QK/PV/DO/
无跨核 flag）：host 构造 S 写 GM → AIC-only 剥离 kernel 复刻段2 调用形态 → DumpTensor
+ host 自带 fp32/fp16 参考逐元素比对。编译：bisheng -x asc --npu-arch=dav-2201
--cce-auto-infer-kernel-type=false -O2 ... -lascendcl -lm（见文件头）。
关键踩坑（复现器构建过程中）：
① 无标注 __global__（含 CUBE/VEC 双 guard）被 bisheng 自动判 **MIX 型**（subNum=2 实证）；
  __vector__ 属性版 subNum=1——SplitBSoftmax 行分摊硬编码 /2，单子核下半区无人算；
② 外层手动 half 循环 + 类内 subIdx 分摊 = 双重切分（指纹全乱，曾误判"第 2 次调用必坏"）；
③ P 的 GM 布局：half 索引 = base*2 + s*COLS_PAD + j（×2 只作用于 float 基址）。
**终版指纹（100% 确定性）**：batch 串行 + 双 AIV：b0（第 1-2 次调用）全对；
**b1（第 3-4 次）起全 32 行坏**；后续轮次 b0 也出现 h0 行 8-15（repeat-8 第 2 组）错——
与全流程 kernel 的"第 3 次 smEpilogue 调用坏 + 行 8-15"指纹完全一致。
**结论：bug 在 splitb_softmax.hpp 的跨调用状态（UB 残留或事件链），非段间交互。**
下一步：在复现器内对第 3 次调用做 UB 级三点 dump（lsUb 载入后/RowMax 后/Exp 后，
各 16 行）定位坏点进入的环节。

**#44.19**｜**逐 tile 握手恢复 + softmax 剥离模式 + CalcExp 回退（2026-08-21，用户主导）**：
三项代码级修复：
① **恢复 FAInfer 逐 tile 握手**（用户此前已认可的方案，一直未实施）：段1 每 tile 后
  set qkReady（FAInfer :675）；段2 每 tile 前 wait、后 set softmaxReady（:729/:849）；
  段3 PV 内部等 softmaxReady（DispatchPolicyPV 第三参 WAIT_SOFTMAX 恢复 true，FAInfer
  原样）+ 每 tile 后 set pvReady（:904）；段4 每 tile 前 wait（:915）。批级一次的
  set/wait 结构（#39/#40 自创组合）正式回退——它使双 AIV 同 id set 的"任一通过"
  语义下 PV 可早读未完成的 P（#44.9 分析）。三处 DBG PIPE_ALL 注入同步移除。
② **softmax 剥离模式**（用户提议）：tiling 新增 softmaxOnly（env
  FLASH_ATTN_SPLITB_SOFTMAX_ONLY）——只跑段1+段2（S 来自真实 QK），跳过 PV/DO
  （跨段 flag/时序全部排除），P/stats dump+比对链路复用。判定：剥离模式下仍错 →
  bug 在 softmax 代码内部（参数/同步/UB）；剥离模式下全对 → bug 在段间交互。
  脚本 --softmax-only 同步（跳过 OTmp/O/LSE 比对与 [OUT] 判定）。flag 平衡核算：
  softmaxReady 无消费者时 set ≤8 次 < MAX_REVERSE_DEPTH=15 ✓；drain 全部预置 ✓。
③ **CalcExp 的 Brcb→Duplicate 改动回退**：Duplicate 单 repeat 写 64 元素（8 块），
  逐行调用会覆盖后续 7 行槽位、破坏与 Sub(src1RepStride=4) 配对的 tvUb 布局——
  错误改法，已回退为 FAInfer 原样 Brcb 并注明不可替换原因。教训：**改向量指令前
  必须核对其 write 布局与下游 read 布局的配对**（Brcb 的块广播布局是约定俗成的
  接口，不能按"语义等价"替换）。

**#44.18**｜**DumpTensor 实际缓冲阈值 ≈128KB（远小于文档 1MB）+ s8-15 坏组分析
（2026-08-21）**：
Sq=64 实验 dump 大量缺失（中间段丢、头尾在）：实测 t44（Sq=32）每轮 82KB 全齐、
t45（Sq=64）每轮 149KB 丢中间 → **实际缓冲阈值在 (82,149]KB**（疑似 128KB），
文档的 1MB 不可信。修复：S/P/OTmp 只 dump AIV0 行（rowNum/2，坏行 s8-15 在半区内）、
stats 256→192——每轮 ~80KB。同步比对器（ref 只取 AIV0 行）。
同期主证据（Sq=32，b1 tile0）：坏行恒 s8-15（AIV0 的 16 行后半 8 行 = 行串行指令
repeat-8 的第 2 组）；max 坏值 2⁻⁷ 恒定、P 坏值 0.992 恒定（−ln(0.992)=0.0078≈max）
→ P=exp(S−max) 变成 exp(−max)，即 lsUb 行 8-15 在 Exp 前被清零/减成 0；S 行 8-15 正确、
b0 全对、b1 tile0（kernel 内第 3 次 smEpilogue 调用）坏、B=1 恒 PASS。
候选：V 指令 repeat-8 第 2 组坏 vs MTE3 拷贝 8+8 拆分第 2 条丢。下一步：smEpilogue
内三点 UB 探针（RowMax/CalcExp/Cast 后各 dump 行 8-15，UB 对而 GM 错 = 拷贝坏）。

**#44.17**｜**编译：DumpTensor 带 ShapeInfo 的初始化列表陷阱（2026-08-21，用户发现）**：
用户给四段 dump 加 ShapeInfo（shape 化打印）后编译报错：`cannot convert initializer
list argument to 'const uint32_t *'`。根因：文档示例 `ShapeInfo(2, {8, 8})` 在主机侧
合法（初始化列表退化为指针），但设备编译器（BiSheng）不接受——与 #12（catlass 引用
参数不绑临时值）同族：**设备代码禁用初始化列表/临时值绑指针**。
修复：具名数组变量（`uint32_t shapeD[2] = {...}; AscendC::ShapeInfo infoD(2, shapeD);`）
+ 独立作用域块；O/LSE 两处同理。六处全部改毕。
教训：**文档示例是主机代码，设备侧一切"传临时/列表给指针参数"都要改写为具名变量**。

**#44.16**｜**分阶段验证的收束结果与工具链修复（2026-08-21）**：
B=1 压缩样例首轮完整比对：**S/P/max/OTmp/O/LSE 全 ✓、sum 全 0**——但 O/LSE 正确
（O=OTmp/sum 若 sum 真为 0 必炸）→ 比对器读错位置。根因：stats GM 布局是
max[0..rows) + **sum[128..128+rows)**（行距 ROW_NUM_MAX=128），kernel 曾连续 dump
2×rows 只覆盖 max+未写区（全 0）。修复：kernel 改 dump stats 整块 STATS_LEN=256、
比对器 sum 偏移 128。工具链另一批修复（#44.15 内）：
- kernel 末 dump 刷不出 → 移入 batch 循环（段4 后，运行中时机）——顺带解开 t28-t33
  的 411/412 从不输出悬案；
- 段4 #endif 曾误删（用户发现少括号）→ 补回；
- redirect_stdout/同进程捕获抓不到异步刷出的 device dump → 改两步式（跑 kernel 由
  用户 tee 收集 + --log 离线比对）；
- 脚本全参数化（--batch/--sq/--sk/--heads/--kv-heads/--dim），Q/K/V 公式自适应
  （任意 shape exp 输入 ≤0 不溢出）。
当前状态：B=1 样例 max_err=0.0000 PASS——错误未复现；工具链已就绪，下一步逐步
放大 shape 找触发边界（用户主导）。

**#44.14**｜**临时禁用 ScaleS 使中间结果更直观（2026-08-21，用户要求）**：
#44.12 方案的小调整：kernel splitb_softmax.hpp operator() 中 ScaleS 调用临时注释
（DBG 标记），基准脚本 make_ref 同步 `s_sc = s_raw`（不乘 SCALE）。数值健全性已验证：
S_raw≤64 不溢出、P=exp(S−max)∈(1e-27,1]（fp16 下溢部分两侧一致）、sum∈[1.16,25.4]
跨行变化大（可辨识性更强）、O∈[17.8,32.8]、LSE∈[3.7,64.1]。调试结束后恢复 ScaleS。

**#44.13**｜**代码批判式复审发现（2026-08-21，用户主导逐行审阅）**：
用户亲自审阅 splitb_divout.hpp 发现：⑤ LSE 块 Brcb(tvUb[64]←lseUb) 与
SetFlag<V_MTE3>(EVENT_ID4) 之间缺 PipeBarrier<PIPE_V>——FAInfer rescale_o.hpp
:535-545 同款位置有（Brcb → InvalidLineLSEProcess → PipeBarrier<PIPE_V> →
SetFlag<V_MTE3>(4) → Wait）。#44.10 在此处误用 V_MTE2 自配对（方向错误 + 同样
Scalar 发射乱序），PipeBarrier<PIPE_V>（Scalar 等 V 排空）才是正解。后果：-O2 下
MTE3 的 LSE gather 可能读 Brcb 写入前的旧 tvUb → LSE 错。已修（用户补丁）。
经验：照抄权威时漏 PipeBarrier 的最常见位置 = V 指令序列的**最后一条与紧随的
SetFlag 之间**（FAInfer 的屏障常散落在函数中间，逐行对照才能发现）。

---
## 四、待解决（追踪中）

- **挂死收窄（2026-08-18）**：stage3 单核 ✓ 通过；stage0 全真实 4 核 ✗ 挂。
  待判别：debug 单核 stage=4（+PV）→ stage=5（+divout=全）→ 若单核全过则嫌疑=
  多核（workspace 越界/核间干扰/blockDim>1 的调度）。
- S-dump 数值判别（脚本索引已修 #25）。
- S4：causal/SWA mask、softcap tanh。

---

## 五、经验法则速查（从上述问题提炼）

1. 照搬 checklist：类型 → 对象 → **init → 事件预置** → 主循环 → 收尾（#17）
2. flag 握手所有消费者对称参与；且 set 前的真工作也要对称（行分摊而非 owner 分摊）（#15/#23）
3. 设备挂死定位用执行掩码二分，不依赖设备 printf（#16）
4. 向量指令 count：元素级=总元素/64；行串行=行数（建议分批 ≤16，FAInfer 验证区间）（#18/#20）
5. `__gm__` 指针只能解引用，不能转普通指针（#2）
6. 设备函数一律 `__aicore__ inline`（#9）；printf 用 `AscendC::printf`（#14），且不得跟 `fflush(stdout)`——host-only，输出 kernel 结束自动刷（#44.23）
7. CANN 9.0.0 API 形态先查重载/在本仓库找已编译先例（#10/#13）
8. catlass 引用参数传具名变量（#12）
9. 同进程无法切换 C++ 静态 env 开关（#7）
10. inline 定义放调用方可见的头（#3）
11. **任何"挂死/回归"结论前：换 device 复测 + 查残留进程**（#21，两次误导教训）
12. 数值错误逐位不变 ⇒ 二进制未变或上游主导 bug，先验证二进制再怀疑新改动（#22）
13. 调试设施自身先验证默认行为（stage=0 应打印/表现为"全真实"），工具 bug 会污染一切观测（#24）

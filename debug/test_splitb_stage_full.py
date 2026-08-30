#!/usr/bin/env python3
"""SplitB 分阶段全量验证（devlog #44.12，用户主导方案）。

样例：B=2, Sq=32, Sk=32, H=2, D=128, fp16 —— 可辨识结构化输入：
    Q[b,s,h,d] = (b+1)*(h+1)*(s+1)/256      （batch/head/token 三维可辨，fp16 精确）
    K[b,j,h,d] = (j+1)/32                    （kv 列可辨，fp16 精确）
    V[b,j,h,d] = (j+1) + (d+1)/128           （列 + 通道可辨）
  ⇒ S_raw[b,h,s,j] = (b+1)(h+1)(s+1)(j+1)/64（小整数乘积，肉眼可验）
  ⇒ O[b,s,h,d]   = m(b,h,s) + (d+1)/128，m∈[1,32]（每行常数偏移 + 通道斜坡）

kernel 侧（--dump 开启 FLASH_ATTN_SPLITB_DUMP；--multi-core 切多核；dump 全核可用，#44.46）：
    desc=100+b  段1末  S 整区（float 视图；S 会被 P 原地覆盖，仅此时可读）
    desc=200+b  段2末  同一整区（half 视图 = P 未归一 exp）
    desc=300    kernel末 workspace 整区（float 视图 = OTmp + stats 终态）
    desc=400    kernel末 O 全量（fp16）
    desc=450    kernel末 LSE 全量
  每 dump 前有 [SB-DUMP] printf 声明来源与布局参数。

本脚本三种用法：
  python debug/test_splitb_stage_full.py                 # 完整流程（默认 3 轮 kernel）
  python debug/test_splitb_stage_full.py --iters 5       # 指定轮数
  python debug/test_splitb_stage_full.py --print-ref     # 只打印基准各阶段结果（不跑 NPU）
  python debug/test_splitb_stage_full.py --log FILE      # 只解析比对已有日志
"""
import argparse
import os
import re
import sys

# ---- 测例与布局（默认值；全部可由命令行参数覆盖） ----
B, Sq, Sk, H, Hkv, D = 2, 32, 32, 2, 2, 128
ROW_NUM_MAX = 128                # stats 行距 = Q_TILE_CEIL
Q_TILE_CEIL = 128

# 派生几何（惰性：init_geom() 在参数确定后重算；模块级占位防 import 期报错）
TILES = []
N_TILE = 0
COLS_PAD = D_PAD = S1_AREA = O_AREA = STATS_LEN = PER_TILE = PER_BATCH = STRIDE_O = 0
SCALE = 1.0


def init_geom():
    """按当前 B/Sq/Sk/H/Hkv/D 重算全部派生几何（含 tile 复刻与 workspace 布局）。"""
    global G, SCALE, TILES, N_TILE, COLS_PAD, D_PAD, S1_AREA, O_AREA, STATS_LEN, \
        PER_TILE, PER_BATCH, STRIDE_O
    G = H // Hkv
    SCALE = 1.0 / (D ** 0.5)
    qnbt = (Q_TILE_CEIL // Sq) // 2 * 2 if Sq else Q_TILE_CEIL
    qnbt = max(1, min(qnbt, G))
    per_group = (G + qnbt - 1) // qnbt
    n_tile = per_group * Hkv
    q_sblk = Sq
    tiles = []
    for t in range(n_tile):
        kv = t // per_group
        ig = t % per_group
        q_start = kv * G + ig * qnbt
        qn_size = (G - ig * qnbt) if ig == per_group - 1 else qnbt
        tiles.append((t, q_start, qn_size, q_sblk, q_sblk * qn_size))
    TILES = tiles
    N_TILE = n_tile
    COLS_PAD = (Sk + 15) // 16 * 16   # RoundUp(Sk,16)
    D_PAD = (D + 15) // 16 * 16
    S1_AREA = ROW_NUM_MAX * COLS_PAD
    O_AREA = ROW_NUM_MAX * D_PAD
    STATS_LEN = 2 * ROW_NUM_MAX
    PER_TILE = S1_AREA + O_AREA + STATS_LEN
    PER_BATCH = N_TILE * PER_TILE
    STRIDE_O = H * D


def make_ref():
    """基准各阶段（量化语义与 kernel 一致：输入 fp16；P 在 DownCast 处 fp16 量化；
    sum 用 fp32 exp（kernel RowSum 读 fp32 lsUb）；OTmp = fp16P @ fp16V fp32 累加）。"""
    init_geom()
    import torch
    q = torch.zeros(B, Sq, H, D)
    k = torch.zeros(B, Sk, Hkv, D)
    v = torch.zeros(B, Sk, Hkv, D)
    qden = float(max(1, B * H * Sq))   # 归一化：任意 shape 下 Q∈(0, 2]、S_raw≤Sk
    for b in range(B):
        for s in range(Sq):
            for h in range(H):
                q[b, s, h, :] = (b + 1) * (h + 1) * (s + 1) / qden
    dv = torch.arange(1, D + 1, dtype=torch.float32) / float(max(1, D))  # 通道斜坡 (d+1)/D
    # K/V 按 kv 头平移（devlog #44.49）：GQA 下若 kernel 读错 kv 头，S/OTmp 将错位可检
    # （修复前 bug：k/v 曾用 H 建 tensor —— Hkv 恒等于 H，GQA 从未被真正测过）
    for j in range(Sk):
        for h in range(Hkv):
            k[:, j, h, :] = (j + 1 + h) / float(max(1, Sk))
            v[:, j, h, :] = (j + 1 + h) + dv
    q16 = q.half().float()
    # GQA 金标展开：kernel 输入保持 Hkv 头（ref["k"]/["v"] 原形状），仅基准数学用
    # repeat_interleave 按组扩展到 H 头（devlog #44.49；否则 H≠Hkv 时 matmul 崩）
    ke16 = k.half().float().repeat_interleave(G, dim=2)
    ve16 = v.half().float().repeat_interleave(G, dim=2)
    s_raw = torch.matmul(q16.transpose(1, 2), ke16.transpose(1, 2).transpose(-1, -2))  # [B,H,Sq,Sk]
    # DBG（2026-08-21 用户要求，临时禁用 ScaleS）：kernel 已注释 ScaleS，此处同样不乘。
    # 数值健全性：S_raw≤64 不溢出；max=64；P=exp(S−64)∈(0,1]；sum≈2-3；O≈30-32。
    s_sc = s_raw * SCALE   # 恢复 scale（devlog #44.42；调试期曾用 s_raw 对齐禁用的 ScaleS）
    mx = s_sc.max(dim=-1, keepdim=True).values                    # [B,H,Sq,1]
    p32 = torch.exp(s_sc - mx)
    p16 = p32.half().float()
    sumv = p32.sum(dim=-1)                                        # [B,H,Sq]
    otmp = torch.matmul(p16, ve16.transpose(1, 2))                # [B,H,Sq,D]
    o32 = otmp / sumv.unsqueeze(-1)                               # [B,H,Sq,D]
    o16 = o32.transpose(1, 2).half().float()                      # [B,Sq,H,D] 与 kernel out 同形态
    lse = torch.log(sumv) + mx.squeeze(-1)                        # [B,H,Sq]
    return dict(q=q, k=k, v=v, s_raw=s_raw, mx=mx.squeeze(-1), p32=p32,
                p16=p16, sumv=sumv, otmp=otmp, o16=o16, lse=lse)


def fmt_mat(a, cols=None, width=12, prec=6):   # 与 dump 打印精度一致（6 位小数）
    """矩阵按行打印（可截列）"""
    import torch
    t = a if isinstance(a, torch.Tensor) else torch.tensor(a)
    if cols:
        t = t[..., list(range(cols[0])) + list(range(cols[1], t.shape[-1]))] \
            if len(cols) == 2 else t[..., :cols]
    lines = []
    for r in range(t.shape[0]):
        lines.append("  s%02d | " % r + " ".join(
            ("%*.*f" % (width, prec, x)) if x == x else "%*s" % (width, "nan")
            for x in t[r].tolist()))
    return "\n".join(lines)


def print_ref(ref):
    import torch
    print("=" * 100)
    print("基准各阶段结果（B=%d Sq=%d Sk=%d H=%d D=%d, scale=1/sqrt(%d)≈%.6f）" % (B, Sq, Sk, H, D, D, SCALE))
    print("tile 几何：%s（每 tile: qStart/头数/每头行数/打包行 rowNum）" % TILES)
    print("workspace 布局：colsPad=%d dPad=%d s1AreaF=%d oAreaF=%d stats=%d perTileF=%d perBatchF=%d"
          % (COLS_PAD, D_PAD, S1_AREA, O_AREA, STATS_LEN, PER_TILE, PER_BATCH))
    for b in range(B):
        for h in range(H):
            print("\n----- S_raw[b=%d, h=%d]（kernel 段1 产物；= (b+1)(h+1)(s+1)(j+1)/64）-----" % (b, h))
            print(fmt_mat(ref["s_raw"][b, h]))
    for b in range(B):
        for h in range(H):
            print("\n----- P_unorm(fp16 量化)[b=%d, h=%d]（段2 产物 = exp(S*scale - max)）-----" % (b, h))
            print(fmt_mat(ref["p16"][b, h]))
    print("\n----- stats（max / sum，行 = tile 打包行）-----")
    for b in range(B):
        for (t, qs, qn, qsblk, rows) in TILES:
            head = "[%d..%d)" % (qs, qs + qn)
            print("b=%d tile=%d heads=%s：" % (b, t, head))
            print("  max:", " ".join("%12.6f" % x for x in ref["mx"][b, qs].tolist()))
            print("  sum:", " ".join("%12.6f" % x for x in ref["sumv"][b, qs].tolist()))
    print("\n----- OTmp（段3 产物 = P16 @ V；shape=[%d,%d] 全量，每行元素数=列数）-----" % (Sq, D))
    for b in range(B):
        for h in range(H):
            print("b=%d h=%d：" % (b, h))
            print(fmt_mat(ref["otmp"][b, h]))
    print("\n----- O（段4 产物 = OTmp/sum，fp16 量化；shape=[%d,%d] 全量，每行元素数=列数）-----" % (Sq, D))
    for b in range(B):
        for h in range(H):
            print("b=%d h=%d：" % (b, h))
            print(fmt_mat(ref["o16"][b, :, h]))
    print("\n----- LSE（= ln(sum)+max）-----")
    for b in range(B):
        for h in range(H):
            print("b=%d h=%d：" % (b, h),
                  " ".join("%12.6f" % x for x in ref["lse"][b, h].tolist()))
    print("=" * 100)


# ---------------- 日志解析 ----------------
DUMP_HDR = re.compile(r"DumpTensor: desc=(\d+),.*dump_size=(\d+)")
SBDUMP = re.compile(r"\[SB-DUMP\] (.*)")
NUM = re.compile(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?")


def parse_log(text):
    """按出现序返回 dump 记录 [(desc, values)]；同 desc 第 k 次出现 ↔ 第 k 轮 kernel。"""
    records = []
    cur = None
    for ln in text.splitlines():
        if "Block" in ln and "DumpTensor" not in ln and ln.strip().startswith("["):
            # "[SOFTMAX_DEBUG] ... rowActualThisSubBlock" 等含 Block 子串的设备 printf
            # 也会命中此分支——先闭合未完成记录再重置，否则整条 dump 被静默丢弃
            #（t51 desc=331 丢失根因，devlog #44.31）
            if cur is not None:
                records.append(tuple([cur[0], cur[1], cur[2]]))
                cur = None
            continue
        m = DUMP_HDR.search(ln)
        if m:
            if cur is not None:          # 上一条未闭合：新 dump 头到来即闭合它
                records.append(tuple([cur[0], cur[1], cur[2]]))
                cur = None
            cur = [int(m.group(1)), int(m.group(2)), []]
            continue
        if "DumpTensor" in ln or "[SB-DUMP]" in ln or "[RUN]" in ln or "[OUT]" in ln:
            if cur is not None:
                records.append(tuple([cur[0], cur[1], cur[2]]))
                cur = None
            continue
        if "shape is" in ln or "dumpSize is" in ln:
            continue   # ShapeInfo 元信息行（shape>dumpSize 时前置）：跳过，不闭合记录
        if cur is not None:
            vals = NUM.findall(ln)
            if vals and not re.search(r"[A-Za-z]", ln.replace("e", "").replace("E", "")):
                cur[2].extend(float(x) for x in vals)
                continue
            if ln.strip() and not re.match(r"^[\s\[\],.\-0-9eE+]+$", ln):
                records.append(tuple([cur[0], cur[1], cur[2]]))
                cur = None
    if cur is not None:
        records.append(tuple([cur[0], cur[1], cur[2]]))
    return records


def nth(records, desc, k):
    """第 k 次出现的 desc 记录"""
    hits = [r for r in records if r[0] == desc]
    return hits[k] if k < len(hits) else None


# ---------------- 逐段比对 ----------------
def compare_all(records, iters, ref, softmax_only=False):
    import torch
    s_raw = ref["s_raw"]; p16 = ref["p16"]; mx = ref["mx"]; sumv = ref["sumv"]
    otmp = ref["otmp"]; o16 = ref["o16"]; lse = ref["lse"]
    stages_ok = {st: True for st in ("S", "P", "max", "sum", "OTmp", "O", "LSE")}
    have = {r[0] for r in records}   # [T19 #45.1] 已采集 desc 全集（族存在性判据）
    all_out = []

    for it in range(iters):
        print("\n" + "#" * 90)
        print("### 第 %d 轮 kernel" % it)
        print("#" * 90)
        rep = []

        def check(stage, dump, ref_flat, tol_abs, tol_rel, label):
            """dump/ref_flat 同长扁平数组；label 含定位信息回调"""
            if dump is None:
                stages_ok[stage] = False
                rep.append((stage, "(missing)", "无 dump 数据（可能超 1MB 被丢弃）"))
                return
            # 自适应长度：dump 可短于 ref（AIV0 半行版）或等于 ref（全行版），
            # 比对公共前缀；dump 为空报缺失
            n_exp = len(ref_flat)
            n = min(len(dump), n_exp)
            if n == 0:
                stages_ok[stage] = False
                rep.append((stage, "(empty)", "dump 无数据"))
                return
            if len(dump) > n_exp:
                rep.append((stage, "(warn)", "dump %d > 期望 %d，比对前 %d" % (len(dump), n_exp, n)))
            d = torch.tensor(dump[:n])
            r = torch.tensor(ref_flat[:n])
            bad = ((d - r).abs() > tol_abs + tol_rel * r.abs())
            n = int(bad.sum())
            if n:
                stages_ok[stage] = False
                idx = bad.nonzero().flatten().tolist()
                for i in idx[:6]:
                    rep.append((stage, label(i), "dump=%.6f ref=%.6f diff=%.6f"
                                % (d[i].item(), r[i].item(), (d[i] - r[i]).item())))
                rep.append((stage, "...", "共 %d/%d 处不符" % (n, bad.numel())))
            else:
                rep.append((stage, "全部相符", "maxdiff=%.2e"
                            % ((d - r).abs().max().item() if d.numel() else 0.0)))

        # ---- S（desc=100+b*10+tile；每条 = 全 tile rows×colsPad 紧凑）----
        # [T19 #45.1] 全行比对（原只比 AIV0 半区，AIV1 分区坏点漏检）；缺失 tile
        # 双侧跳过（原填 0 会制造假阳性坏点）。
        tile_descs = lambda base: {base + b * 10 + t for b in range(B) for t, *_ in TILES}
        if not (have & tile_descs(100)):
            stages_ok.pop("S", None)
            rep.append(("S", "(skip)", "S 族（100）未采集，跳过"))
        for b in range(B):
            kept = []   # 本轮实际有 dump 的 tile：[(t, qs, rows)]
            dump_flat = []
            for (t, qs, qn, qsblk, rows) in TILES:
                rec = nth(records, 100 + b * 10 + t, it)
                if rec is not None:
                    dump_flat.extend(rec[2][: rows * COLS_PAD])
                    kept.append((t, qs, rows))
                else:
                    rep.append(("S", "(warn)", "b%dt%d dump 缺失（丢弃/漏解析），本轮跳过该 tile" % (b, t)))
            if not kept:
                continue
            ref_flat = []
            for (t, qs, rows) in kept:
                ref_flat.extend(s_raw[b, qs][: rows].reshape(-1).tolist())

            def label(i, b=b, kept=kept):
                # 定位：tile 内行主序
                for (t, qs2, rows2) in kept:
                    blk = rows2 * Sk
                    if i < blk:
                        return "b%d tile%d(head%d) s%d j%d" % (b, t, qs2, i // Sk, i % Sk)
                    i -= blk
                return "b%d idx%d" % (b, i)
            check("S", dump_flat, ref_flat, 5e-3, 1e-3, label)

        # ---- P（desc=200+b*10+tile；每条 = 全 tile rows×colsPad half）----
        # [T19 #45.1] 全行比对 + 缺失 tile 双侧跳过（同 S）
        if not (have & tile_descs(200)):
            stages_ok.pop("P", None)
            rep.append(("P", "(skip)", "P 族（200）未采集，跳过"))
        for b in range(B):
            kept = []
            dump_flat = []
            for (t, qs, qn, qsblk, rows) in TILES:
                rec = nth(records, 200 + b * 10 + t, it)
                if rec is not None:
                    dump_flat.extend(rec[2][: rows * COLS_PAD])
                    kept.append((t, qs, rows))
                else:
                    rep.append(("P", "(warn)", "b%dt%d dump 缺失，本轮跳过该 tile" % (b, t)))
            if not kept:
                continue
            ref_flat = []
            for (t, qs, rows) in kept:
                ref_flat.extend(p16[b, qs][: rows].reshape(-1).tolist())

            def label(i, b=b, kept=kept):
                for (t, qs2, rows2) in kept:
                    blk = rows2 * Sk
                    if i < blk:
                        return "b%d tile%d(head%d) s%d j%d" % (b, t, qs2, i // Sk, i % Sk)
                    i -= blk
                return "b%d idx%d" % (b, i)
            check("P", dump_flat, ref_flat, 1e-3, 0.0, label)

        # ---- stats + OTmp（desc=330/600 + b*10 + tile）----
        # [T19 #45.1] 族未采集（精简关闭）时整块跳过，不再报假 ✗
        if not (have & tile_descs(330)) and not (have & tile_descs(600)):
            for st in ("max", "sum", "OTmp"):
                stages_ok.pop(st, None)
            rep.append(("max/sum/OTmp", "(skip)", "stats 终态（330）/OTmp（600）族未采集（T19 精简），跳过"))
        else:
            for b in range(B):
                # max / sum（stats 每条 = 2×rowNum：max 前 rows、sum 后 rows）
                for name, ref_t, tol in (("max", mx, 1e-4), ("sum", sumv, 2e-3)):
                    dump_flat, ref_flat = [], []
                    for (t, qs, qn, qsblk, rows) in TILES:
                        rec = nth(records, 330 + b * 10 + t, it)
                        if rec is not None:
                            off = 0 if name == "max" else 128   # sum 在 stats 块 [128..128+rows)
                            dump_flat.extend(rec[2][off: off + rows])
                        else:
                            # 记录缺失（t51：大单行偶被解析器漏）——标记缺 dump 而非填 0 假错
                            print(f"  [WARN] b{b} tile{t} 的 stats dump 记录缺失（解析丢失），跳过该项比对")
                            dump_flat.extend([-1.0] * rows)   # -1 与任何 ref 不等但明显非 0 假象
                        ref_flat.extend(ref_t[b, qs].tolist())

                    def label(i, b=b, name=name):
                        for (t, qs2, qn2, qsblk2, rows2) in TILES:
                            if i < rows2:
                                return "b%d tile%d(head%d) %s s%d" % (b, t, qs2, name, i)
                            i -= rows2
                        return "b%d %s idx%d" % (b, name, i)
                    check(name, dump_flat, ref_flat, tol, 0.0, label)
                # OTmp（desc=600+b*10+tile，devlog #44.41 起；原 310 系在 b≥2 与 stats 撞号；
                #       每条 = AIV0 的 rows/2 行 × dPad 紧凑）
                # softmax-only 模式：段3 未运行，跳过
                if not softmax_only:
                    dump_flat, ref_flat = [], []
                    for (t, qs, qn, qsblk, rows) in TILES:
                        rec = nth(records, 600 + b * 10 + t, it)
                        if rec is not None:
                            dump_flat.extend(rec[2][: (rows // 2) * D_PAD])
                        else:
                            dump_flat.extend([0.0] * ((rows // 2) * D_PAD))
                        ref_flat.extend(otmp[b, qs][: rows // 2].reshape(-1).tolist())

                    def label(i, b=b):
                        for (t, qs2, qn2, qsblk2, rows2) in TILES:
                            blk = rows2 * D
                            if i < blk:
                                return "b%d tile%d(head%d) s%d d%d" % (b, t, qs2, i // D, i % D)
                            i -= blk
                        return "b%d idx%d" % (b, i)
                    check("OTmp", dump_flat, ref_flat, 5e-2, 1e-3, label)

        # ---- O（desc=400+b，逐 batch 区；softmax-only 模式段4 未运行跳过）----
        # [T19 #45.1] O/LSE 族未采集（精简关闭）时跳过——两者另有张量级外部校验兜底
        if not (have & {400 + b for b in range(B)}) and not (have & {450 + b for b in range(B)}):
            for st in ("O", "LSE"):
                stages_ok.pop(st, None)
            rep.append(("O/LSE", "(skip)", "O（400）/LSE（450）族未采集（T19 精简），跳过"))
        elif not softmax_only:
            dump_flat, ref_flat = [], []
            for b in range(B):
                rec = nth(records, 400 + b, it)
                if rec is not None:
                    dump_flat.extend(rec[2][: Sq * STRIDE_O])
                else:
                    dump_flat.extend([0.0] * (Sq * STRIDE_O))
                ref_flat.extend(o16[b].reshape(-1).tolist())   # [Sq,H,D] 行主序 = s,h,d

            def label(i):
                b, rem = divmod(i, Sq * STRIDE_O)
                s, rem = divmod(rem, STRIDE_O)
                h, d = divmod(rem, D)
                return "b%d s%d h%d d%d" % (b, s, h, d)
            check("O", dump_flat, ref_flat, 5e-2, 1e-3, label)

            # ---- LSE（desc=450+b，逐 batch 区）----
            dump_flat, ref_flat = [], []
            for b in range(B):
                rec = nth(records, 450 + b, it)
                if rec is not None:
                    dump_flat.extend(rec[2][: H * Sq])
                else:
                    dump_flat.extend([0.0] * (H * Sq))
                ref_flat.extend(lse[b].reshape(-1).tolist())   # [H,Sq] 头主序 = h,s

            def label(i):
                b, rem = divmod(i, H * Sq)
                h, s = divmod(rem, Sq)
                return "b%d h%d s%d" % (b, h, s)
            check("LSE", dump_flat, ref_flat, 1e-3, 0.0, label)
        else:
            rep.append(("O/LSE", "(skip)", "softmax-only 模式未运行段3/4，跳过"))

        for (st, loc, msg) in rep:
            mark = "  " if loc == "全部相符" else "✗ "
            print("%s[%-5s] %s | %s" % (mark, st, loc, msg))
        all_out.append(rep)
    return stages_ok


# ---------------- 跑 kernel（同进程：env 必须在 import 前设置） ----------------
def run_kernel(iters, softmax_only=False, multi_core=False, debug=False, dump=False):
    """跑 iters 轮 kernel；dump 输出走本进程 stdout（由调用方捕获/重定向）。"""
    os.environ["FLASH_ATTN_FORCE_SPLITB"] = "1"
    if softmax_only:
        os.environ["FLASH_ATTN_SPLITB_SOFTMAX_ONLY"] = "1"
    if multi_core:
        os.environ["FLASH_ATTN_SPLITB_MULTI_CORE"] = "1"
    if debug:
        os.environ["FLASH_ATTN_SPLITB_DEBUG"] = "1"    # 设备 [SB] printf + host 调试输出
    if dump:
        os.environ["FLASH_ATTN_SPLITB_DUMP"] = "1"     # 七项 DumpTensor（配 --log 比对）
    import torch
    import torch_npu
    from flash_attn_npu import flash_attn_func
    torch.npu.set_device(3)

    ref = make_ref()
    q = ref["q"].half().npu()
    k = ref["k"].half().npu()
    v = ref["v"].half().npu()
    o16 = ref["o16"]
    lse_ref = ref["lse"]          # [B,H,Sq] 头主序（devlog #44.49：pytest GQA LSE-only 错，加张量级校验）
    for it in range(iters):
        print("[RUN] iter=%d begin" % it, flush=True)
        out, lse, _ = flash_attn_func(q, k, v, 0.0, SCALE, False, return_attn_probs=True)
        torch.npu.synchronize()
        err = (lse.float().cpu() - lse_ref).abs()
        mx = err.max().item()
        am = torch.unravel_index(err.argmax(), err.shape)
        nbad = int((err > 1e-2).sum())
        print("[LSE] iter=%d max_err=%.4f nbad=%d/%d argmax=(b=%d h=%d s=%d) %s"
              % (it, mx, nbad, err.numel(), am[0], am[1], am[2],
                 "PASS" if mx < 1e-2 else "FAIL"), flush=True)
        if softmax_only:
            print("[OUT] iter=%d softmax-only：O 未计算，判定以 P/stats 比对为准" % it, flush=True)
            continue
        err = (out.float().cpu() - o16).abs()
        # 判定与 pytest 对齐（rtol=atol=1e-2 → tol = 1e-2 + 1e-2*|ref|）：kernel 的
        # cube 累加→div→cast 路径与 ref（matmul 一次取整）存在 1 fp16 ULP 的合法
        # 量化翻转（#44.53f 实测全部 ≤1 ULP、相对误差 ~1e-3），绝对阈值误报。
        tol = 1e-2 + 1e-2 * o16.abs()
        nbad = int((err > tol).sum())
        mx = err.max().item()
        am = torch.unravel_index(err.argmax(), err.shape)
        print("[OUT] iter=%d max_err=%.4f nbad=%d/%d argmax=(b=%d s=%d h=%d d=%d) %s"
              % (it, mx, nbad, err.numel(), am[0], am[1], am[2], am[3],
                 "PASS" if nbad == 0 else "FAIL"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--sq", type=int, default=32)
    ap.add_argument("--sk", type=int, default=32)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--softmax-only", action="store_true",
                    help="只跑段1+段2（FLASH_ATTN_SPLITB_SOFTMAX_ONLY）：S 来自真实 QK，跳过 PV/DO")
    ap.add_argument("--multi-core", action="store_true",
                    help="设 FLASH_ATTN_SPLITB_MULTI_CORE：usedCoreNum=min(B,aicNum)（与 dump 解耦，#44.46）")
    ap.add_argument("--debug", action="store_true",
                    help="设 FLASH_ATTN_SPLITB_DEBUG：设备 [SB] printf + host 调试输出")
    ap.add_argument("--dump", action="store_true",
                    help="设 FLASH_ATTN_SPLITB_DUMP：七项 DumpTensor（默认不开；配 --log 比对）")
    ap.add_argument("--print-ref", action="store_true")
    ap.add_argument("--log", type=str, default=None, help="只解析比对已有日志")
    args = ap.parse_args()
    global B, Sq, Sk, H, Hkv, D
    B, Sq, Sk, H, Hkv, D = args.batch, args.sq, args.sk, args.heads, args.kv_heads, args.dim
    init_geom()

    if args.print_ref:
        print_ref(make_ref())
        return
    if args.log:
        text = open(args.log, encoding="utf-8", errors="replace").read()
        do_compare(text, args.iters)
        return

    # 两步式（devlog #44.15）：dump 刷出是异步的（kernel 结束后才陆续刷到 stdout），
    # 同进程捕获读不到完整数据 → 本步只跑 kernel，由用户 tee 收集日志；
    # 第二步 --log FILE 解析比对。
    flags = "".join([" --" + f for f, on in (("softmax-only", args.softmax_only),
                                             ("multi-core", args.multi_core),
                                             ("debug", args.debug),
                                             ("dump", args.dump)) if on])
    print("[RUN] 开始跑 kernel：请用 `python debug/test_splitb_stage_full.py --batch %d --iters %d%s "
          "2>&1 | tee debug/log/t4x.log` 收集日志，随后 --log 比对" % (B, args.iters, flags), flush=True)
    run_kernel(args.iters, args.softmax_only, args.multi_core, args.debug, args.dump)
    if not args.dump:
        print("[RUN] 提示：未开 --dump，日志中无 DumpTensor 记录，--log 比对将无数据（仅验证运行/正确性靠 O 张量外部校验）")
    print("[RUN] 完成。执行：python debug/test_splitb_stage_full.py --batch %d --iters %d --log <日志文件>"
          % (B, args.iters), flush=True)


def do_compare(text, iters, softmax_only=False):
    ref = make_ref()
    records = parse_log(text)
    descs = sorted(set(r[0] for r in records))
    print("\n解析到 dump 记录 %d 条，desc 集合：%s" % (len(records), descs))
    # [T19 精简 #45.1] 只核对"已出现"的族（未采集族由 compare_all 自动跳过）；
    # 缺失/计数异常合并一行，替代原 40 行逐 desc 期望清单。
    tile_set = lambda base: {base + b * 10 + t for b in range(B) for t, *_ in TILES}
    batch_set = lambda base: {base + b for b in range(B)}
    for name, exp in (("S", tile_set(100)), ("P", tile_set(200)),
                      ("SMstats(max/sum)", tile_set(890)), ("DOentry", tile_set(700)),
                      ("OTmp", tile_set(600)), ("stats终态", tile_set(330)),
                      ("O", batch_set(400)), ("LSE", batch_set(450))):
        cnt = {d: sum(1 for r in records if r[0] == d) for d in exp}
        got = [d for d in exp if cnt[d] > 0]
        if not got:
            continue   # 族未采集（T19 精简关闭），不刷存在感
        n_all = sum(cnt.values())
        odd = [d for d in exp if cnt[d] != iters]
        note = "" if not odd else "   ⚠ 缺失/计数异常: %s" % odd
        print("  族 %-14s %d/%d 条%s" % (name, n_all, len(exp) * iters, note))
    stages_ok = compare_all(records, iters, ref, softmax_only)
    print("\n" + "=" * 60)
    verdict = " | ".join("%s:%s" % (k, "✓" if v else "✗") for k, v in stages_ok.items())
    print("各阶段汇总：%s" % verdict)
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bug③a 决定性三方对照（t15）：同刻 GM stats 快照链。

desc 家族：
  890+b*10+tile / 930+…(+40)  段2 写完时刻（softmaxReady set 前），双 AIV 各 dump 各自分区
  880+b*10+tile               段4 入口（divout 读前一刻），AIV0-only 整块 [max128|sum128]
  330+b*10+tile               kernel 末终态
  800/850                     divout LoadStats 后 UB 视图（既有）
判定矩阵：
  890 对 + 880 对 + O/LSE 仍坏 → divout 内部数学（读路径之外的算术/散射）
  890 对 + 880 坏             → 段2 写完后、段4 读之前被改写（谁写的？）
  890 坏                      → softmax 写本身错/晚（与 UB 视图证据互证）
用法：
  python debug/test_splitb_stage_full.py --batch 1 --heads 2 --kv-heads 2 \
    --sq 31 --sk 47 --dim 64 --iters 1 --dump > debug/log/t15.log 2>&1
  python debug/analyze_t15.py [LOG] [SQ]
"""
import importlib.util
import re
import sys

import torch

spec = importlib.util.spec_from_file_location(
    "sfull", "/data0/liaojy/workspace/FA/flash-attention-npu-smh/debug/test_splitb_stage_full.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

log = sys.argv[1] if len(sys.argv) > 1 else \
    "/data0/liaojy/workspace/FA/flash-attention-npu-smh/debug/log/t15.log"
sq = int(sys.argv[2]) if len(sys.argv) > 2 else 31
m.B, m.Sq, m.Sk, m.H, m.Hkv, m.D = 1, sq, 47, 2, 2, 64
ref = m.make_ref()

text = open(log).read()
NUM = re.compile(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?")
H_SUB = re.compile(r"desc=(\d+),")


def allrecs(desc):
    out, i = [], 0
    while True:
        i = text.find("desc=%d," % desc, i)
        if i < 0:
            break
        j = text.find("[[", i)
        if j < 0:
            break
        k = text.find("]]", j)
        out.append(torch.tensor([float(x) for x in NUM.findall(text[j:k])]))
        i = k
    return out


def cmp_tag(vals, refv, tol):
    if vals is None:
        return "✗缺"
    n = min(len(vals), len(refv))
    bad = [i for i in range(n) if abs(vals[i] - refv[i]) > tol]
    return ("✓" if not bad else "✗坏%s" % bad[:8]), bad


print("=" * 100)
print("几何 Sq=%d  AIV 分区：split=RoundDown(%d/2,8)=%d  AIV0=[0,%d) AIV1=[%d,%d)"
      % (sq, sq, sq // 2 // 8 * 8, sq // 2 // 8 * 8, sq // 2 // 8 * 8, sq))
split = sq // 2 // 8 * 8

for t in range(m.N_TILE):
    h = t  # G=1 单头 tile
    print("\n===== tile%d (head %d) =====" % (t, h))
    for kind, base, gtab, tol in (("max", 890, ref["mx"], 1e-3), ("sum", 930, ref["sumv"], 1e-2)):
        s2 = allrecs(base + t)
        # 段2 快照：每 sub 一条（分区各自）；按 sub= 标记无法从 desc 分，按出现序
        for r_i, rec in enumerate(s2):
            sub = r_i  # 出现序近似 sub 序（printf 异步，仅参考）
            off = 0 if sub == 0 else split
            n = split if sub == 0 else (sq - split)
            seg = rec[:n]
            tag, bad = cmp_tag(seg, gtab[0, h, off:off + n].tolist(), tol)
            print("[S2END %s sub?] desc=%d rec%d off=%d n=%d: %s"
                  % (kind, base + t, r_i, off, n, tag))
    do = allrecs(880 + t)
    if do:
        rec = do[-1]
        mx, sm = rec[:128], rec[128:256]
        t1, _ = cmp_tag(mx[:sq].tolist(), ref["mx"][0, h, :sq].tolist(), 1e-3)
        t2, _ = cmp_tag(sm[:sq].tolist(), ref["sumv"][0, h, :sq].tolist(), 1e-2)
        # padding 区 [sq,40) 是否垃圾（预期垃圾，非判定项）
        print("[DOENTRY desc=%d] max:%s sum:%s" % (880 + t, t1, t2))
    else:
        print("[DOENTRY desc=%d] ✗缺" % (880 + t))

# 终态与 O/LSE
for t in range(m.N_TILE):
    st = allrecs(330 + t)
    if st:
        rec = st[-1]
        mx, sm = rec[:128], rec[128:256]
        t1, _ = cmp_tag(mx[:sq].tolist(), ref["mx"][0, t, :sq].tolist(), 1e-3)
        t2, _ = cmp_tag(sm[:sq].tolist(), ref["sumv"][0, t, :sq].tolist(), 1e-2)
        print("[FINAL desc=%d] max:%s sum:%s" % (330 + t, t1, t2))
for d, name, shp in ((400, "O", (m.Sq, m.H, m.D)), (450, "LSE", (m.H, m.Sq))):
    r = allrecs(d)
    if r:
        v = r[-1][:shp[0] * shp[1] * (shp[2] if len(shp) > 2 else 1)]
        v = v.reshape(*shp) if len(shp) == 3 else v.reshape(shp)
        if name == "O":
            bad = ((v - ref["o16"][0]).abs() > 1e-2).any(dim=2)
            print("[O] 坏行:", [(int(a), int(b)) for a, b in bad.nonzero().tolist()][:20])
        else:
            for h in range(m.H):
                bl = [s for s in range(m.Sq) if abs(v[h, s] - ref["lse"][0, h, s]) > 1e-2]
                print("[LSE h%d] 坏行:" % h, bl)
print("=" * 100)

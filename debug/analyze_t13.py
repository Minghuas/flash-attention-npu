#!/usr/bin/env python3
"""Bug③a 决定性取证（t13）：B1 H2 Sq=31 Sk=47 dump 三方对拍。

问题：S/P/stats GM 终态全对、O/LSE 尾块坏——错在 PV（OTmp 本身脏）还是 divout（读到
陈旧 stats）？三方证据：
  ① OTmp 全行 dump（600 系，commit 已改全行）→ 直接判 PV
  ② divout UB 视图（800/850 系，本轮新增）→ divout 消费时刻的 max/sum 实际值
  ③ GM 终态（330 系）+ O/LSE（400/450）
判定矩阵：
  OTmp 脏                     → PV 侧 bug（cube 尾块）
  OTmp 净 + UB stats 垃圾     → softmax 写与 divout 读的时序竞争（终态正确）
  OTmp 净 + UB stats 也净     → divout 自身算术/寻址错
用法：先采 `python debug/test_splitb_stage_full.py --batch 1 --heads 2 --kv-heads 2 \
      --sq 31 --sk 47 --dim 64 --iters 1 --dump > debug/log/t13_sq31_h2.log 2>&1`，
再 `python debug/analyze_t13.py [LOG]`。
"""
import importlib.util
import re
import sys

import torch

spec = importlib.util.spec_from_file_location(
    "sfull", "/data0/liaojy/workspace/FA/flash-attention-npu-smh/debug/test_splitb_stage_full.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.B, m.Sq, m.Sk, m.H, m.Hkv, m.D = 1, 31, 47, 2, 2, 64
ref = m.make_ref()

log = sys.argv[1] if len(sys.argv) > 1 else \
    "/data0/liaojy/workspace/FA/flash-attention-npu-smh/debug/log/t13_sq31_h2.log"
text = open(log).read()
NUM = re.compile(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?")


def rawdesc(d):
    """按 desc 提取最后一个记录（UB 视图每 tile 每 AIV 各一条，取末条即可比尾块）。"""
    out = []
    i = 0
    while True:
        i = text.find("desc=%d," % d, i)
        if i < 0:
            break
        j = text.find("[[", i)
        if j < 0:
            break
        k = text.find("]]", j)
        out.append(torch.tensor([float(x) for x in NUM.findall(text[j:k])]))
        i = k
    return out[-1] if out else None


def rows_of(rec, n, cols):
    return rec[:n * cols].reshape(n, cols) if rec is not None and rec.numel() >= n * cols else None


print("=" * 96)
for h in range(m.H):
    print("\n===== b0 h%d =====" % h)
    # ① OTmp 全行（600+h*10... N_TILE tiles；此 shape 1 tile/头，desc=600+t）
    ot = rawdesc(600 + h)
    otm = rows_of(ot, m.Sq, m.D_PAD)
    if otm is None:
        print("[OTmp] ✗ 无记录/不足")
    else:
        bad = [(otm[s] - ref["otmp"][0, h, s]).abs().max() > 0.05 for s in range(m.Sq)]
        badrows = [s for s in range(m.Sq) if bad[s]]
        print("[OTmp 全行] %s 坏行=%s%s"
              % ("✓" if not badrows else "✗", badrows[:16],
                 "..." if len(badrows) > 16 else ""))
    # ③ GM 终态 stats（330+t）
    stg = rawdesc(330 + h)
    mx_gm = rows_of(stg, m.Sq, 1).flatten() if stg is not None else None
    sm_gm = stg[128:128 + m.Sq] if stg is not None and stg.numel() >= 128 + m.Sq else None
    # ② UB 视图：AIV0 与 AIV1 各一条（出现序 = 先 AIV0 后 AIV1……但 rawdesc 取末条；
    #    两 sub 的 UB 内容不同 → 需全部列出）
    for base, tag, refv in ((800, "max", ref["mx"]), (850, "sum", ref["sumv"])):
        recs, i = [], 0
        dd = base + h  # 此 shape 每头 1 tile：AIV0/AIV1 同 desc
        while True:
            i = text.find("desc=%d," % dd, i)
            if i < 0:
                break
            j = text.find("[[", i)
            k = text.find("]]", j)
            recs.append(torch.tensor([float(x) for x in NUM.findall(text[j:k])]))
            i = k
        if not recs:
            print("[ST_UB %s] ✗ 无记录 (desc=%d)" % (tag, dd))
            continue
        print("[ST_UB %s] %d 条记录（应=AIV 数）：GM 终态对照" % (tag, len(recs)))
        gm_ref = refv[0, h]
        for r_i, rec in enumerate(recs):
            vals = rec[:m.Sq]
            badv = [s for s in range(m.Sq) if abs(vals[s] - gm_ref[s].item()) > 1e-2]
            junk = int(torch.isnan(vals).sum()) + int(torch.isinf(vals).sum())
            huge = int((vals.abs() > 1e6).sum())
            head = "  rec%d(sub?)" % r_i
            if not badv and junk == 0 and huge == 0:
                print(head + " ✓ 全部 31 行与金标一致")
            else:
                print(head + " ✗ 与金标不一致 行=%s%s | nan/inf=%d 超大=%d"
                      % (badv[:10], "..." if len(badv) > 10 else "", junk, huge))
                if badv:
                    s0 = badv[0]
                    print("      s=%d: UB=%.4f GMref=%.4f | UB[后4]=%s"
                          % (s0, vals[s0], gm_ref[s0].item(),
                             [round(x, 3) for x in vals[m.Sq:m.Sq + 4].tolist()]))

# ④ O / LSE 坏行复核
o = rawdesc(400)
om = rows_of(o, m.Sq, m.H * m.D)
if om is not None:
    omr = om.reshape(m.Sq, m.H, m.D)
    bad = ((omr - ref["o16"][0]).abs() > 1e-2).any(dim=2)
    print("\n[O] 坏行 (s,h):", [(int(a), int(b)) for a, b in bad.nonzero().tolist()])
l = rawdesc(450)
lm = rows_of(l, m.H, m.Sq)
for h in range(m.H):
    if lm is not None:
        bl = [s for s in range(m.Sq) if abs(lm[h, s] - ref["lse"][0, h, s].item()) > 1e-2]
        print("[LSE h%d] 坏行=" % h, bl)
print("=" * 96)

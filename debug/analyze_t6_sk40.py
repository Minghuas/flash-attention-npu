#!/usr/bin/env python3
"""t6_sk40.log 取证：Sk=40（Bug③b）——b1 的 O/LSE vs 金标；h5/h6/h7 的 stats/OTmp vs 金标。

背景（devlog #44.50 后）：b0 tile0 的 S/P/stats/OTmp 已肉眼验证正确，错误 argmax 在 b1
（LSE b1 h6 s15；O b1 s9 h5 d0）。本脚本回答：b1 坏行的输入（stats/OTmp）是否也正确？
  - 若 stats/OTmp 正确而 O/LSE 错 → divout 数学/寻址缺陷（定位到行/值模式）
  - 若 stats/OTmp 也错 → 上游（softmax/PV）对特定 tile 出错（推翻 b0-tile0 外推）
顺带全量校验 b0/b1 的 S（100-117）与 b0 的 P（200-207）。
用法：python debug/analyze_t6_sk40.py
"""
import importlib.util
import torch

spec = importlib.util.spec_from_file_location(
    "sfull", "/data0/liaojy/workspace/FA/flash-attention-npu-smh/debug/test_splitb_stage_full.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# t6 实测 shape（从 S dump 反推：8 tile/batch、rows=16 → qn=1 → G=1 → Hkv=8）
m.B, m.Sq, m.Sk, m.H, m.Hkv, m.D = 2, 16, 40, 8, 8, 64
ref = m.make_ref()
print("几何：COLS_PAD=%d D_PAD=%d N_TILE=%d tiles(q_start)=%s"
      % (m.COLS_PAD, m.D_PAD, m.N_TILE, [t[1] for t in m.TILES]))

recs = m.parse_log(open("/data0/liaojy/workspace/FA/flash-attention-npu-smh/debug/log/t6_sk40.log").read())
by_desc = {}
for d, sz, vals in recs:
    by_desc.setdefault(d, []).append(torch.tensor(vals))
print("解析记录 desc:", sorted(by_desc))


def get(d):
    return by_desc[d][0] if d in by_desc else None


def report(tag, dump, refv, tol=2e-3):
    if dump is None:
        print("%-18s ✗ 无 dump" % tag)
        return None
    n = min(dump.numel(), refv.numel())
    d, r = dump[:n].float(), refv.reshape(-1)[:n].float()
    bad = (d - r).abs() > tol
    print("%-18s %s n=%d nbad=%d maxdiff=%.4f"
          % (tag, "✓" if not bad.any() else "✗", n, int(bad.sum()),
             (d - r).abs().max().item() if n else 0.0))
    return bad, d, r


# ---- 1) b1 全量 O（desc=401, [Sq,H,D] 内存序）与 LSE（desc=451, [H,Sq] 头主序）----
o_dump = get(401)
lse_dump = get(451)
if o_dump is not None:
    o = o_dump.reshape(m.Sq, m.H, m.D)            # [s,h,d]
    r = ref["o16"][1]                              # [Sq,H,D]
    diff = (o - r).abs()
    bad = diff > 2e-3
    print("\n[O b1] nbad=%d/%d maxdiff=%.4f" % (int(bad.sum()), bad.numel(), diff.max()))
    rows = bad.any(dim=2).nonzero()               # (s,h)
    print("坏行 (s,h) 共 %d：" % len(rows), [(int(a), int(b_)) for a, b_ in rows.tolist()])
    for s, h in rows.tolist()[:6]:
        ratio = (o[s, h] * ref["sumv"][1, h, s] / ref["otmp"][1, h, s].clamp(min=1e-9))
        print("  s=%d h=%d: dump前4=%s ref前4=%s  dump×sum/otmp(有效除数倍率)≈%.4f"
              % (s, h, [round(x, 3) for x in o[s, h, :4].tolist()],
                 [round(x, 3) for x in r[s, h, :4].tolist()],
                 ratio[diff[s, h] > 2e-3][:4].mean().item()))
if lse_dump is not None:
    l = lse_dump.reshape(m.H, m.Sq)               # [h,s]
    r = ref["lse"][1]
    diff = (l - r).abs()
    bad = diff > 2e-3
    print("\n[LSE b1] nbad=%d/%d maxdiff=%.4f" % (int(bad.sum()), bad.numel(), diff.max()))
    cells = bad.nonzero()
    print("坏格 (h,s)：", [(int(a), int(b_)) for a, b_ in cells.tolist()])

# ---- 2) b1 tile5/6/7 stats（desc=345/346/347: [max128|sum128]）与 OTmp（desc=615/616/617: 前8行×64）----
print()
for t in (5, 6, 7):
    h = t  # G=1 → q_start=t=head
    st = get(345 - 5 + t) if False else get(340 + t)
    ot = get(610 + t)
    if st is not None and st.numel() >= 2 * 128:
        mx, sm = st[:128][:m.Sq], st[128:144][:m.Sq]
        print("[stats b1 h=%d] max %s | sum %s" % (
            h,
            "✓" if (mx - ref["mx"][1, h]).abs().max() < 1e-3 else
            "✗ maxdiff=%.4f" % (mx - ref["mx"][1, h]).abs().max(),
            "✓" if (sm - ref["sumv"][1, h]).abs().max() < 1e-2 else
            "✗ maxdiff=%.4f" % (sm - ref["sumv"][1, h]).abs().max()))
        print("   dump max[:4]=%s ref=%s" % ([round(x, 4) for x in mx[:4].tolist()],
                                             [round(x, 4) for x in ref["mx"][1, h, :4].tolist()]))
        print("   dump sum[:4]=%s ref=%s" % ([round(x, 4) for x in sm[:4].tolist()],
                                             [round(x, 4) for x in ref["sumv"][1, h, :4].tolist()]))
    if ot is not None:
        ot8 = ot[:8 * m.D].reshape(8, m.D)
        r8 = ref["otmp"][1, h][:8]
        print("[OTmp b1 h=%d 前8行] %s maxdiff=%.4f"
              % (h, "✓" if (ot8 - r8).abs().max() < 0.05 else "✗",
                 (ot8 - r8).abs().max()))

# ---- 3) S/P 全量校验（b0: 100-107/200-207；b1: 110-117）----
print()
for b in range(m.B):
    for t, qs, qn, qsblk, rows in m.TILES:
        s_rec, p_rec = get(100 + b * 10 + t), get(200 + b * 10 + t) if b == 0 else None
        if s_rec is not None:
            sm_ = s_rec[:rows * m.COLS_PAD].reshape(rows, m.COLS_PAD)
            rr = ref["s_raw"][b, qs].repeat_interleave(qn, 0) if qn > 1 else ref["s_raw"][b, qs]
            dd = (sm_[:, :m.Sk] - rr).abs().max()
            padz = sm_[:, m.Sk:].abs().max()
            print("[S b%d tile%d h=%d] 值%s(%.5f) pad零%s(%.5f)"
                  % (b, t, qs, "✓" if dd < 2e-3 else "✗", dd,
                     "✓" if padz == 0 else "✗", padz))
        if p_rec is not None:
            pm = p_rec[:rows * m.COLS_PAD].reshape(rows, m.COLS_PAD)
            rr = ref["p16"][b, qs].repeat_interleave(qn, 0) if qn > 1 else ref["p16"][b, qs]
            dd = (pm[:, :m.Sk] - rr).abs().max()
            padz = pm[:, m.Sk:].abs().max()
            print("[P b0 tile%d h=%d] 值%s(%.5f) pad零%s(%.5f)"
                  % (t, qs, "✓" if dd < 2e-2 else "✗", dd,
                     "✓" if padz == 0 else "✗", padz))

# ---- 4) S 错误行级模式：错行是否 = "Q 取自 h+1"（s_raw×(h+2)/(h+1)）----
print("\n==== S 错误行级取证 ====")
for (b, t) in [(0, 6), (1, 5), (1, 6)]:
    h = t
    rec = get(100 + b * 10 + t)
    sm_ = rec[:16 * m.COLS_PAD].reshape(16, m.COLS_PAD)[:, :m.Sk]
    own = ref["s_raw"][b, h]
    nxt = own * (h + 2) / (h + 1)          # q 线性 ⇒ 换头 q 即整体乘 (h+2)/(h+1)
    nxtfull = ref["s_raw"][b, h + 1] if h + 1 < m.H else None   # q 和 k 都取 h+1
    rows_bad, rows_nxtq, rows_nxtqk = [], 0, 0
    for s in range(16):
        d1 = (sm_[s] - own[s]).abs().max().item()
        d2 = (sm_[s] - nxt[s]).abs().max().item()
        d3 = (nxtfull[s] - sm_[s]).abs().max().item() if nxtfull is not None else -1
        if d1 > 2e-3:
            rows_bad.append(s)
            if d2 < 2e-2:
                rows_nxtq += 1
            if d3 >= 0 and d3 < 2e-2:
                rows_nxtqk += 1
    print("b%d tile%d(h=%d): 坏行=%s | 其中匹配『仅Q换h+1』%d 行 | 匹配『Q,K都换h+1』%d 行"
          % (b, t, h, rows_bad, rows_nxtq, rows_nxtqk))
    s0 = rows_bad[0] if rows_bad else 0
    print("   示例行 s=%d: dump j[0,8]=%s" % (s0, [round(x, 3) for x in sm_[s0, :8].tolist()]))
    print("            own    j[0,8]=%s" % [round(x, 3) for x in own[s0, :8].tolist()])
    print("            nxtq   j[0,8]=%s" % [round(x, 3) for x in nxt[s0, :8].tolist()])

# b0 stats 330-333 复查（此前只验了 b0 tile0）
for t in range(4):
    st = get(330 + t)
    if st is not None and st.numel() >= 144:
        mx, sm2 = st[:16], st[128:144]
        print("[stats b0 h=%d] max %s | sum %s" % (
            t, "✓" if (mx - ref["mx"][0, t]).abs().max() < 1e-3 else "✗",
            "✓" if (sm2 - ref["sumv"][0, t]).abs().max() < 1e-2 else "✗"))

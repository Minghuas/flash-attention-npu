#!/usr/bin/env python3
"""Bug③a 取证（Sq=31）：按 tile/AIV 行范围核对 S/P/stats 坏行首现阶段。

签名（bug_tracker.md）：每 AIV 行范围 R 的非整 8 尾块 [8⌊R/8⌋, R) 整块坏（O/LSE 同坏）。
Sq=31 G=1：AIV0 rows0-14（尾块 8-14 坏）、AIV1 rows15-30（整 8 块净）。
用法：先 `python debug/test_splitb_stage_full.py --batch 2 --heads 8 --kv-heads 8 \
      --sq 31 --sk 47 --dim 64 --iters 1 --dump > debug/log/t9_sq31.log 2>&1`，
再 `python debug/analyze_sq31.py`。
"""
import importlib.util
import torch

spec = importlib.util.spec_from_file_location(
    "sfull", "/data0/liaojy/workspace/FA/flash-attention-npu-smh/debug/test_splitb_stage_full.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.B, m.Sq, m.Sk, m.H, m.Hkv, m.D = 2, 31, 47, 8, 8, 64
ref = m.make_ref()
AIV0_R, AIV1_OFF = m.Sq // 2, m.Sq // 2          # 15, 15（AIV1 R=16）
print("几何：N_TILE=%d colsPad=%d AIV0=[0,%d) AIV1=[%d,%d)"
      % (m.N_TILE, m.COLS_PAD, AIV0_R, AIV1_OFF, m.Sq))

recs = m.parse_log(open("/data0/liaojy/workspace/FA/flash-attention-npu-smh/debug/log/t9_sq31.log").read())
by_desc = {}
for d, sz, vals in recs:
    by_desc.setdefault(d, []).append(torch.tensor(vals))
print("desc:", sorted(by_desc))
get = lambda d: by_desc[d][0] if d in by_desc else None

# ---- 逐 tile 核对 S（rows=31×colsPad）、P、stats ----
for b in range(m.B):
    for t, qs, qn, qsblk, rows in m.TILES:
        h = qs
        sr = get(100 + b * 10 + t)
        if sr is not None and sr.numel() >= m.Sq * m.COLS_PAD:
            sm_ = sr[:m.Sq * m.COLS_PAD].reshape(m.Sq, m.COLS_PAD)
            own = ref["s_raw"][b, h]
            rowbad = [(sm_[s, :m.Sk] - own[s]).abs().max() > 2e-3 for s in range(m.Sq)]
            bad = [s for s in range(m.Sq) if rowbad[s]]
            print("[S  b%d h%d] 坏行=%s%s  pad零=%s"
                  % (b, h, bad[:12], "..." if len(bad) > 12 else "",
                     sm_[:, m.Sk:].abs().max() == 0))
        pr = get(200 + b * 10 + t)
        if pr is not None and pr.numel() >= m.Sq * m.COLS_PAD:
            pm = pr[:m.Sq * m.COLS_PAD].reshape(m.Sq, m.COLS_PAD)
            own = ref["p16"][b, h]
            diff = (pm[:, :m.Sk] - own).abs()
            rowbad = [(diff[s] > 2e-2).any() for s in range(m.Sq)]
            bad = [s for s in range(m.Sq) if rowbad[s]]
            print("[P  b%d h%d] 坏行=%s%s  pad零=%s"
                  % (b, h, bad[:12], "..." if len(bad) > 12 else "",
                     pm[:, m.Sk:].abs().max() == 0))
        st = get(330 + b * 10 + t)
        if st is not None and st.numel() >= 2 * m.ROW_NUM_MAX:
            mx, sm2 = st[:128], st[128:256]
            bmx = [(mx[s] - ref["mx"][b, h, s]).abs() > 1e-3 for s in range(m.Sq)]
            bsm = [(sm2[s] - ref["sumv"][b, h, s]).abs() > 1e-2 for s in range(m.Sq)]
            print("[ST b%d h%d] max坏行=%s | sum坏行=%s"
                  % (b, h, [s for s in range(m.Sq) if bmx[s]][:12],
                     [s for s in range(m.Sq) if bsm[s]][:12]))
            if any(bmx[:15]) or any(bsm[:15]):
                print("    AIV0 区 dump max[0:16]=%s" % [round(x, 3) for x in mx[:16].tolist()])
                print("    AIV0 区 ref  max[0:16]=%s"
                      % [round(x, 3) for x in ref["mx"][b, h, :16].tolist()])
                print("    AIV0 区 dump sum[0:16]=%s" % [round(x, 2) for x in sm2[:16].tolist()])
                print("    AIV0 区 ref  sum[0:16]=%s"
                      % [round(x, 2) for x in ref["sumv"][b, h, :16].tolist()])

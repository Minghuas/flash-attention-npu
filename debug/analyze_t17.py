#!/usr/bin/env python3
"""Bug⑥ 残余取证（t17/t19）：softmax 写出时刻 stats 对拍（S/P dump 视开关）。

desc 家族（在役，[T19 精简 #45.1]）：
  890/930(+40)  段2 写出时刻 max/sum（双 AIV 各自分区，printf 头带 sub/off/n）
  700(+b*10+t)  段4 入口 GM 整块（AIV0-only；原 880 改号避免与 890 撞号）
  100           段1 S_raw（全 tile）——CUBE/AIV 侧裁决主证据
判定（t19 裁决树，devlog #45.1）：
  S@100 撕裂（首行、比值≈(h+2)/(h+1)） → CUBE 侧（QK 内部，守卫被穿透）
  S@100 净 + 890 错                      → AIV 侧（softmax 读/算路径）
用法（shape 自动从日志的 [splitb host] 行解析，可覆盖）：
  python debug/test_splitb_stage_full.py --batch 3 --heads 4 --kv-heads 4 \
    --sq 32 --sk 96 --dim 64 --iters 3 --dump > debug/log/t19.log 2>&1
  python debug/analyze_t17.py [LOG]
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
    "/data0/liaojy/workspace/FA/flash-attention-npu-smh/debug/log/t19.log"
text = open(log).read()

# ---- shape 来源优先级：CLI 尾参 > 日志 [splitb host] 行 > 默认 ----
# CLI：python analyze_t17.py [LOG] [B Sq Sk H Hkv D]（dump 轮不开 --debug 时用，避免
# printf 观测效应；--debug 轮则可自动解析）
if len(sys.argv) >= 8:
    m.B, m.Sq, m.Sk, m.H, m.Hkv, m.D = (int(x) for x in sys.argv[2:8])
    print("shape（CLI）: B=%d Sq=%d Sk=%d H=%d Hkv=%d D=%d" % (m.B, m.Sq, m.Sk, m.H, m.Hkv, m.D))
else:
    host = re.search(r"\[splitb host\] B=(\d+) Sq=(\d+) Sk=(\d+) H=(\d+) Hkv=(\d+) D=(\d+)", text)
    if host:
        m.B, m.Sq, m.Sk, m.H, m.Hkv, m.D = (int(x) for x in host.groups())
        print("shape（自日志）: B=%d Sq=%d Sk=%d H=%d Hkv=%d D=%d" % (m.B, m.Sq, m.Sk, m.H, m.Hkv, m.D))
    else:
        print("⚠ 无 CLI shape 且日志无 [splitb host] 行，用默认 B16 H8 Sq32 Sk96")
        m.B, m.Sq, m.Sk, m.H, m.Hkv, m.D = 16, 32, 96, 8, 8, 64
ref = m.make_ref()
NUM = re.compile(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?")

# 每 iter 一个 kernel 调用；按 [RUN] iter=N 分段
iters = re.split(r"\[RUN\] iter=(\d+) begin", text)
chunks = [(int(iters[i]), iters[i + 1]) for i in range(1, len(iters) - 1, 2)]
print("日志含 %d 个 iter 段；[LSE] 行：" % len(chunks))
for ln in text.splitlines():
    if ln.startswith("[LSE]"):
        print("  " + ln)


def recs(chunk, desc):
    out, i = [], 0
    while True:
        i = chunk.find("desc=%d," % desc, i)
        if i < 0:
            break
        j = chunk.find("[[", i)
        if j < 0:
            break
        k = chunk.find("]]", j)
        out.append(torch.tensor([float(x) for x in NUM.findall(chunk[j:k])]))
        i = k
    return out


# AIV 行分摊（复刻 softmax 对齐版）：qn=1 → split=RoundDown(Sq/2,8)
SPLIT = (m.Sq // 2) // 8 * 8 if m.H == m.Hkv and m.Sq >= 16 else 0
print("AIV 行分摊 split=%d（AIV0 [0,%d) / AIV1 [%d,%d)）" % (SPLIT, SPLIT, SPLIT, m.Sq))

for itno, ch in chunks:
    lsere = re.search(r"\[LSE\] iter=%d max_err=[\d.]+ nbad=(\d+)/\d+" % itno, text)
    if not lsere or int(lsere.group(1)) == 0:
        continue
    print("\n===== iter %d（LSE 有坏点，开始对拍）=====" % itno)
    # ---- ① 裁决主证据：S@100 全 tile 撕裂检查（对 golden s_raw，前 4 行报告）----
    s_bad = 0
    for h in range(m.H):
        for b in range(m.B):
            rs = recs(ch, 100 + b * 10 + h)   # tile 序 == 头序（qn=1）
            if not rs:
                continue
            d = rs[-1]                        # 本 iter 最后一条
            g = ref["s_raw"][b, h].reshape(-1)
            n = min(d.numel(), g.numel())
            bad = [(i // m.Sk, i % m.Sk, d[i].item(), g[i].item())
                   for i in range(n) if abs(d[i].item() - g[i].item()) > 5e-3 + 1e-3 * abs(g[i].item())]
            if bad:
                s_bad += len(bad)
                rows = sorted(set(x[0] for x in bad))
                print("  [S@100 撕裂] b%d h%d：%d 点，行集 %s（首例 s%d j%d UB=%.4f ref=%.4f 比值=%.4f）"
                      % (b, h, len(bad), rows[:6], bad[0][0], bad[0][1],
                         bad[0][2], bad[0][3], bad[0][2] / bad[0][3] if bad[0][3] else 0.0))
    if s_bad == 0:
        print("  [S@100 全净] → 裁决：AIV 侧（softmax 读/算路径），继续查 890/930")
    # ---- ② 890/930 写出时刻 stats（按 printf 头 sub/off/n 分区解析）----
    for h in range(m.H - 1, -1, -1):
        verdict = []
        for b in range(m.B):
            for kind, base, gtab, tol in (("max", 890, ref["mx"], 2e-3),
                                          ("sum", 930, ref["sumv"], 5e-3)):
                rs = recs(ch, base + b * 10 + h)
                # 逐条找 printf 头的 sub/off/n（行偏移分区信息在头里，按头解析）
                hdrs = re.findall(r"STATS_GM_S2END\(%s sub=(\d+)\).*?off=(\d+) n=(\d+) desc=%d,"
                                  % (kind, base + b * 10 + h), ch)
                for (sub, off, n), rec in zip(hdrs, rs):
                    off, n = int(off), int(n)
                    if rec.numel() < n:
                        continue
                    bad = [s for s in range(n)
                           if abs(rec[s].item() - gtab[b, h, off + s].item()) > tol]
                    if bad:
                        verdict.append("890族 %s sub%s 行%s（首例 UB=%.4f ref=%.4f）"
                                       % (kind, sub, [off + x for x in bad[:4]],
                                          rec[bad[0]].item(), gtab[b, h, off + bad[0]].item()))
        if verdict:
            print("  h%d: %s" % (h, "; ".join(verdict[:3])))
            break
print("done")

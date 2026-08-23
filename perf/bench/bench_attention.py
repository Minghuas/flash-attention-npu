#!/usr/bin/env python3
# Copyright (c) 2026, perf project — 大 Batch 小 SeqLen 性能优化
#
# FlashAttention v2 (our impl)  vs  torch_npu.npu_fusion_attention (CANN baseline)
# 基准测试脚本：测量 fwd 延迟与算力利用率，定位大 Batch 小 SeqLen 场景的性能差距。
#
# 用法：
#   conda activate FA2
#   python perf/bench/bench_attention.py                       # 跑默认小 seqlen 网格
#   python perf/bench/bench_attention.py --batch 1 8 64 256 --seqlen 64 128 512 2048
#   python perf/bench/bench_attention.py --dtype bf16 --headdim 128 --causal
#   python perf/bench/bench_attention.py --out results/bench.csv --json results/bench.json

import argparse
import json
import math
import statistics
import sys
import time
from contextlib import contextmanager

import torch
import torch_npu  # noqa: F401  注册 NPU 后端


# --------------------------------------------------------------------------- #
#  被测对象封装
# --------------------------------------------------------------------------- #

# 我方 FA v2 编译产物：site-packages 里装成 flash_attn_npu_arch22_v2 这个 .so。
# 它注册的 op schema（位置参数）：
#   fwd(q[B,Sq,H,D], k[B,Sk,Hkv,D], v[B,Sk,Hkv,D],
#      out|None, alibi_slopes|None, dropout_p, scale,
#      causal, win_left, win_right, softcap, return_softmax, gen|None)
#      -> [out, softmax_lse, S_dmask, rng_state]
_FA = None
def _load_fa():
    global _FA
    if _FA is None:
        import flash_attn_npu_arch22_v2 as fa
        _FA = fa
    return _FA


def fa_fwd(q, k, v, *, scale, causal, softcap=0.0):
    """调用我方 FA v2 前向（BSND 布局 [B, Sq, H, D]）。返回 out。"""
    fa = _load_fa()
    # 输入需 contiguous（接口约定）
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    out, softmax_lse, S_dmask, rng_state = fa.fwd(
        q, k, v,
        None,           # out_  -> 由 kernel 分配
        None,           # alibi_slopes_ -> 当前分支不支持
        0.0,            # dropout_p
        scale,
        causal,
        -1, -1,         # window_size_left/right  -> -1 表示无 SWA
        softcap,
        False,          # return_softmax
        None,           # generator
    )
    return out


def npu_fa_fwd(q_bnsd, k_bnsd, v_bnsd, *, head_num, scale, causal):
    """调用 torch_npu.npu_fusion_attention（CANN baseline，BNSD 布局 [B,H,S,D]）。返回 out。"""
    if causal:
        # 下三角参与计算：mask=True 处被遮蔽 -> 上三角（不含对角线）置 True
        sq, skv = q_bnsd.shape[2], k_bnsd.shape[2]
        atten_mask = torch.triu(
            torch.ones(sq, skv, dtype=torch.bool, device=q_bnsd.device), diagonal=1
        )
        # 右下因果：next_tockens=0, pre_tockens>=Sq
        out, _, _, _, _, _, _ = torch_npu.npu_fusion_attention(
            q_bnsd, k_bnsd, v_bnsd, head_num, "BNSD",
            atten_mask=atten_mask, scale=scale,
            pre_tockens=sq, next_tockens=0,
        )
    else:
        out, _, _, _, _, _, _ = torch_npu.npu_fusion_attention(
            q_bnsd, k_bnsd, v_bnsd, head_num, "BNSD", scale=scale,
        )
    return out


# --------------------------------------------------------------------------- #
#  计时
# --------------------------------------------------------------------------- #

def _sync():
    for _ in range(3):
        torch.npu.synchronize()


def time_callable(fn, *, warmup, repeat):
    """返回单次调用中位数延迟（毫秒）。"""
    _sync()
    for _ in range(warmup):
        fn()
    _sync()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1e3)  # ms
    return statistics.median(times), min(times)


def flops_fwd(batch, sq, skv, nheads, headdim, causal):
    """FA 前向 FLOP 数（QK + PV，causal 减半）。"""
    return 4 * batch * sq * skv * nheads * headdim // (2 if causal else 1)


# --------------------------------------------------------------------------- #
#  单个 (B, S, ...) 配置的测量
# --------------------------------------------------------------------------- #

def run_one(batch, seqlen, nheads, headdim, dtype, causal, softcap,
            warmup, repeat, device):
    torch.manual_seed(0)
    # 我方 FA：BSND [B, Sq, H, D]，k/v 同（取 H_kv=H，即非 GQA，便于与 baseline 对齐）
    q_fa = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
    k_fa = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
    v_fa = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
    # baseline：BNSD [B, H, S, D]
    q_nb = q_fa.transpose(1, 2).contiguous()
    k_nb = k_fa.transpose(1, 2).contiguous()
    v_nb = v_fa.transpose(1, 2).contiguous()

    scale = 1.0 / math.sqrt(headdim)
    fl = flops_fwd(batch, seqlen, seqlen, nheads, headdim, causal)

    results = {}
    # 我方 FA v2
    try:
        fa_fwd(q_fa, k_fa, v_fa, scale=scale, causal=causal, softcap=softcap)  # 触发 JIT/编译路径
        med, mn = time_callable(
            lambda: fa_fwd(q_fa, k_fa, v_fa, scale=scale, causal=causal, softcap=softcap),
            warmup=warmup, repeat=repeat,
        )
        results["fa_v2"] = {"median_ms": med, "min_ms": mn, "tflops": fl / med / 1e9}
    except Exception as e:
        results["fa_v2"] = {"error": f"{type(e).__name__}: {e}"}

    # CANN baseline
    try:
        npu_fa_fwd(q_nb, k_nb, v_nb, head_num=nheads, scale=scale, causal=causal)
        med, mn = time_callable(
            lambda: npu_fa_fwd(q_nb, k_nb, v_nb, head_num=nheads, scale=scale, causal=causal),
            warmup=warmup, repeat=repeat,
        )
        results["npu_fusion_attn"] = {"median_ms": med, "min_ms": mn, "tflops": fl / med / 1e9}
    except Exception as e:
        results["npu_fusion_attn"] = {"error": f"{type(e).__name__}: {e}"}

    # 加速比
    fa_r = results.get("fa_v2", {})
    nb_r = results.get("npu_fusion_attn", {})
    if "median_ms" in fa_r and "median_ms" in nb_r and nb_r["median_ms"] > 0:
        results["speedup_fa_over_baseline"] = nb_r["median_ms"] / fa_r["median_ms"]
    return results


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="FA v2 vs npu_fusion_attention bench (small-seq, large-batch)")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 4, 16, 64, 256, 1024],
                    help="batch 列表（默认聚焦大 batch）")
    ap.add_argument("--seqlen", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024],
                    help="seqlen 列表（默认含小 seqlen）")
    ap.add_argument("--nheads", type=int, default=32)
    ap.add_argument("--headdim", type=int, default=128, choices=[64, 128])
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--causal", action="store_true", default=False)
    ap.add_argument("--softcap", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=100,
                    help="单点重复次数取中位数（2026-08-14 用户要求 ≥100；此前数据为 repeat=10/20）")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--out", default=None, help="CSV 输出路径")
    ap.add_argument("--json", default=None, help="JSON 输出路径")
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = args.device
    torch.npu.set_device(device)

    print(f"# bench: device={torch_npu.npu.get_device_name()} dtype={args.dtype} "
          f"nheads={args.nheads} headdim={args.headdim} causal={args.causal} "
          f"softcap={args.softcap} warmup={args.warmup} repeat={args.repeat}")
    print(f"# grid: batch={args.batch} seqlen={args.seqlen}")
    print()
    header = f"{'batch':>6} {'seqlen':>7} {'fa_ms':>9} {'base_ms':>9} {'speedup':>8} {'fa_TF':>8} {'base_TF':>8}"
    print(header)
    print("-" * len(header))

    rows = []
    for b in args.batch:
        for s in args.seqlen:
            r = run_one(b, s, args.nheads, args.headdim, dtype, args.causal, args.softcap,
                        args.warmup, args.repeat, device)
            fa, nb = r.get("fa_v2", {}), r.get("npu_fusion_attn", {})
            fa_ms = fa.get("median_ms")
            nb_ms = nb.get("median_ms")
            sp = r.get("speedup_fa_over_baseline")
            row = {
                "batch": b, "seqlen": s, "nheads": args.nheads, "headdim": args.headdim,
                "dtype": args.dtype, "causal": args.causal, "softcap": args.softcap,
                "fa_v2_median_ms": fa_ms, "fa_v2_min_ms": fa.get("min_ms"),
                "fa_v2_tflops": fa.get("tflops"),
                "baseline_median_ms": nb_ms, "baseline_min_ms": nb.get("min_ms"),
                "baseline_tflops": nb.get("tflops"),
                "speedup_fa_over_baseline": sp,
                "fa_v2_error": fa.get("error"), "baseline_error": nb.get("error"),
            }
            rows.append(row)
            fa_s = f"{fa_ms:9.3f}" if fa_ms is not None else f"{'ERR':>9}"
            nb_s = f"{nb_ms:9.3f}" if nb_ms is not None else f"{'ERR':>9}"
            sp_s = f"{sp:8.2f}x" if sp is not None else f"{'-':>8}"
            fa_t = f"{fa.get('tflops', 0):8.1f}" if fa_ms is not None else f"{'-':>8}"
            nb_t = f"{nb.get('tflops', 0):8.1f}" if nb_ms is not None else f"{'-':>8}"
            print(f"{b:>6} {s:>7} {fa_s} {nb_s} {sp_s} {fa_t} {nb_t}")

    # speedup < 1 表示我方慢于 baseline（劣化），即 issue 描述的现象
    print()
    print("# speedup < 1.0x => 我方 FA 慢于 baseline（性能劣化点）")

    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"# CSV -> {args.out}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"config": vars(args), "rows": rows}, f, indent=2)
        print(f"# JSON -> {args.json}")


if __name__ == "__main__":
    main()

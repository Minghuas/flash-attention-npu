#!/usr/bin/env python3
# Copyright (c) 2026, perf project — 大 Batch 小 SeqLen 性能优化
#
# GQA 版 bench2：q 头数 ≠ kv 头数（--nheads / --kv-heads），其余同 bench2.py。
# 背景：v3.3 恢复 GQA qN 打包（fork loadAPackedBSHD），本脚本量化打包收益。
#
# 闸门提醒（shape_supported）：n2g=num_heads（q 头）× alignedS1 × alignedS2 × dtype
# ≤ 128KB——H32 仅 s=32 在闸门内（32×32×32×2=64KB）；H16 可到 s=64（边界 128KB）。
# 超 silently 走旧路径（数字会与 SplitB 无关）。
#
# 用法：
#   FLASH_ATTN_SPLITB_MULTI_CORE=1 python perf/bench/bench2_gqa.py --device 6 \
#     --nheads 32 --kv-heads 4 --seqlen 32 --batch 1 4 16 64 256 320 1024
#   # A/B：加 FLASH_ATTN_DISABLE_SPLITB=1 跑旧路径对照
#   # H16 配置覆盖 s=64：--nheads 16 --kv-heads 2 --seqlen 32 64

import argparse
import json
import math
import statistics

import torch
import torch_npu  # noqa: F401

from flash_attn_npu import flash_attn_func


def fa_fwd(q, k, v, *, scale, causal, softcap=0.0, window_size=(-1, -1)):
    out, *_ = flash_attn_func(q, k, v, 0.0, scale, causal, window_size, softcap)
    return out


def npu_fa_fwd(q_bnsd, k_bnsd, v_bnsd, *, head_num, scale):
    out, *_ = torch_npu.npu_fusion_attention(
        q_bnsd, k_bnsd, v_bnsd, head_num, "BNSD", scale=scale)
    return out


def time_callable(fn, *, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        start_event.record()
        fn()
        end_event.record()
        torch.npu.synchronize()
        times.append(start_event.elapsed_time(end_event))
    return statistics.mean(times), min(times)


def flops_fwd(batch, sq, skv, nheads, headdim):
    return 4 * batch * sq * skv * nheads * headdim


def run_one(batch, seqlen, nheads, kv_heads, headdim, dtype, softcap,
            warmup, repeat, device):
    torch.manual_seed(42)
    q_fa = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device).contiguous()
    k_fa = torch.randn(batch, seqlen, kv_heads, headdim, dtype=dtype, device=device).contiguous()
    v_fa = torch.randn(batch, seqlen, kv_heads, headdim, dtype=dtype, device=device).contiguous()
    q_nb = q_fa.transpose(1, 2).contiguous()
    k_nb = k_fa.transpose(1, 2).contiguous()
    v_nb = v_fa.transpose(1, 2).contiguous()
    scale = 1.0 / math.sqrt(headdim)
    fl = flops_fwd(batch, seqlen, seqlen, nheads, headdim)

    results = {}
    fa_fwd(q_fa, k_fa, v_fa, scale=scale, causal=False, softcap=softcap)
    med, mn = time_callable(
        lambda: fa_fwd(q_fa, k_fa, v_fa, scale=scale, causal=False, softcap=softcap),
        warmup=warmup, repeat=repeat)
    results["fa_v2"] = {"mean_ms": med, "min_ms": mn, "tflops": fl / med / 1e9}

    try:  # baseline 对 GQA 的支持视 CANN 版本，失败置 ERR 不影响 fa 列
        npu_fa_fwd(q_nb, k_nb, v_nb, head_num=nheads, scale=scale)
        med, mn = time_callable(
            lambda: npu_fa_fwd(q_nb, k_nb, v_nb, head_num=nheads, scale=scale),
            warmup=warmup, repeat=repeat)
        results["npu_fusion_attn"] = {"mean_ms": med, "min_ms": mn, "tflops": fl / med / 1e9}
    except Exception as e:
        results["npu_fusion_attn"] = {"error": f"{type(e).__name__}: {e}"}

    fa_r, nb_r = results["fa_v2"], results.get("npu_fusion_attn", {})
    if "mean_ms" in nb_r and nb_r["mean_ms"] > 0:
        results["speedup_fa_over_baseline"] = nb_r["mean_ms"] / fa_r["mean_ms"]
    return results


def main():
    ap = argparse.ArgumentParser(description="FA v2 (GQA) vs npu_fusion_attention bench")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 4, 16, 64, 256, 320, 1024])
    ap.add_argument("--seqlen", type=int, nargs="+", default=[32])
    ap.add_argument("--nheads", type=int, default=32)
    ap.add_argument("--kv-heads", dest="kv_heads", type=int, default=4)
    ap.add_argument("--headdim", type=int, default=64, choices=[64, 128])
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--softcap", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=100)
    ap.add_argument("--device", default=2, type=int)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    torch.npu.set_device(args.device)

    print(f"# bench GQA: device={torch_npu.npu.get_device_name()} dtype={args.dtype} "
          f"H={args.nheads} Hkv={args.kv_heads} D={args.headdim} "
          f"warmup={args.warmup} repeat={args.repeat}")
    print(f"# grid: batch={args.batch} seqlen={args.seqlen}")
    print()
    header = f"{'batch':>6} {'seqlen':>7} {'fa_ms':>9} {'base_ms':>9} {'speedup':>8}"
    print(header)
    print("-" * len(header))

    rows = []
    first_err = False
    for b in args.batch:
        for s in args.seqlen:
            r = run_one(b, s, args.nheads, args.kv_heads, args.headdim, dtype,
                        args.softcap, args.warmup, args.repeat, args.device)
            fa, nb = r.get("fa_v2", {}), r.get("npu_fusion_attn", {})
            fa_ms, nb_ms = fa.get("mean_ms"), nb.get("mean_ms")
            sp = r.get("speedup_fa_over_baseline")
            rows.append({"batch": b, "seqlen": s, "nheads": args.nheads,
                         "kv_heads": args.kv_heads, "headdim": args.headdim,
                         "fa_v2_mean_ms": fa_ms, "fa_v2_min_ms": fa.get("min_ms"),
                         "baseline_mean_ms": nb_ms, "baseline_min_ms": nb.get("min_ms"),
                         "speedup_fa_over_baseline": sp,
                         "baseline_error": nb.get("error")})
            if not first_err and nb.get("error"):
                print(f"# [baseline 首个错误] b={b} s={s}: {nb['error']}")
                first_err = True
            fa_s = f"{fa_ms:9.3f}" if fa_ms is not None else f"{'ERR':>9}"
            nb_s = f"{nb_ms:9.3f}" if nb_ms is not None else f"{'ERR':>9}"
            sp_s = f"{sp:8.2f}x" if sp is not None else f"{'-':>8}"
            print(f"{b:>6} {s:>7} {fa_s} {nb_s} {sp_s}")

    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"# CSV -> {args.out}")


if __name__ == "__main__":
    main()

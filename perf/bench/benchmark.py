# 用法：
#   python perf/bench/bench2.py                       # MHA，跑默认小 seqlen 网格
#   python perf/bench/bench2.py --batch 1 8 64 256 --seqlen 64 128 512 2048
#   python perf/bench/bench2.py --kv-heads 2          # GQA（默认 0 = MHA）
#   python perf/bench/bench2.py --dtype bf16 --headdim 128 --causal
#   python perf/bench/bench2.py --out results/bench.csv --json results/bench.json

import argparse
import json
import math
import statistics
import sys

import torch
import torch_npu  

from flash_attn_npu import flash_attn_func

def fa_fwd(q, k, v, *, scale, causal, softcap=0.0, dropout_p=0, window_size=(-1,-1)):
    flash_attn_func(
        q, k, v,
        dropout_p,
        scale,
        causal,
        window_size,
        softcap,
    )

# FIXME: 暂不支持causal
def npu_fa_fwd(q_bnsd, k_bnsd, v_bnsd, *, head_num, scale, causal):
    torch_npu.npu_fusion_attention(
        q_bnsd, k_bnsd, v_bnsd, head_num, "BNSD",
        scale=scale,
    )


def time_callable(fn, *, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()  # 确保预热完成

    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        start_event.record()
        fn()
        end_event.record()
        torch.npu.synchronize()  # 等待所有操作完成
        times.append(start_event.elapsed_time(end_event))  # 返回毫秒
    return statistics.mean(times), min(times)


def run_one(batch, seqlen, nheads, kvheads, headdim, dtype, causal, softcap,
            warmup, repeat, device):
    torch.manual_seed(42)
    # 我方 FA：BSHD [B, Sq, H, D]，k/v 用 kvheads（=nheads 即 MHA；更少即 GQA）
    q_fa = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device).contiguous()
    k_fa = torch.randn(batch, seqlen, kvheads, headdim, dtype=dtype, device=device).contiguous()
    v_fa = torch.randn(batch, seqlen, kvheads, headdim, dtype=dtype, device=device).contiguous()
    
    # baseline：BNSD [B, H, S, D]
    q_nb = q_fa.transpose(1, 2).contiguous()
    k_nb = k_fa.transpose(1, 2).contiguous()
    v_nb = v_fa.transpose(1, 2).contiguous()

    scale = 1.0 / math.sqrt(headdim)

    results = {}
    # FA
    fa_fwd(q_fa, k_fa, v_fa, scale=scale, causal=causal, softcap=softcap)  # 触发 JIT/编译路径
    med, mn = time_callable(
        lambda: fa_fwd(q_fa, k_fa, v_fa, scale=scale, causal=causal, softcap=softcap),
        warmup=warmup, repeat=repeat,
    )
    results["fa"] = {"mean_ms": med, "min_ms": mn}

    # CANN baseline
    try:
        npu_fa_fwd(q_nb, k_nb, v_nb, head_num=nheads, scale=scale, causal=causal)
        med, mn = time_callable(
            lambda: npu_fa_fwd(q_nb, k_nb, v_nb, head_num=nheads, scale=scale, causal=causal),
            warmup=warmup, repeat=repeat,
        )
        results["npu_fusion_attn"] = {"mean_ms": med, "min_ms": mn}
    except Exception as e:
        results["npu_fusion_attn"] = {"error": f"{type(e).__name__}: {e}"}

    fa_r = results.get("fa", {})
    nb_r = results.get("npu_fusion_attn", {})
    if "mean_ms" in fa_r and "mean_ms" in nb_r and nb_r["mean_ms"] > 0:
        results["speedup_fa_over_baseline"] = nb_r["mean_ms"] / fa_r["mean_ms"]
    return results



def main():
    # batch = 320     NUM_HEADS = 24    NUM_HEADS_KV = 4   HEAD_DIM = 128 AVG_SEQ_LEN = 150      MIN_SEQ_LEN = 32   MAX_SEQ_LEN = 512
    
    ap = argparse.ArgumentParser(description="FA v2 vs npu_fusion_attention bench (small-seq, large-batch)")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 4, 16, 64, 256, 320, 1024, 2048, 4096], help="batch 列表（默认聚焦大 batch）")
    ap.add_argument("--seqlen", type=int, nargs="+", default=[18, 24, 35, 50, 64, 77, 128], help="seqlen 列表（默认含小 seqlen）")
    ap.add_argument("--nheads", type=int, default=24)
    ap.add_argument("--kv-heads", dest="kv_heads", type=int, default=4)
    ap.add_argument("--headdim", type=int, default=128)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--causal", action="store_true", default=False)
    ap.add_argument("--softcap", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=100)
    ap.add_argument("--device", default=2, type=int)
    ap.add_argument("--out", default=None, help="CSV 输出路径")
    ap.add_argument("--json", default=None, help="JSON 输出路径")
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = args.device
    kv_heads = args.kv_heads if args.kv_heads > 0 else args.nheads
    torch.npu.set_device(device)

    print(f"# bench: device={torch_npu.npu.get_device_name()} dtype={args.dtype} "
          f"nheads={args.nheads} kv_heads={kv_heads} headdim={args.headdim} causal={args.causal} "
          f"softcap={args.softcap} warmup={args.warmup} repeat={args.repeat}")
    print(f"# grid: batch={args.batch} seqlen={args.seqlen}")
    print()
    header = f"{'batch':>6} {'seqlen':>7} {'fa_ms':>9} {'base_ms':>9} {'speedup':>8}"
    print(header)
    print("-" * len(header))

    rows = []
    for b in args.batch:
        for s in args.seqlen:
            r = run_one(b, s, args.nheads, kv_heads, args.headdim, dtype, args.causal, args.softcap,
                        args.warmup, args.repeat, device)
            fa, nb = r.get("fa", {}), r.get("npu_fusion_attn", {})
            fa_ms = fa.get("mean_ms")
            nb_ms = nb.get("mean_ms")
            sp = r.get("speedup_fa_over_baseline")
            row = {
                "batch": b, "seqlen": s, "nheads": args.nheads, "kv_heads": kv_heads,
                "headdim": args.headdim, "dtype": args.dtype, 
                "fa_mean_ms": fa_ms, "fa_min_ms": fa.get("min_ms"),
                "baseline_mean_ms": nb_ms, "baseline_min_ms": nb.get("min_ms"),
                "speedup_fa_over_baseline": sp,
            }
            rows.append(row)
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
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"config": vars(args), "rows": rows}, f, indent=2)
        print(f"# JSON -> {args.json}")


if __name__ == "__main__":
    main()

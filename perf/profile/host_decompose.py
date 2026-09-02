#!/usr/bin/env python3
"""host 开销分解测量（#63）：裁决 #60(async 0.39ms) 与 bench 反推(~2.7ms) 谁成立。

四种口径 × {grad(bench 条件), no_grad}：
  async   : fn() 返回即计时（不含 kernel）—— host 真实开销（#60 的口径）
  event   : bench2.py 同款（start.record/fn/end.record/sync）—— 设备时间线含空泡
  syncwall: perf_counter 包住 fn()+synchronize —— 端到端
  thru    : 连续 N 次后 sync —— host 可被流水隐藏时 ≈ kernel，否则 ≈ host

判读：
  async ≈ 2.7ms  ⇒ host 真瓶颈（加 --profile 用 cProfile 定位热函数）
  async ≈ 0.4ms  ⇒ host 不是问题，4.2ms 来自设备时间线空泡（enqueue 延迟/事件语义）

用法：
  python perf/profile/host_decompose.py --device 4                 # MHA（bench2.py 同款）
  python perf/profile/host_decompose.py --device 4 --kv-heads 2    # GQA
  python perf/profile/host_decompose.py --device 4 --profile       # 附 cProfile top20
"""
import argparse
import cProfile
import io
import pstats
import statistics
import time

import torch
import torch_npu  # noqa: F401

from flash_attn_npu import flash_attn_func


def measure(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    # async：fn() 返回即计时（每轮 sync 排空队列）
    t = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1e3)
        torch.npu.synchronize()
    async_ms = statistics.mean(t)

    # event：bench2.py 同款
    s = torch.npu.Event(enable_timing=True)
    e = torch.npu.Event(enable_timing=True)
    t = []
    for _ in range(repeat):
        s.record()
        fn()
        e.record()
        torch.npu.synchronize()
        t.append(s.elapsed_time(e))
    event_ms = statistics.mean(t)

    # syncwall
    t = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        t.append((time.perf_counter() - t0) * 1e3)
    sync_ms = statistics.mean(t)

    # thru：吞吐模式
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.npu.synchronize()
    thru_ms = (time.perf_counter() - t0) * 1e3 / repeat
    return async_ms, event_ms, sync_ms, thru_ms


def main():
    ap = argparse.ArgumentParser(description="host 开销分解（#63）")
    ap.add_argument("--device", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seqlen", type=int, default=128)
    ap.add_argument("--nheads", type=int, default=8)
    ap.add_argument("--kv-heads", dest="kv_heads", type=int, default=0,
                    help="0 = MHA（bench2.py 同款）")
    ap.add_argument("--headdim", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--bench-like", action="store_true",
                    help="复现 bench2.py 内存状态：计时前先建 3 份 BNSD 转置副本并保持存活")
    ap.add_argument("--profile", action="store_true", help="附 cProfile top20（50 次调用）")
    args = ap.parse_args()

    kv = args.kv_heads or args.nheads
    torch.npu.set_device(args.device)
    q = torch.randn(args.batch, args.seqlen, args.nheads, args.headdim,
                    dtype=torch.float16).npu().contiguous()
    k = torch.randn(args.batch, args.seqlen, kv, args.headdim,
                    dtype=torch.float16).npu().contiguous()
    v = torch.randn(args.batch, args.seqlen, kv, args.headdim,
                    dtype=torch.float16).npu().contiguous()
    scale = args.headdim ** -0.5

    keepalive = []
    if args.bench_like:
        # bench2.py run_one：fa 计时前 q_nb/k_nb/v_nb 转置副本已建好且存活
        keepalive = [q.transpose(1, 2).contiguous(),
                     k.transpose(1, 2).contiguous(),
                     v.transpose(1, 2).contiguous()]
    print(f"# mem_allocated={torch.npu.memory_allocated() / 2**30:.2f} GiB "
          f"mem_reserved={torch.npu.memory_reserved() / 2**30:.2f} GiB "
          f"({'bench-like' if keepalive else 'clean'})")

    def fn():
        flash_attn_func(q, k, v, 0.0, scale, causal=False,
                        window_size=(-1, -1), softcap=0.0)

    print(f"# host_decompose: device={args.device} B={args.batch} S={args.seqlen} "
          f"H={args.nheads} kvH={kv} D={args.headdim} "
          f"warmup={args.warmup} repeat={args.repeat}")
    print(f"{'mode':>8} {'async_ms':>9} {'event_ms':>9} {'sync_ms':>9} {'thru_ms':>9}")
    for label, use_nograd in [("grad", False), ("nograd", True)]:
        if use_nograd:
            with torch.no_grad():
                r = measure(fn, args.warmup, args.repeat)
        else:
            r = measure(fn, args.warmup, args.repeat)
        print(f"{label:>8} {r[0]:9.3f} {r[1]:9.3f} {r[2]:9.3f} {r[3]:9.3f}")

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(50):
            fn()
        torch.npu.synchronize()
        pr.disable()
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(20)
        print(buf.getvalue())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""FA kernel 性能剖析用最小测试（msopprof / MindStudio 配套）。

设计原则（profiler 友好）：
  - 只跑一次 kernel launch——timeline 干净，MindStudio 可视化不混淆
  - 无预热（profiler 自己有 warm-up 机制；预热会在 timeline 里产生重复段）
  - 无 GPU→CPU 同步调用（不污染 timeline）
  - 输入 contiguous、BSHD 布局

用法：
  # 测我们的 SplitB（默认）
  python perf/profile/fa_test.py --batch 1024 --seqlen 128 --nheads 8

  # 测 torch_npu baseline（npu_fusion_attention）
  python perf/profile/fa_test.py --test-torch --batch 1024 --seqlen 128 --nheads 8
"""
import argparse
import os
import sys

import torch
import torch_npu

from flash_attn_npu import flash_attn_func


def run_ours(q, k, v, scale, args):
    """我们的 SplitB / 旧路径"""
    if args.disable_splitb:
        os.environ["FLASH_ATTN_DISABLE_SPLITB"] = "1"
    else:
        os.environ.pop("FLASH_ATTN_DISABLE_SPLITB", None)

    if args.single_core:
        os.environ["FLASH_ATTN_SPLITB_SINGLE_CORE"] = "1"
    else:
        os.environ.pop("FLASH_ATTN_SPLITB_SINGLE_CORE", None)

    flash_attn_func(q, k, v, 0.0, scale, causal=False, window_size=(-1, -1), softcap=0.0)
    return "splitb"


def run_torch(q, k, v, scale, args):
    """torch_npu baseline（npu_fusion_attention，BNSD 布局）"""
    # npu_fusion_attention 要求 BNSD [B, H, S, D]
    q_bnsd = q.transpose(1, 2).contiguous()
    k_bnsd = k.transpose(1, 2).contiguous()
    v_bnsd = v.transpose(1, 2).contiguous()
    torch_npu.npu_fusion_attention(q_bnsd, k_bnsd, v_bnsd, args.nheads, "BNSD", scale=scale)
    return "torch_npu"


def main():
    ap = argparse.ArgumentParser(description="FA kernel 性能剖析用最小测试（单次调用）")
    ap.add_argument("--device", type=int, default=6)
    ap.add_argument("--test-torch", action="store_true",
                    help="测 torch_npu baseline（npu_fusion_attention）而非我们的算子")
    ap.add_argument("--disable-splitb", action="store_true")
    ap.add_argument("--single-core", action="store_true")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seqlen", type=int, default=128)
    ap.add_argument("--nheads", type=int, default=8)
    ap.add_argument("--kv-heads", dest="kv_heads", type=int, default=2,
                    help="0 = MHA（等于 nheads）")
    ap.add_argument("--headdim", type=int, default=128, choices=[64, 128])
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    args = ap.parse_args()

    kv_heads = args.kv_heads if args.kv_heads > 0 else args.nheads
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    torch.npu.set_device(args.device)

    # 输入统一 BSHD [B, S, H, D]（我们的布局；torch 侧在 run_torch 内转 BNSD）
    q = torch.randn(args.batch, args.seqlen, args.nheads, args.headdim,
                    dtype=dtype).contiguous().npu()
    k = torch.randn(args.batch, args.seqlen, kv_heads, args.headdim,
                    dtype=dtype).contiguous().npu()
    v = torch.randn(args.batch, args.seqlen, kv_heads, args.headdim,
                    dtype=dtype).contiguous().npu()
    scale = 1.0 / (args.headdim ** 0.5)

    if args.test_torch:
        backend = run_torch(q, k, v, scale, args)
    else:
        backend = run_ours(q, k, v, scale, args)

    print(f"[DONE] B={args.batch} S={args.seqlen} H={args.nheads} kvH={kv_heads} "
          f"D={args.headdim} backend={backend}")


if __name__ == "__main__":
    sys.exit(main() or 0)

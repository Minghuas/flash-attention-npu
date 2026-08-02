#!/usr/bin/env python3
"""
debug/test_bwd.py — ALiBi backward-pass debug script.

检查 ALiBi 开启时反向 (dQ, dK, dV) 是否正确。
  - 被测(NPU kernel)：flash_attn_func / flash_attn_varlen_func → torch.autograd.grad
  - 基准(CPU golden)：golden_bsnd_bwd_from_fwd / golden_tnd_bwd_from_fwd

两条 bwd 内核路径覆盖：
  - 默认(BSND, flash_attn_func)        → Epilogue1 (FAGGeneral)，mha_bwd 必走
  - --varlen(flash_attn_varlen_func)   → sq==sk & headdim==128 & 非local 时走 Epilogue2 (FAGVarlenOpt)

Usage:
    # BSND（走 Epilogue1）
    python debug/test_bwd.py --batch 2 --heads 4 --kv-h 2 --sq 256 --sk 512 --hdim 128
    # varlen（sq==sk & headdim==128 → 走 Epilogue2）
    python debug/test_bwd.py --varlen --batch 1 --heads 4 --kv-h 2 --sq 512 --sk 512 --hdim 128 --causal
"""

import sys, os, argparse
import torch
import torch_npu

TESTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from flash_attn_npu_v2 import flash_attn_func, flash_attn_varlen_func
from fa_small_op_golden import golden_bsnd_bwd_from_fwd, golden_tnd_bwd_from_fwd

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_alibi_slopes(batch_size, num_heads):
    """0.5 / (2^h) per head。返回 [batch_size, num_heads] CPU tensor。"""
    _h = torch.tensor([0.5 / (2 ** h) for h in range(num_heads)], dtype=torch.float32)
    return _h.unsqueeze(0).repeat(batch_size, 1)


def make_random_inputs_bsnd(B, Sq, Sk, H, KVH, D, dtype):
    """BSND: q[B,Sq,H,D] k/v[B,Sk,KVH,D] dout[B,Sq,H,D]，CPU，[-1,1]。"""
    gen = torch.Generator(); gen.manual_seed(42)
    q = (2 * torch.rand(B, Sq, H, D, generator=gen) - 1).to(dtype)
    k = (2 * torch.rand(B, Sk, KVH, D, generator=gen) - 1).to(dtype)
    v = (2 * torch.rand(B, Sk, KVH, D, generator=gen) - 1).to(dtype)
    dout = (2 * torch.rand(B, Sq, H, D, generator=gen) - 1).to(dtype)
    return q, k, v, dout


def get_cu_seqlens(seqlens):
    cu = [0]
    for s in seqlens:
        cu.append(cu[-1] + s)
    return torch.tensor(cu, dtype=torch.int32)


# ---------------------------------------------------------------------------
# 两条路径：各自返回 (dq_fa, dk_fa, dv_fa, dq_g, dk_g, dv_g)
# ---------------------------------------------------------------------------

def run_bsnd(args):
    B, H, KVH = args.batch, args.heads, args.kv_heads
    Sq, Sk, D = args.sq, args.sk, args.hdim
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    scale = 1.0 / (D ** 0.5)

    q_cpu, k_cpu, v_cpu, dout_cpu = make_random_inputs_bsnd(B, Sq, Sk, H, KVH, D, dtype)
    slopes_cpu = None if args.no_alibi else make_alibi_slopes(B, H)

    q = q_cpu.clone().npu().requires_grad_(True)
    k = k_cpu.clone().npu().requires_grad_(True)
    v = v_cpu.clone().npu().requires_grad_(True)
    dout = dout_cpu.clone().npu()
    slopes = slopes_cpu.npu() if slopes_cpu is not None else None

    out_fa, lse_fa, _ = flash_attn_func(
        q, k, v, dropout_p=0.0, softmax_scale=scale, softcap=args.softcap,
        causal=args.causal, window_size=(-1, -1), alibi_slopes=slopes,
        return_attn_probs=True)
    torch.npu.synchronize()
    print(f"    [BSND] FWD done — out={tuple(out_fa.shape)} lse={tuple(lse_fa.shape)}")

    dq_fa, dk_fa, dv_fa = torch.autograd.grad(out_fa, (q, k, v), dout)
    torch.npu.synchronize()
    print(f"    [BSND] BWD done — dq={tuple(dq_fa.shape)}")

    dq_g, dk_g, dv_g = golden_bsnd_bwd_from_fwd(
        q_cpu, k_cpu, v_cpu, dout_cpu, out_fa.detach().cpu(), lse_fa.detach().cpu(),
        H, KVH, scale, args.softcap, 0.0, args.causal, -1, -1,
        gtype=torch.float64, alibi_slopes=slopes_cpu)
    print(f"    [BSND] GOLDEN done")
    return dq_fa, dk_fa, dv_fa, dq_g, dk_g, dv_g


def run_varlen(args):
    B, H, KVH = args.batch, args.heads, args.kv_heads
    Sq, Sk, D = args.sq, args.sk, args.hdim
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    scale = 1.0 / (D ** 0.5)

    # 生成 BSND 形状再 reshape 成 TND [total, H, D]，保证与 BSND 同一份数据
    q_b, k_b, v_b, dout_b = make_random_inputs_bsnd(B, Sq, Sk, H, KVH, D, dtype)
    # TND: q[B*Sq,H,D] k/v[B*Sk,KVH,D]
    q_cpu = q_b.reshape(B * Sq, H, D)
    k_cpu = k_b.reshape(B * Sk, KVH, D)
    v_cpu = v_b.reshape(B * Sk, KVH, D)
    dout_cpu = dout_b.reshape(B * Sq, H, D)

    # varlen 用 [1, H]（与 test_flash_attn_npu_v2_bwd.py 一致）
    slopes_cpu = None if args.no_alibi else make_alibi_slopes(1, H)

    q = q_cpu.clone().npu().requires_grad_(True)
    k = k_cpu.clone().npu().requires_grad_(True)
    v = v_cpu.clone().npu().requires_grad_(True)
    dout = dout_cpu.clone().npu()
    slopes = slopes_cpu.npu() if slopes_cpu is not None else None

    seqlens_q = [Sq] * B
    seqlens_k = [Sk] * B
    cu_q = get_cu_seqlens(seqlens_q).npu()
    cu_k = get_cu_seqlens(seqlens_k).npu()

    out_fa, lse_fa, _ = flash_attn_varlen_func(
        q, k, v, cu_q, cu_k, Sq, Sk,
        dropout_p=0.0, softmax_scale=scale, softcap=args.softcap,
        causal=args.causal, window_size=(-1, -1), alibi_slopes=slopes,
        return_attn_probs=True)
    torch.npu.synchronize()
    print(f"    [VARLEN] FWD done — out={tuple(out_fa.shape)} lse={tuple(lse_fa.shape)}")

    dq_fa, dk_fa, dv_fa = torch.autograd.grad(out_fa, (q, k, v), dout)
    torch.npu.synchronize()
    print(f"    [VARLEN] BWD done — dq={tuple(dq_fa.shape)}")

    dq_g, dk_g, dv_g = golden_tnd_bwd_from_fwd(
        q_cpu, k_cpu, v_cpu, dout_cpu, out_fa.detach().cpu(), lse_fa.detach().cpu(),
        H, KVH, seqlens_q, seqlens_k, scale, args.softcap, 0.0,
        args.causal, -1, -1, gtype=torch.float64, alibi_slopes=slopes_cpu)
    print(f"    [VARLEN] GOLDEN done")
    return dq_fa, dk_fa, dv_fa, dq_g, dk_g, dv_g


# ---------------------------------------------------------------------------
# 形状无关的比较（BSND [B,Sq,H,D] 与 varlen [total,H,D] 都用 reshape(-1,H,D) 归一）
# ---------------------------------------------------------------------------

def compare_and_report(dq_fa, dk_fa, dv_fa, dq_g, dk_g, dv_g, H, KVH, D):
    dq_fa_c = dq_fa.cpu().float()
    dk_fa_c = dk_fa.cpu().float()
    dv_fa_c = dv_fa.cpu().float()
    dq_diff = (dq_fa_c - dq_g.float()).abs()
    dk_diff = (dk_fa_c - dk_g.float()).abs()
    dv_diff = (dv_fa_c - dv_g.float()).abs()

    print()
    print(f"    ---- dQ/dK/dV comparison ----")
    print(f"    dQ  max diff: {dq_diff.max().item():.6f}  mean: {dq_diff.mean().item():.6f}")
    print(f"    dK  max diff: {dk_diff.max().item():.6f}  mean: {dk_diff.mean().item():.6f}")
    print(f"    dV  max diff: {dv_diff.max().item():.6f}  mean: {dv_diff.mean().item():.6f}")

    THRESH = 0.5
    # reshape(-1, H, D)：BSND [B,Sq,H,D]→[B*Sq,H,D]，varlen [total,H,D] 不变。
    # 两种布局 H 都在 dim=-2，故按 head 取 max 对两者都成立。
    # 注意：本 torch 版本 .max(dim=...) 不支持 tuple，只能单个 int，故分两次 reduce。
    print(f"\n    ---- per-head overview (diff > {THRESH} marked ⚠) ----")
    print(f"    [dQ: {H} heads]  [dK/dV: {KVH} kv-heads]")
    dq_by_head = dq_diff.reshape(-1, H, D).max(dim=2).values.max(dim=0).values
    for h in range(H):
        v = dq_by_head[h].item()
        print(f"    dQ head={h}: max_diff={v:.4f} {'⚠' if v > THRESH else '✓'}")
    dk_by_head = dk_diff.reshape(-1, KVH, D).max(dim=2).values.max(dim=0).values
    dv_by_head = dv_diff.reshape(-1, KVH, D).max(dim=2).values.max(dim=0).values
    for h in range(KVH):
        mk, mv = dk_by_head[h].item(), dv_by_head[h].item()
        m = max(mk, mv)
        print(f"    dK/dV kv-head={h}: max_diff={m:.4f} (dK={mk:.4f} dV={mv:.4f})"
              f" {'⚠' if m > THRESH else '✓'}")

    # worst mismatches（topk，O(1) Python 循环）。按 [N, H, D] 解索引。
    MAX_ROWS = 50

    def _worst(name, diff_t, fa_t, g_t, nhead):
        print(f"\n    ---- {name} worst mismatches (first {MAX_ROWS}, idx=(n,h,d)) ----")
        print(f"    {'n':>7} {'h':>3} {'d':>3} | {'kernel':>12} {'golden':>12} {'diff':>12}")
        print(f"    {'---':>7} {'---':>3} {'---':>3} | {'---':>12} {'---':>12} {'---':>12}")
        flat = diff_t.reshape(-1, nhead, D)
        flat1 = flat.reshape(-1)
        k = min(MAX_ROWS, flat1.numel())
        if k > 0:
            top_vals, top_idx = flat1.topk(k)
            nh = nhead * D
            fa_r = fa_t.reshape(-1, nhead, D)
            g_r = g_t.float().reshape(-1, nhead, D)
            for i in range(k):
                rem = int(top_idx[i].item())
                n = rem // nh; rem %= nh
                h = rem // D; rem %= D
                d_ = rem
                print(f"    {n:>7} {h:>3} {d_:>3} | {fa_r[n,h,d_].item():>12.4f}"
                      f" {g_r[n,h,d_].item():>12.4f} {top_vals[i].item():>12.4f}")

    _worst("dQ", dq_diff, dq_fa_c, dq_g, H)
    _worst("dK", dk_diff, dk_fa_c, dk_g, KVH)
    _worst("dV", dv_diff, dv_fa_c, dv_g, KVH)

    print()
    all_ok = (dq_diff.max().item() < THRESH and
              dk_diff.max().item() < THRESH and
              dv_diff.max().item() < THRESH)
    print(f"    {'✅ PASS' if all_ok else '⚠ MISMATCH'}"
          f" — backward {'matches' if all_ok else 'differs from'} golden reference")
    print('-' * 55)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args):
    H, KVH, D = args.heads, args.kv_heads, args.hdim
    slopes_hdr = None if args.no_alibi else make_alibi_slopes(1, H)[0].tolist()
    print(f"==> ALiBi BWD test  mode={'VARLEN' if args.varlen else 'BSND'}"
          f" batch={args.batch} H={H} kvH={KVH} Sq={args.sq} Sk={args.sk} D={D}"
          f" dtype={args.dtype} causal={args.causal} softcap={args.softcap}"
          f" alibi={not args.no_alibi}")
    if slopes_hdr is not None:
        print(f"    slopes: {slopes_hdr}")
    print()

    if args.varlen:
        dq_fa, dk_fa, dv_fa, dq_g, dk_g, dv_g = run_varlen(args)
    else:
        dq_fa, dk_fa, dv_fa, dq_g, dk_g, dv_g = run_bsnd(args)

    compare_and_report(dq_fa, dk_fa, dv_fa, dq_g, dk_g, dv_g, H, KVH, D)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ALiBi backward debug script")
    p.add_argument("--varlen", action="store_true", help="用 flash_attn_varlen_func (TND)，可触发 Epilogue2")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--kv-h", "--kv-heads", dest="kv_heads", type=int, default=1)
    p.add_argument("--sq", type=int, default=64, help="query seqlen")
    p.add_argument("--sk", type=int, default=64, help="key seqlen")
    p.add_argument("--hdim", type=int, default=32, help="head dim")
    p.add_argument("--causal", action="store_true")
    p.add_argument("--no-alibi", action="store_true")
    p.add_argument("--softcap", type=float, default=0.0)
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    args = p.parse_args()
    main(args)

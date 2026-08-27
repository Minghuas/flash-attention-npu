#!/usr/bin/env python3
"""t1 失败诊断（devlog #44.49）：定位错误元素的坐标模式。

用法（不改 kernel，直接复用已装 .so）：
  python debug/test_splitb_t1_diag.py gqa2     # case9: G=2 LSE-only 错
  python debug/test_splitb_t1_diag.py b128     # case2: B128 H8 64×64 O 小比例错
  python debug/test_splitb_t1_diag.py odd      # case8: 33×47 非对齐 O 错
  python debug/test_splitb_t1_diag.py fullg4   # case5: G4 128×128 满tile O 错
"""
import sys
import torch
import torch_npu
from flash_attn_npu import flash_attn_func

torch.manual_seed(42)


def run(B, H, Hkv, Sq, Sk, D, dtype=torch.float16):
    q = (-5 + 10 * torch.rand(B, Sq, H, D)).to(dtype).npu()
    k = (-5 + 10 * torch.rand(B, Sk, Hkv, D)).to(dtype).npu()
    v = (-5 + 10 * torch.rand(B, Sk, Hkv, D)).to(dtype).npu()
    scale = D ** -0.5
    out, lse, _ = flash_attn_func(q, k, v, 0.0, causal=False, window_size=(-1, -1),
                                  softcap=0.0, return_attn_probs=True)
    # fp32 golden：O 与 LSE（与 ref_flash_attention 的 interm fp32 同级）
    g = H // Hkv
    s = torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float().repeat_interleave(g, dim=2)) * scale
    lse_ref = torch.logsumexp(s, dim=-1)                      # [B,H,Sq]
    p = torch.softmax(s, dim=-1)
    o_ref = torch.einsum('bhqk,bkhd->bqhd', p, v.float().repeat_interleave(g, dim=2))
    torch.npu.synchronize()
    return (out.float().cpu(), lse.float().cpu(), o_ref.cpu(), lse_ref.cpu())


def report(name, got, ref, coord_names, tol=2e-2):
    bad = (got - ref).abs() > tol
    n = bad.sum().item()
    total = bad.numel()
    print(f"[{name}] mismatch {n}/{total} ({100.0 * n / total:.2f}%)")
    if n == 0:
        return
    idx = bad.nonzero()
    print(f"  max_abs_err={float((got - ref).abs().max()):.4f}")
    # 坐标分布：每个轴的错误值集合（前 12 个）
    for ax, cn in enumerate(coord_names):
        vals = sorted(set(idx[:, ax].tolist()))
        preview = vals[:12]
        print(f"  axis {cn}: {len(vals)} distinct, first {preview}"
              + (" ..." if len(vals) > 12 else ""))
    # 前 8 个错误样点的 got/ref
    for row in idx[:8]:
        c = tuple(row.tolist())
        print(f"    at {dict(zip(coord_names, c))}: got={float(got[c]):.4f} ref={float(ref[c]):.4f}")


CASES = {
    "gqa2":   dict(B=4,  H=8, Hkv=4, Sq=32, Sk=32, D=64),   # LSE-only 错（case9 缩小 B）
    "gqa8":   dict(B=4,  H=8, Hkv=1, Sq=32, Sk=32, D=64),   # case10 G=8
    "b128":   dict(B=128, H=8, Hkv=8, Sq=64, Sk=64, D=64),  # case2 小比例 O 错
    "b64":    dict(B=64, H=8, Hkv=8, Sq=64, Sk=64, D=64),   # 对照：t1 里 fwd case14 同形状过
    "odd":    dict(B=8,  H=8, Hkv=8, Sq=33, Sk=47, D=64),   # case8 非对齐
    "fullg4": dict(B=4,  H=4, Hkv=1, Sq=128, Sk=128, D=64), # case5 满tile GQA
    "fullh4": dict(B=4,  H=4, Hkv=4, Sq=128, Sk=128, D=64), # 对照：case4 同形状过
    "d128":   dict(B=8,  H=8, Hkv=8, Sq=64, Sk=64, D=128),  # case12
    "sk96":   dict(B=8,  H=8, Hkv=8, Sq=32, Sk=96, D=64),   # case7
}

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "gqa2"
    kw = CASES[tag]
    out, lse, o_ref, lse_ref = run(**kw)
    B, H, Sq = kw["B"], kw["H"], kw["Sq"]
    report(f"{tag}/O",   out.permute(0, 2, 1, 3), o_ref.permute(0, 2, 1, 3), ["b", "h", "s", "d"])
    report(f"{tag}/LSE", lse, lse_ref, ["b", "h", "s"])

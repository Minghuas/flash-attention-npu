# Copyright (c) 2026, Minghua Shen.

"""Attention 测试共用的数值比较规则。"""

import torch


def assert_fa_close(actual, ref, pt, *, softcap=0.0, name="out"):
    """按照 Tri Dao 的双基准规则比较实现结果。

    ``ref`` 是高精度 reference，``pt`` 是 PyTorch 基准；两者之间的最大差异
    用于估计数值误差范围：``max|actual - ref| <= rtol * max|pt - ref| + 2 * ULP(ref)``。
    """
    actual = actual.detach().cpu()
    ref = ref.detach().cpu()
    pt = pt.detach().cpu()
    assert actual.shape == ref.shape == pt.shape, (
        f"{name}: shape mismatch actual={tuple(actual.shape)} "
        f"ref={tuple(ref.shape)} pt={tuple(pt.shape)}"
    )
    if actual.numel() == 0:
        assert ref.numel() == 0 and pt.numel() == 0, f"{name}: empty shape mismatch"
        return

    # NaN 一律视为错误；共同的 inf 则先做精确的语义比较，不能参与减法。
    assert not torch.isnan(actual).any(), f"{name}: actual contains NaN"
    assert not torch.isnan(ref).any(), f"{name}: ref contains NaN"
    assert not torch.isnan(pt).any(), f"{name}: pt contains NaN"

    actual_inf = torch.isinf(actual)
    ref_inf = torch.isinf(ref)
    pt_inf = torch.isinf(pt)
    assert torch.equal(actual_inf, ref_inf), f"{name}: actual/ref inf mask mismatch"
    assert torch.equal(pt_inf, ref_inf), f"{name}: pt/ref inf mask mismatch"
    if ref_inf.any():
        assert torch.equal(actual[ref_inf], ref[ref_inf]), (
            f"{name}: actual/ref inf value mismatch"
        )
        assert torch.equal(pt[ref_inf], ref[ref_inf]), f"{name}: pt/ref inf value mismatch"

    # 混合 finite / inf 的 tensor 只在有限元素上做数值误差比较。
    finite = torch.isfinite(ref)
    if not finite.any():
        return
    actual = actual[finite]
    ref = ref[finite]
    pt = pt[finite]

    rtol = 3.0 if softcap != 0.0 else 2.0
    # 必须在 ref 原始 dtype 上计算 ULP，先转 float32 会丢失 fp16/bf16
    # 实际需要测量的 ULP。
    ulp = (ref + 0.3 - 0.3 - ref).abs().max().item()
    diff = (actual - ref).abs()
    max_diff = diff.max().item()
    pt_diff = (pt - ref).abs().max().item()
    tolerance = rtol * pt_diff + 2.0 * ulp
    assert max_diff <= tolerance, (
        f"{name}: max|actual-ref|={max_diff} exceeds "
        f"{rtol} * max|pt-ref|={pt_diff} + 2*ULP(ref)={2.0 * ulp} "
        f"(softcap={softcap})"
    )

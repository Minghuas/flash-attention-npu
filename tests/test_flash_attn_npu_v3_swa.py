# Copyright (c) 2026, Minghua Shen.
"""Dedicated SWA (sliding-window attention) tests for flash_attn_npu_3 / Ascend950.

Covers the MASK_SWA kernel path in csrc/ascend950/flash_attn_npu_3:
  - host normalize (window >= max_kv_seqlen → -1; causal forces right=0;
    infinite local side → max_kv_seqlen)
  - kvBaseTile / qBaseTile = 128 band skip + Pre/Next mask on tile edges
  - narrow window (Pre+Next on the same KV tile)
  - empty window (InitEmptyWindowOut: O=0, LSE=+inf)
  - partial-empty Q rows (rescale_o ZeroInvalidSwaRows / delStartRow, delEndRow)
  - negative window sizes
  - Sq vs Sk mismatch, decode, GQA hang regression, TND/BSND, paged KV, varlen

Stress suite (test_fa_swa_stress): long KV (up to 10240), large batch/GQA, many
128-tile KV loops, paged/varlen at scale. Run separately:

  pytest tests/test_flash_attn_npu_v3_swa.py -m stress -q
"""

import torch
import torch_npu
import pytest

from flash_attn_npu_3 import flash_attn_with_kvcache
from test_flash_attn_npu_v3 import ref_flash_attention

# Kernel tile used by fai_kernel.cpp MASK_SWA (kvBaseTile_ / qBaseTile_).
_SWA_TILE = 128


def _is_swa_local(kv_seqlen, is_causal, window_size_left, window_size_right):
    """Match mha_fwd.cpp host normalize; True iff MASK_SWA (is_local) is launched."""
    wL, wR = window_size_left, window_size_right
    if kv_seqlen > 0 and wL >= kv_seqlen:
        wL = -1
    if kv_seqlen > 0 and wR >= kv_seqlen:
        wR = -1
    if is_causal:
        wR = 0
    is_causal_g = (wL < 0 and wR == 0)
    return (wL >= 0 or wR >= 0) and not is_causal_g


def _normalize_window(kv_seqlen, is_causal, window_size_left, window_size_right):
    """Golden window after the same host normalize as mha_fwd.cpp."""
    wL, wR = window_size_left, window_size_right
    if kv_seqlen > 0 and wL >= kv_seqlen:
        wL = -1
    if kv_seqlen > 0 and wR >= kv_seqlen:
        wR = -1
    if is_causal:
        wR = 0
    is_causal_g = (wL < 0 and wR == 0)
    is_local_g = (wL >= 0 or wR >= 0) and not is_causal_g
    if is_local_g:
        if wL < 0:
            wL = kv_seqlen
        if wR < 0:
            wR = kv_seqlen
    return is_causal_g, is_local_g, wL, wR


def create_binary_matrix(q_seqlen, kv_seqlen, pre_token, next_token):
    """SWA band mask: True = masked. Same geometry as test_flash_attn_npu_v3."""
    pre = kv_seqlen - q_seqlen - pre_token
    nxt = kv_seqlen - q_seqlen + next_token
    i = torch.arange(q_seqlen).unsqueeze(1)
    j = torch.arange(kv_seqlen).unsqueeze(0)
    delta = j - i
    return (delta < pre) | (delta > nxt)


def _build_swa_cases():
    """~300 unique SWA geometries. Only cases that stay on the is_local/MASK_SWA path."""
    bf16, fp16 = torch.bfloat16, torch.float16
    cases = []
    seen = set()

    def add(
        dtype=bf16,
        batch=1,
        heads=1,
        kv_heads=1,
        q_seqlen=128,
        kv_seqlen=128,
        head_size=128,
        cache_mode=0,
        is_causal=False,
        layout="BSND",
        is_varied=False,
        window_size_left=0,
        window_size_right=0,
    ):
        if heads % kv_heads != 0:
            return
        if is_varied and layout != "TND":
            return
        if not _is_swa_local(kv_seqlen, is_causal, window_size_left, window_size_right):
            return
        item = (
            dtype, batch, heads, kv_heads, q_seqlen, kv_seqlen, head_size,
            cache_mode, is_causal, layout, is_varied,
            window_size_left, window_size_right,
        )
        if item in seen:
            return
        seen.add(item)
        cases.append(item)

    # --- G1: window vs kv tile 128 (aligned Sq=Sk) ---
    # 0/1: diagonal / 1-token; 63/64: mid-tile; 127/128/129: tile boundary ±1.
    for sq, sk in ((128, 128), (256, 256)):
        for wL in (0, 1, 63, 64, 127, 128, 129):
            for wR in (0, 1, 63, 64, 127, 128):
                add(q_seqlen=sq, kv_seqlen=sk, window_size_left=wL, window_size_right=wR)
    for wL in (0, 1, 128, 256, 511):
        for wR in (0, 1, 128):
            add(q_seqlen=512, kv_seqlen=512, window_size_left=wL, window_size_right=wR)

    # --- G2: causal SWA (Mistral-style). causal forces wR=0; finite wL → MASK_SWA. ---
    for sq, sk, wL in (
        (1, 128, 1), (1, 128, 64), (1, 128, 127),
        (1, 256, 32), (1, 256, 128), (1, 256, 255),
        (1, 512, 64), (1, 512, 128), (1, 512, 256),
        (1, 1024, 128), (1, 1024, 512), (1, 1024, 1023),
        (16, 256, 32), (16, 256, 128), (32, 512, 64),
        (64, 512, 128), (64, 512, 256),
        (128, 128, 1), (128, 128, 64), (128, 128, 127),
        (128, 256, 64), (128, 256, 128), (128, 256, 255),
        (256, 256, 1), (256, 256, 127), (256, 256, 128), (256, 256, 255),
        (256, 512, 128), (256, 512, 256),
        (512, 512, 128), (512, 512, 256), (512, 512, 511),
    ):
        add(q_seqlen=sq, kv_seqlen=sk, is_causal=True,
            window_size_left=wL, window_size_right=0)

    # --- G3: one-sided infinite (host fills the -1 side with max_kv_seqlen) ---
    for sk in (128, 256, 512):
        for wL in (1, 64, 128):
            add(q_seqlen=sk, kv_seqlen=sk, window_size_left=wL, window_size_right=-1)
        for wR in (1, 64, 128):
            add(q_seqlen=sk, kv_seqlen=sk, window_size_left=-1, window_size_right=wR)

    # --- G4: negative windows + fully empty band (InitEmptyWindowOut) ---
    for sq, sk, wL, wR in (
        (512, 512, 508, -256),
        (1024, 1024, -128, 256),
        (256, 256, -1, 64),
        (256, 256, -32, 64),
        (256, 256, -64, 32),
        (128, 128, -64, 32),
        (128, 128, -127, 64),
        (128, 128, -128, 64),   # -wL >= Sq → skip all KV tiles
        (128, 128, -256, 32),
        (128, 256, -128, 0),
        (64, 256, -64, 32),
        (128, 128, 64, -32),
        (128, 128, 64, -64),
        (128, 128, 32, -128),   # -wR >= Sk → noSkipKvS=0
        (128, 128, 64, -128),
        (256, 256, 128, -256),
        (256, 128, -64, 32),
        (128, 512, -200, 64),
        (256, 256, -255, 10),
        (128, 128, -128, 1),
        (256, 256, 8, -256),
    ):
        add(q_seqlen=sq, kv_seqlen=sk, window_size_left=wL, window_size_right=wR)

    # --- G5: decode Sq=1; KV lengths around tile boundaries ---
    for sk in (64, 65, 127, 128, 129, 255, 256, 257, 512, 1024):
        for wL, wR, causal in (
            (0, 0, False),
            (64, 0, True),
            (128, 0, True),
            (0, 32, False),
            (32, 32, False),
        ):
            add(q_seqlen=1, kv_seqlen=sk, is_causal=causal,
                window_size_left=wL, window_size_right=wR)

    # --- G6: Q-tile tails (qBaseTile=128) + last-tile remainder ---
    for sq in (2, 3, 7, 15, 16, 31, 32, 63, 65, 96, 127, 129, 192, 255, 257):
        add(q_seqlen=sq, kv_seqlen=512, is_causal=True,
            window_size_left=128, window_size_right=0)
        add(q_seqlen=sq, kv_seqlen=512, window_size_left=64, window_size_right=64)

    # --- G7: Sq vs Sk mismatch, including Sq>>Sk fully-masked rows ---
    for sq, sk, wL, wR, causal in (
        (16, 256, 32, 0, True), (16, 256, 64, 64, False),
        (32, 512, 64, 0, True), (64, 1024, 128, 0, True),
        (128, 512, 64, 64, False), (256, 1024, 256, 0, True),
        (256, 64, 32, 32, False), (256, 128, 64, 0, True),
        (512, 128, 64, 64, False), (1024, 128, 497, 265, False),
        (127, 128, 32, 0, True), (129, 128, 64, 32, False),
        (255, 256, 128, 0, True), (257, 256, 64, 64, False),
        (65, 256, 32, 16, False), (192, 256, 128, 0, True),
        (8, 1024, 256, 0, True), (48, 512, 64, 64, False),
        (384, 128, 32, 0, True), (200, 64, 16, 16, False),
    ):
        add(q_seqlen=sq, kv_seqlen=sk, is_causal=causal,
            window_size_left=wL, window_size_right=wR)

    # --- G8: GQA / MQA, plus rowLoopNum>1 hang regression (EVENT_ID0) ---
    for heads, kv_heads, sq, sk, wL, wR, causal in (
        (2, 1, 1, 256, 64, 0, True),
        (4, 1, 16, 256, 64, 64, False),
        (4, 2, 1, 512, 128, 0, True),
        (8, 2, 16, 256, 32, 32, False),
        (8, 4, 32, 512, 128, 0, True),
        (16, 4, 1, 512, 64, 0, True),
        (16, 8, 16, 256, 64, 64, False),
        (32, 8, 1, 1024, 256, 0, True),
        (64, 1, 1, 1024, 542, 647, True),
        (128, 1, 1, 1024, 542, 647, True),
        (6, 2, 16, 512, 128, 0, True),
        (24, 2, 8, 256, 64, 64, False),
    ):
        add(heads=heads, kv_heads=kv_heads, q_seqlen=sq, kv_seqlen=sk,
            is_causal=causal, window_size_left=wL, window_size_right=wR)

    # --- G9: layout / cache / dtype / D=64 on a small seed of distinct geometries ---
    seed_specs = (
        dict(q_seqlen=128, kv_seqlen=128, window_size_left=0, window_size_right=0),
        dict(q_seqlen=256, kv_seqlen=256, window_size_left=127, window_size_right=128),
        dict(q_seqlen=256, kv_seqlen=256, is_causal=True, window_size_left=128, window_size_right=0),
        dict(q_seqlen=1, kv_seqlen=512, is_causal=True, window_size_left=64, window_size_right=0),
        dict(q_seqlen=128, kv_seqlen=128, window_size_left=-128, window_size_right=64),
        dict(q_seqlen=256, kv_seqlen=64, window_size_left=32, window_size_right=32),
        dict(heads=8, kv_heads=2, q_seqlen=16, kv_seqlen=256, window_size_left=32, window_size_right=32),
        dict(q_seqlen=129, kv_seqlen=512, window_size_left=64, window_size_right=64),
    )
    for spec in seed_specs:
        add(**spec, layout="TND")
        add(**spec, cache_mode=1)
        add(**spec, dtype=fp16)
        add(**spec, head_size=64)

    # --- G10: batch > 1 ---
    for batch in (2, 4):
        add(batch=batch, q_seqlen=128, kv_seqlen=256, is_causal=True,
            window_size_left=64, window_size_right=0)
        add(batch=batch, q_seqlen=64, kv_seqlen=256,
            window_size_left=32, window_size_right=32)
        add(batch=batch, heads=4, kv_heads=2, q_seqlen=16, kv_seqlen=256,
            is_causal=True, window_size_left=128, window_size_right=0)

    # --- G11: varlen Q (TND); host window is vs max_kv_seqlen across the batch ---
    add(batch=2, layout="TND", is_varied=True, q_seqlen=32, kv_seqlen=256,
        is_causal=True, window_size_left=64, window_size_right=0)
    add(batch=4, layout="TND", is_varied=True, q_seqlen=64, kv_seqlen=512,
        window_size_left=32, window_size_right=32)
    add(batch=2, layout="TND", is_varied=True, heads=4, kv_heads=2,
        q_seqlen=16, kv_seqlen=256, window_size_left=8, window_size_right=8)
    add(batch=3, layout="TND", is_varied=True, q_seqlen=128, kv_seqlen=256,
        is_causal=True, window_size_left=127, window_size_right=0)
    add(batch=2, layout="TND", is_varied=True, q_seqlen=8, kv_seqlen=128,
        window_size_left=0, window_size_right=16)
    add(batch=4, layout="TND", is_varied=True, dtype=fp16, q_seqlen=32, kv_seqlen=256,
        is_causal=True, window_size_left=32, window_size_right=0)

    # --- G12: paged KV × both layouts ---
    for sq, sk, wL, wR, causal in (
        (1, 256, 64, 0, True),
        (16, 256, 32, 32, False),
        (128, 128, 64, 64, False),
        (1, 1024, 512, 0, True),
        (256, 256, 127, 0, True),
    ):
        add(cache_mode=1, q_seqlen=sq, kv_seqlen=sk, is_causal=causal,
            window_size_left=wL, window_size_right=wR, layout="BSND")
        add(cache_mode=1, q_seqlen=sq, kv_seqlen=sk, is_causal=causal,
            window_size_left=wL, window_size_right=wR, layout="TND")

    # --- G13: host collapse (window == kv-1 stays SWA; window == kv collapses one side) ---
    for sk in (128, 256):
        add(q_seqlen=sk, kv_seqlen=sk, window_size_left=sk, window_size_right=1)
        add(q_seqlen=sk, kv_seqlen=sk, window_size_left=sk - 1, window_size_right=0)
        add(q_seqlen=sk, kv_seqlen=sk, window_size_left=0, window_size_right=sk)
        add(q_seqlen=sk, kv_seqlen=sk, window_size_left=sk + 1, window_size_right=32)

    # --- G14: narrow bidirectional windows (Pre+Next mask on the same KV tile) ---
    for sq, sk in ((128, 128), (256, 256)):
        for w in (2, 4, 8, 16):
            add(q_seqlen=sq, kv_seqlen=sk, window_size_left=w, window_size_right=w)
        add(q_seqlen=sq, kv_seqlen=sk, window_size_left=16, window_size_right=0)
        add(q_seqlen=sq, kv_seqlen=sk, window_size_left=0, window_size_right=8)

    return cases


def _build_swa_stress_cases():
    """Large-scale SWA stress geometries (long KV, large batch/GQA, many KV tiles)."""
    bf16, fp16 = torch.bfloat16, torch.float16
    cases = []
    seen = set()

    def add(
        dtype=bf16,
        batch=1,
        heads=1,
        kv_heads=1,
        q_seqlen=128,
        kv_seqlen=128,
        head_size=128,
        cache_mode=0,
        is_causal=False,
        layout="BSND",
        is_varied=False,
        window_size_left=0,
        window_size_right=0,
    ):
        if heads % kv_heads != 0:
            return
        if is_varied and layout != "TND":
            return
        if not _is_swa_local(kv_seqlen, is_causal, window_size_left, window_size_right):
            return
        item = (
            dtype, batch, heads, kv_heads, q_seqlen, kv_seqlen, head_size,
            cache_mode, is_causal, layout, is_varied,
            window_size_left, window_size_right,
        )
        if item in seen:
            return
        seen.add(item)
        cases.append(item)

    # --- S1: long-KV decode (many 128-tile KV loops, narrow effective band) ---
    for sk in (2048, 4096, 8192, 10240):
        for wL, causal in (
            (64, True), (128, True), (256, True), (512, True), (1024, True),
            (64, False), (128, False),
        ):
            add(q_seqlen=1, kv_seqlen=sk, is_causal=causal,
                window_size_left=wL, window_size_right=0 if causal else 32)
        add(q_seqlen=1, kv_seqlen=sk, is_causal=True,
            window_size_left=542, window_size_right=647)
        add(q_seqlen=1, kv_seqlen=sk, window_size_left=0, window_size_right=0)

    # --- S2: long aligned prefill, causal SWA (Mistral / Llama-style) ---
    for sq, sk in ((1024, 1024), (2048, 2048), (4096, 4096)):
        for wL in (127, 128, 129, 255, 256, 257, 511, 512, 513, 1023):
            add(q_seqlen=sq, kv_seqlen=sk, is_causal=True,
                window_size_left=wL, window_size_right=0)

    # --- S3: long KV + moderate Sq (multi Q-tile × many KV-tile) ---
    for sk in (2048, 4096, 8192):
        for sq in (128, 256, 512, 1024):
            add(q_seqlen=sq, kv_seqlen=sk, is_causal=True,
                window_size_left=128, window_size_right=0)
            add(q_seqlen=sq, kv_seqlen=sk, window_size_left=64, window_size_right=64)
        add(q_seqlen=129, kv_seqlen=sk, is_causal=True,
            window_size_left=256, window_size_right=0)

    # --- S4: Sq >> Sk at scale (partial-empty / fully-masked Q rows) ---
    for sq, sk, wL, wR, causal in (
        (1024, 128, 497, 265, False),
        (2048, 128, 497, 265, False),
        (1024, 256, 128, 0, True),
        (2048, 256, 256, 0, True),
        (512, 64, 32, 32, False),
        (4096, 128, 64, 0, True),
    ):
        add(q_seqlen=sq, kv_seqlen=sk, is_causal=causal,
            window_size_left=wL, window_size_right=wR)

    # --- S5: large GQA decode (rowLoopNum>1 hang regression at scale) ---
    for heads, sk in (
        (64, 4096), (128, 4096), (256, 4096), (512, 4096),
        (64, 8192), (128, 8192), (512, 8192),
    ):
        add(heads=heads, kv_heads=1, q_seqlen=1, kv_seqlen=sk, is_causal=True,
            window_size_left=542, window_size_right=647)
        add(heads=heads, kv_heads=1, q_seqlen=1, kv_seqlen=sk, is_causal=True,
            window_size_left=512, window_size_right=0)

    # --- S6: large batch × long KV ---
    for batch in (8, 16):
        add(batch=batch, q_seqlen=128, kv_seqlen=4096, is_causal=True,
            window_size_left=256, window_size_right=0)
        add(batch=batch, q_seqlen=64, kv_seqlen=2048,
            window_size_left=128, window_size_right=128)
        add(batch=batch, heads=8, kv_heads=2, q_seqlen=32, kv_seqlen=4096,
            is_causal=True, window_size_left=512, window_size_right=0)

    # --- S7: paged KV, long context ---
    for sk in (4096, 8192, 10240):
        for sq, wL, causal in (
            (1, 512, True), (1, 542, True), (128, 256, True), (256, 128, False),
        ):
            add(cache_mode=1, q_seqlen=sq, kv_seqlen=sk, is_causal=causal,
                window_size_left=wL, window_size_right=647 if wL == 542 else (0 if causal else 64),
                layout="BSND")
            add(cache_mode=1, q_seqlen=sq, kv_seqlen=sk, is_causal=causal,
                window_size_left=wL, window_size_right=647 if wL == 542 else (0 if causal else 64),
                layout="TND")

    # --- S8: TND varlen at scale ---
    add(batch=8, layout="TND", is_varied=True, q_seqlen=128, kv_seqlen=4096,
        is_causal=True, window_size_left=256, window_size_right=0)
    add(batch=8, layout="TND", is_varied=True, q_seqlen=64, kv_seqlen=8192,
        window_size_left=128, window_size_right=128)
    add(batch=16, layout="TND", is_varied=True, heads=4, kv_heads=2,
        q_seqlen=32, kv_seqlen=4096, is_causal=True, window_size_left=512, window_size_right=0)
    add(batch=4, layout="TND", is_varied=True, dtype=fp16, q_seqlen=256, kv_seqlen=8192,
        is_causal=True, window_size_left=1024, window_size_right=0)

    # --- S9: negative / empty window at long Sk ---
    for sk in (2048, 4096, 8192):
        add(q_seqlen=128, kv_seqlen=sk, window_size_left=-128, window_size_right=512)
        add(q_seqlen=256, kv_seqlen=sk, window_size_left=512, window_size_right=-256)
        add(q_seqlen=64, kv_seqlen=sk, window_size_left=-64, window_size_right=32)

    # --- S10: dtype/head/layout variants on representative stress seeds ---
    stress_seeds = (
        dict(q_seqlen=1, kv_seqlen=8192, is_causal=True, window_size_left=512, window_size_right=0),
        dict(q_seqlen=1024, kv_seqlen=4096, is_causal=True, window_size_left=512, window_size_right=0),
        dict(heads=128, kv_heads=1, q_seqlen=1, kv_seqlen=4096, is_causal=True,
             window_size_left=542, window_size_right=647),
        dict(batch=8, q_seqlen=128, kv_seqlen=4096, is_causal=True,
             window_size_left=256, window_size_right=0),
    )
    for spec in stress_seeds:
        add(**spec, layout="TND")
        add(**spec, cache_mode=1)
        add(**spec, dtype=fp16)
        add(**spec, head_size=64)

    return cases


def _build_swa_stress_cases():
    """~100+ large-scale SWA stress geometries (long KV, large batch/GQA, many tiles)."""
    bf16, fp16 = torch.bfloat16, torch.float16
    cases = []
    seen = set()

    def add(
        dtype=bf16,
        batch=1,
        heads=1,
        kv_heads=1,
        q_seqlen=128,
        kv_seqlen=128,
        head_size=128,
        cache_mode=0,
        is_causal=False,
        layout="BSND",
        is_varied=False,
        window_size_left=0,
        window_size_right=0,
    ):
        if heads % kv_heads != 0:
            return
        if is_varied and layout != "TND":
            return
        if not _is_swa_local(kv_seqlen, is_causal, window_size_left, window_size_right):
            return
        item = (
            dtype, batch, heads, kv_heads, q_seqlen, kv_seqlen, head_size,
            cache_mode, is_causal, layout, is_varied,
            window_size_left, window_size_right,
        )
        if item in seen:
            return
        seen.add(item)
        cases.append(item)

    # --- S1: long KV decode (many 128-tile KV loops) ---
    for sk in (2048, 4096, 8192, 10240):
        for wL, wR, causal in (
            (128, 0, True),
            (256, 0, True),
            (512, 0, True),
            (1024, 0, True),
            (542, 647, True),   # hang-regression window on long KV
            (64, 64, False),
            (128, 128, False),
            (0, 256, False),
        ):
            add(q_seqlen=1, kv_seqlen=sk, is_causal=causal,
                window_size_left=wL, window_size_right=wR)
        # tile-boundary windows
        for wL in (127, 128, 129, 255, 256, 257, 511, 512, 513):
            add(q_seqlen=1, kv_seqlen=sk, is_causal=True,
                window_size_left=wL, window_size_right=0)

    # --- S2: long aligned prefill (Sq=Sk, many Q+KV tiles) ---
    for sk in (1024, 2048, 4096):
        for wL in (128, 256, 512, 1023, 1024, 1025):
            add(q_seqlen=sk, kv_seqlen=sk, is_causal=True,
                window_size_left=wL, window_size_right=0)
        for wL, wR in ((128, 128), (256, 256), (512, 64), (64, 512)):
            add(q_seqlen=sk, kv_seqlen=sk, window_size_left=wL, window_size_right=wR)

    # --- S3: large Sq prefill on long KV (multi Q-tile × multi KV-tile) ---
    for sq, sk in ((512, 4096), (1024, 4096), (1024, 8192), (2048, 4096)):
        add(q_seqlen=sq, kv_seqlen=sk, is_causal=True,
            window_size_left=512, window_size_right=0)
        add(q_seqlen=sq, kv_seqlen=sk, is_causal=True,
            window_size_left=256, window_size_right=0)
        add(q_seqlen=sq, kv_seqlen=sk, window_size_left=128, window_size_right=128)
        add(q_seqlen=sq, kv_seqlen=sk, window_size_left=64, window_size_right=256)

    # --- S4: Sq >> Sk on long context (partial-empty / fully-masked Q rows) ---
    for sq, sk, wL, wR in (
        (1024, 128, 497, 265),
        (2048, 256, 128, 128),
        (4096, 512, 256, 0),
        (8192, 128, 64, 64),
    ):
        add(q_seqlen=sq, kv_seqlen=sk, window_size_left=wL, window_size_right=wR)

    # --- S5: extreme GQA decode / prefill ---
    for heads, kv_heads, sq, sk, wL, wR, causal in (
        (512, 1, 1, 4096, 542, 647, True),
        (512, 1, 1, 8192, 512, 0, True),
        (256, 1, 1, 4096, 256, 0, True),
        (128, 4, 1, 4096, 128, 0, True),
        (64, 8, 16, 2048, 128, 0, True),
        (128, 16, 32, 4096, 256, 0, True),
        (32, 4, 64, 2048, 64, 64, False),
    ):
        add(heads=heads, kv_heads=kv_heads, q_seqlen=sq, kv_seqlen=sk,
            is_causal=causal, window_size_left=wL, window_size_right=wR)

    # --- S6: large batch ---
    for batch in (8, 16):
        add(batch=batch, q_seqlen=128, kv_seqlen=2048, is_causal=True,
            window_size_left=128, window_size_right=0)
        add(batch=batch, q_seqlen=64, kv_seqlen=4096,
            window_size_left=64, window_size_right=64)
        add(batch=batch, heads=8, kv_heads=2, q_seqlen=16, kv_seqlen=2048,
            is_causal=True, window_size_left=256, window_size_right=0)
    add(batch=8, q_seqlen=1, kv_seqlen=8192, is_causal=True,
        window_size_left=512, window_size_right=0)

    # --- S7: paged KV on long sequences ---
    for sq, sk, wL, wR, causal in (
        (1, 4096, 512, 0, True),
        (1, 8192, 256, 0, True),
        (128, 4096, 128, 128, False),
        (256, 4096, 256, 0, True),
        (512, 8192, 512, 0, True),
    ):
        add(cache_mode=1, q_seqlen=sq, kv_seqlen=sk, is_causal=causal,
            window_size_left=wL, window_size_right=wR, layout="BSND")
        add(cache_mode=1, q_seqlen=sq, kv_seqlen=sk, is_causal=causal,
            window_size_left=wL, window_size_right=wR, layout="TND")

    # --- S8: TND varlen stress (varied q/kv per batch, long max kv) ---
    add(batch=8, layout="TND", is_varied=True, q_seqlen=128, kv_seqlen=4096,
        is_causal=True, window_size_left=256, window_size_right=0)
    add(batch=8, layout="TND", is_varied=True, q_seqlen=64, kv_seqlen=8192,
        window_size_left=128, window_size_right=128)
    add(batch=16, layout="TND", is_varied=True, heads=4, kv_heads=2,
        q_seqlen=32, kv_seqlen=2048, is_causal=True, window_size_left=512, window_size_right=0)
    add(batch=4, layout="TND", is_varied=True, dtype=fp16, q_seqlen=256, kv_seqlen=4096,
        is_causal=True, window_size_left=128, window_size_right=0)

    # --- S9: D=64 + fp16 on long KV ---
    for sq, sk in ((1, 4096), (256, 4096), (1, 8192)):
        add(head_size=64, dtype=fp16, q_seqlen=sq, kv_seqlen=sk,
            is_causal=True, window_size_left=256, window_size_right=0)
        add(head_size=64, q_seqlen=sq, kv_seqlen=sk,
            window_size_left=128, window_size_right=128)

    # --- S10: negative / empty window at scale ---
    for sq, sk, wL, wR in (
        (128, 4096, -512, 256),
        (256, 8192, 512, -1024),
        (512, 4096, -4096, 32),
        (1024, 8192, 8, -8192),
    ):
        add(q_seqlen=sq, kv_seqlen=sk, window_size_left=wL, window_size_right=wR)

    return cases


SWA_CASES = _build_swa_cases()
SWA_STRESS_CASES = _build_swa_stress_cases()


def _case_id(c):
    dtype, batch, heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, \
        is_causal, layout, is_varied, wL, wR = c
    dt = "bf16" if dtype == torch.bfloat16 else "fp16"
    return (
        f"{dt}_B{batch}_H{heads}x{kv_heads}_q{q_seqlen}_k{kv_seqlen}_D{head_size}"
        f"_{layout}_cm{cache_mode}_c{int(is_causal)}_v{int(is_varied)}_w{wL}_{wR}"
    )


def test_swa_case_count():
    assert 280 <= len(SWA_CASES) <= 340, len(SWA_CASES)


def test_swa_stress_case_count():
    assert 100 <= len(SWA_STRESS_CASES) <= 200, len(SWA_STRESS_CASES)


def _gather_paged_kv(key_cache, value_cache, block_table, kv_seqlen, block_size, kv_heads, head_size):
    keys, values = [], []
    key_cpu = key_cache.detach().cpu()
    value_cpu = value_cache.detach().cpu()
    table = block_table.cpu()
    for j in range(kv_seqlen):
        block_number = int(table[j // block_size])
        block_offset = j % block_size
        keys.append(key_cpu[block_number, block_offset].reshape(kv_heads, head_size))
        values.append(value_cpu[block_number, block_offset].reshape(kv_heads, head_size))
    return torch.stack(keys, dim=0), torch.stack(values, dim=0)


def _run_swa_test(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size,
    cache_mode, is_causal, layout, is_varied, window_size_left, window_size_right,
):
    """Shared NPU vs golden SWA check used by functional and stress tests."""
    name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
    if "Ascend950" in name and head_size not in (64, 128):
        pytest.skip("Ascend950 support head_size in 64,128")
    if is_varied and layout != "TND":
        pytest.skip("is_varied requires TND (varlen-q) layout")

    q_min_range, q_max_range = -5.0, 5.0
    kv_min_range, kv_max_range = -5.0, 5.0
    block_size = _SWA_TILE
    max_num_blocks_per_seq = (kv_seqlen + block_size - 1) // block_size
    num_blocks = max(64, max_num_blocks_per_seq * batch_size)
    gen = torch.Generator().manual_seed(1234)

    if is_varied:
        q_sequences = torch.randint(low=1, high=q_seqlen + 1, size=(batch_size,), generator=gen).tolist()
        kv_sequences = [
            int(torch.randint(low=q, high=kv_seqlen + 1, size=(1,), generator=gen))
            for q in q_sequences
        ]
    else:
        q_sequences = [q_seqlen] * batch_size
        kv_sequences = [kv_seqlen] * batch_size
    t_q_sum = sum(q_sequences)
    t_kv_sum = sum(kv_sequences)

    if layout == "BSND":
        query = (q_min_range + (q_max_range - q_min_range) * torch.rand(
            batch_size, q_seqlen, num_heads, head_size, generator=gen)).to(data_type).npu()
    else:
        query = (q_min_range + (q_max_range - q_min_range) * torch.rand(
            t_q_sum, num_heads, head_size, generator=gen)).to(data_type).npu()

    if cache_mode == 1:
        key_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(
            num_blocks, block_size, kv_heads, head_size, generator=gen)).to(data_type).npu()
        value_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(
            num_blocks, block_size, kv_heads, head_size, generator=gen)).to(data_type).npu()
        block_tables = torch.tensor(
            [[max_num_blocks_per_seq * i + j for j in range(max_num_blocks_per_seq)]
             for i in range(batch_size)],
            dtype=torch.int32,
        ).npu()
    else:
        if layout == "BSND":
            key_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(
                batch_size, kv_seqlen, kv_heads, head_size, generator=gen)).to(data_type).npu()
            value_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(
                batch_size, kv_seqlen, kv_heads, head_size, generator=gen)).to(data_type).npu()
        else:
            key_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(
                t_kv_sum, kv_heads, head_size, generator=gen)).to(data_type).npu()
            value_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(
                t_kv_sum, kv_heads, head_size, generator=gen)).to(data_type).npu()
        block_tables = None

    scale = 1.0 / (head_size ** 0.5)
    kv_seqlen_list = torch.tensor(kv_sequences, dtype=torch.int32).npu()
    # Host normalizes window vs max KV length across the batch (mha_fwd.cpp).
    max_kv_in_batch = max(kv_sequences)
    is_causal_golden, is_local_golden, wL_golden, wR_golden = _normalize_window(
        max_kv_in_batch, is_causal, window_size_left, window_size_right)

    new_q_seqlen_list = None
    new_q_seqlen_list_cpu = None
    new_kv_seqlen_list_cpu = None
    if layout == "TND":
        new_q_seqlen_list_cpu = [0]
        pre_seq_sum = 0
        for i in range(batch_size):
            pre_seq_sum += q_sequences[i]
            new_q_seqlen_list_cpu.append(pre_seq_sum)
        new_q_seqlen_list = torch.tensor(new_q_seqlen_list_cpu, dtype=torch.int32).npu()
        if cache_mode == 0:
            new_kv_seqlen_list_cpu = [0]
            pre_seq_sum = 0
            for i in range(batch_size):
                pre_seq_sum += kv_sequences[i]
                new_kv_seqlen_list_cpu.append(pre_seq_sum)

    out_out, softmax_lse, *rest = flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        None,
        None,
        None,
        rotary_cos=None,
        rotary_sin=None,
        cache_seqlens=kv_seqlen_list,
        cache_batch_idx=None,
        cache_leftpad=None,
        page_table=block_tables,
        cu_seqlens_q=new_q_seqlen_list,
        cu_seqlens_k_new=None,
        max_seqlen_q=q_seqlen,
        rotary_seqlens=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        softmax_scale=None,
        causal=is_causal,
        window_size=[window_size_left, window_size_right],
        attention_chunk=0,
        softcap=0.0,
        rotary_interleaved=False,
        scheduler_metadata=None,
        num_splits=0,
        pack_gqa=None,
        sm_margin=0,
        return_softmax_lse=True,
    )

    if layout == "BSND":
        golden_out = torch.empty((batch_size, q_seqlen, num_heads, head_size), dtype=data_type)
        golden_lseL = torch.empty((batch_size, num_heads, q_seqlen), dtype=torch.float32)
    else:
        golden_out = torch.empty((t_q_sum, num_heads, head_size), dtype=data_type)
        golden_lseL = torch.empty((num_heads, t_q_sum), dtype=torch.float32)

    for i in range(batch_size):
        q_seqlen_per_batch = q_sequences[i]
        kv_seqlen_per_batch = kv_sequences[i]
        atten_mask = None
        if is_causal_golden:
            atten_mask = torch.triu(
                torch.ones(q_seqlen_per_batch, kv_seqlen_per_batch),
                diagonal=(kv_seqlen_per_batch - q_seqlen_per_batch + 1),
            ).bool()
        elif is_local_golden:
            atten_mask = create_binary_matrix(
                q_seqlen_per_batch, kv_seqlen_per_batch, wL_golden, wR_golden,
            )

        if layout == "BSND":
            query_cpu_per_batch = query.detach().cpu()[i]
            if cache_mode == 1:
                key_cache_per_batch, value_cache_per_batch = _gather_paged_kv(
                    key_cache, value_cache, block_tables[i], kv_seqlen_per_batch,
                    block_size, kv_heads, head_size)
            else:
                key_cache_per_batch = key_cache.detach().cpu()[i]
                value_cache_per_batch = value_cache.detach().cpu()[i]
        else:
            query_cpu_per_batch = query.detach().cpu()[new_q_seqlen_list_cpu[i]: new_q_seqlen_list_cpu[i + 1]]
            if cache_mode == 0:
                key_cache_per_batch = key_cache.detach().cpu()[
                    new_kv_seqlen_list_cpu[i]: new_kv_seqlen_list_cpu[i + 1]]
                value_cache_per_batch = value_cache.detach().cpu()[
                    new_kv_seqlen_list_cpu[i]: new_kv_seqlen_list_cpu[i + 1]]
            else:
                key_cache_per_batch, value_cache_per_batch = _gather_paged_kv(
                    key_cache, value_cache, block_tables[i], kv_seqlen_per_batch,
                    block_size, kv_heads, head_size)

        output, golden_lse = ref_flash_attention(
            query_cpu_per_batch, key_cache_per_batch, value_cache_per_batch,
            scale, atten_mask, data_type, 0.0)
        out = output.reshape(q_seqlen_per_batch, num_heads, head_size)
        if is_local_golden and atten_mask is not None:
            fully_masked = atten_mask.all(dim=-1)
            out[fully_masked, :, :] = 0
            golden_lse[:, fully_masked] = torch.inf
        if layout == "BSND":
            golden_out[i:i + 1] = out
            golden_lseL[i:i + 1] = golden_lse.reshape(1, num_heads, q_seqlen_per_batch)
        else:
            golden_out[new_q_seqlen_list[i]: new_q_seqlen_list[i + 1]] = out
            golden_lseL[:, new_q_seqlen_list[i]: new_q_seqlen_list[i + 1]] = \
                golden_lse.reshape(num_heads, q_seqlen_per_batch)

    rtol, atol = 1e-2, 1e-2
    torch.testing.assert_close(out_out.cpu(), golden_out.cpu(), rtol=rtol, atol=atol)
    if "Ascend910" in name:
        torch.testing.assert_close(softmax_lse.cpu(), golden_lseL.cpu(), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, "
    "cache_mode, is_causal, layout, is_varied, window_size_left, window_size_right",
    SWA_CASES,
    ids=[_case_id(c) for c in SWA_CASES],
)
def test_fa_swa(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size,
    cache_mode, is_causal, layout, is_varied, window_size_left, window_size_right,
):
    _run_swa_test(
        data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size,
        cache_mode, is_causal, layout, is_varied, window_size_left, window_size_right,
    )


@pytest.mark.stress
@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, "
    "cache_mode, is_causal, layout, is_varied, window_size_left, window_size_right",
    SWA_STRESS_CASES,
    ids=[_case_id(c) for c in SWA_STRESS_CASES],
)
def test_fa_swa_stress(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size,
    cache_mode, is_causal, layout, is_varied, window_size_left, window_size_right,
):
    _run_swa_test(
        data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size,
        cache_mode, is_causal, layout, is_varied, window_size_left, window_size_right,
    )

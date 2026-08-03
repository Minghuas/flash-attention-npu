# Copyright (c) 2026, Minghua Shen.

from itertools import combinations, product

import pytest
import torch
import torch_npu

from flash_attn_npu import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)


data_types = (torch.float16, torch.bfloat16)
batch_sizes = (1, 2)
head_cases = (
    (1, 1),  # MHA, single head
    (4, 4),  # MHA, multi-head
    (4, 2),  # GQA
    (4, 1),  # MQA
)
shape_cases = (
    # (q_seqlen, kv_seqlen, head_size)
    (128, 128, 64),       # Small, aligned, equal sequence lengths
    (512, 512, 128),      # Medium, aligned, equal sequence lengths
    (512, 1024, 128),     # Query shorter than KV
    (1024, 512, 192),     # Query longer than KV (non-causal only)
    (513, 777, 256),      # Unaligned tail tiles
)
causal_options = (False, True)

test_cases = [
    pytest.param(
        data_type,
        batch_size,
        num_heads,
        kv_heads,
        q_seqlen,
        kv_seqlen,
        head_size,
        is_causal,
        case_id,
        id=(
            f"{str(data_type).split('.')[-1]}-b{batch_size}-"
            f"h{num_heads}-kvh{kv_heads}-sq{q_seqlen}-sk{kv_seqlen}-"
            f"d{head_size}-{'causal' if is_causal else 'noncausal'}"
        ),
    )
    for case_id, (
        data_type,
        batch_size,
        (num_heads, kv_heads),
        (q_seqlen, kv_seqlen, head_size),
        is_causal,
    ) in enumerate(
        product(
            data_types,
            batch_sizes,
            head_cases,
            shape_cases,
            causal_options,
        ),
        start=1,
    )
    if not (is_causal and q_seqlen > kv_seqlen)
]


def _random_npu_tensor(shape, data_type, generator):
    tensor = 2 * torch.rand(shape, generator=generator) - 1
    return tensor.to(data_type).npu()


def _assert_outputs_binary_equal(outputs):
    reference_name, reference = next(iter(outputs.items()))
    output_bits = {}
    for name, output in outputs.items():
        assert output.shape == reference.shape, (
            f"{name} shape {output.shape} differs from "
            f"{reference_name} shape {reference.shape}"
        )
        assert output.dtype == reference.dtype, (
            f"{name} dtype {output.dtype} differs from "
            f"{reference_name} dtype {reference.dtype}"
        )
        output_bits[name] = output.detach().cpu().contiguous().view(torch.int16)

    mismatches = []
    for (left_name, left), (right_name, right) in combinations(
        output_bits.items(), 2
    ):
        mismatch_count = torch.count_nonzero(left != right).item()
        if mismatch_count:
            mismatches.append(
                f"{left_name} vs {right_name}: "
                f"{mismatch_count}/{left.numel()} elements differ"
            )
    assert not mismatches, "outputs are not binary identical; " + "; ".join(mismatches)


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, "
    "head_size, is_causal, seed",
    test_cases,
)
def test_train_infer_consistency(
    data_type,
    batch_size,
    num_heads,
    kv_heads,
    q_seqlen,
    kv_seqlen,
    head_size,
    is_causal,
    seed,
):
    generator = torch.Generator().manual_seed(seed)
    query = _random_npu_tensor(
        (batch_size, q_seqlen, num_heads, head_size), data_type, generator
    )
    key = _random_npu_tensor(
        (batch_size, kv_seqlen, kv_heads, head_size), data_type, generator
    )
    value = _random_npu_tensor(
        (batch_size, kv_seqlen, kv_heads, head_size), data_type, generator
    )

    cu_seqlens_q = torch.arange(
        0,
        (batch_size + 1) * q_seqlen,
        q_seqlen,
        dtype=torch.int32,
    ).npu()
    cu_seqlens_k = torch.arange(
        0,
        (batch_size + 1) * kv_seqlen,
        kv_seqlen,
        dtype=torch.int32,
    ).npu()
    cache_seqlens = torch.full(
        (batch_size,), kv_seqlen, dtype=torch.int32
    ).npu()
    softmax_scale = head_size ** (-0.5)

    output = flash_attn_func(
        query,
        key,
        value,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=is_causal,
    )
    varlen_output = flash_attn_varlen_func(
        query.flatten(0, 1),
        key.flatten(0, 1),
        value.flatten(0, 1),
        cu_seqlens_q,
        cu_seqlens_k,
        q_seqlen,
        kv_seqlen,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=is_causal,
    ).view_as(output)
    kvcache_output = flash_attn_with_kvcache(
        query,
        key,
        value,
        cache_seqlens=cache_seqlens,
        softmax_scale=softmax_scale,
        causal=is_causal,
    )
    torch.npu.synchronize()

    _assert_outputs_binary_equal(
        {
            "flash_attn_func": output,
            "flash_attn_varlen_func": varlen_output,
            "flash_attn_with_kvcache": kvcache_output,
        }
    )

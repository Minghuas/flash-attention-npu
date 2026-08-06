# Copyright (c) 2026, Minghua Shen.

import ctypes

import numpy as np
import pytest
import torch
import torch_npu

from flash_attn_npu import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_scheduler_metadata,
)
from tests.test_flash_attn_npu_v2 import ref_flash_attention


RTOL = 1e-2
ATOL = 1e-2
WINDOW_SIZE = (-1, -1)

SMALL_RANGE = (-1.0, 1.0)
WIDE_RANGE = (-5.0, 5.0)


def _rand_npu(shape, data_type, value_range):
    low, high = value_range
    return (low + (high - low) * torch.rand(shape)).to(data_type).npu()


def _prefix_sums(lengths):
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return offsets


def _int32_npu(values):
    return torch.tensor(values, dtype=torch.int32).npu()


def _causal_mask(q_seqlen, kv_seqlen, is_causal):
    if not is_causal:
        return None
    return torch.triu(
        torch.ones(q_seqlen, kv_seqlen),
        diagonal=kv_seqlen - q_seqlen + 1,
    ).bool()


def _band_mask(q_seqlen, kv_seqlen, window_size_left, window_size_right):
    pre_token = kv_seqlen - q_seqlen - window_size_left
    next_token = kv_seqlen - q_seqlen + window_size_right
    rows = torch.arange(q_seqlen).unsqueeze(1)
    cols = torch.arange(kv_seqlen).unsqueeze(0)
    diag = cols - rows
    return (diag < pre_token) | (diag > next_token)


def _attn_mask(q_seqlen, kv_seqlen, is_causal, window_size):
    # Mirrors the window normalization in the C++ host/metadata paths.
    window_left, window_right = window_size
    if kv_seqlen > 0 and window_left >= kv_seqlen - 1:
        window_left = -1
    if q_seqlen > 0 and window_right >= q_seqlen - 1:
        window_right = -1
    if is_causal:
        window_right = 0
    causal_golden = window_left < 0 and window_right == 0
    local_golden = (window_left >= 0 or window_right >= 0) and not causal_golden
    if causal_golden:
        return _causal_mask(q_seqlen, kv_seqlen, True)
    if local_golden:
        return _band_mask(q_seqlen, kv_seqlen, window_left, window_right)
    return None


def _metadata(
    *,
    batch_size,
    q_seqlen,
    kv_seqlen,
    num_heads,
    kv_heads,
    head_size,
    cache_seqlens,
    data_type,
    cu_seqlens_q=None,
    page_size=None,
    is_causal=False,
    window_size=WINDOW_SIZE,
    softcap=0.0,
    softmax_scale=None,
):
    return get_scheduler_metadata(
        batch_size=batch_size,
        max_seqlen_q=q_seqlen,
        max_seqlen_k=kv_seqlen,
        num_heads_q=num_heads,
        num_heads_kv=kv_heads,
        headdim=head_size,
        cache_seqlens=cache_seqlens,
        qkv_dtype=data_type,
        cu_seqlens_q=cu_seqlens_q,
        page_size=page_size,
        causal=is_causal,
        window_size=window_size,
        softcap=softcap,
        softmax_scale=softmax_scale,
    )


def _make_paged_cache(batch_size, kv_seqlen, kv_heads, head_size, block_size, data_type):
    max_blocks_per_seq = (kv_seqlen + block_size - 1) // block_size
    num_blocks = batch_size * max_blocks_per_seq
    key_cache = _rand_npu((num_blocks, block_size, kv_heads, head_size), data_type, SMALL_RANGE)
    value_cache = _rand_npu((num_blocks, block_size, kv_heads, head_size), data_type, SMALL_RANGE)
    block_table = torch.arange(num_blocks, dtype=torch.int32).reshape(batch_size, max_blocks_per_seq).npu()
    return key_cache, value_cache, block_table


def _paged_kv_for_batch(key_cache_cpu, value_cache_cpu, block_table_cpu, batch_idx, kv_seqlen, block_size):
    key_blocks = []
    value_blocks = []
    table_row = block_table_cpu[batch_idx]
    for pos in range(kv_seqlen):
        block_number = int(table_row[pos // block_size])
        block_offset = pos % block_size
        key_blocks.append(key_cache_cpu[block_number, block_offset])
        value_blocks.append(value_cache_cpu[block_number, block_offset])
    return torch.stack(key_blocks, dim=0), torch.stack(value_blocks, dim=0)


def _ref_out_lse(query_cpu, key_cpu, value_cpu, scale, data_type, is_causal,
                 window_size=WINDOW_SIZE, softcap=0.0):
    output, lse = ref_flash_attention(
        query_cpu,
        key_cpu,
        value_cpu,
        scale,
        _attn_mask(query_cpu.shape[0], key_cpu.shape[0], is_causal, window_size),
        data_type,
        softcap=softcap,
    )
    return output, lse.reshape(query_cpu.shape[1], query_cpu.shape[0])


def _assert_bsnd_matches_ref(
    output_npu,
    softmax_lse_npu,
    query,
    kv_for_batch,
    *,
    batch_size,
    q_seqlen,
    num_heads,
    head_size,
    scale,
    data_type,
    is_causal,
    window_size=WINDOW_SIZE,
    softcap=0.0,
):
    query_cpu = query.detach().cpu()
    golden_out = torch.empty((batch_size, q_seqlen, num_heads, head_size), dtype=data_type)
    golden_lse = torch.empty((batch_size, num_heads, q_seqlen), dtype=torch.float32)

    for batch_idx in range(batch_size):
        key_cpu, value_cpu = kv_for_batch(batch_idx)
        out, lse = _ref_out_lse(query_cpu[batch_idx], key_cpu, value_cpu, scale, data_type,
                                is_causal, window_size, softcap)
        golden_out[batch_idx] = out.reshape(q_seqlen, num_heads, head_size)
        golden_lse[batch_idx] = lse

    torch.testing.assert_close(output_npu.cpu(), golden_out, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(softmax_lse_npu.cpu(), golden_lse, rtol=RTOL, atol=ATOL)


def _assert_tnd_matches_ref(
    output_npu,
    softmax_lse_npu,
    query,
    kv_for_batch,
    *,
    q_offsets,
    batch_size,
    num_heads,
    head_size,
    scale,
    data_type,
    is_causal,
    window_size=WINDOW_SIZE,
    softcap=0.0,
):
    query_cpu = query.detach().cpu()
    golden_out = torch.empty((q_offsets[-1], num_heads, head_size), dtype=data_type)
    golden_lse = None
    if softmax_lse_npu is not None:
        golden_lse = torch.empty((num_heads, q_offsets[-1]), dtype=torch.float32)

    for batch_idx in range(batch_size):
        q_start, q_end = q_offsets[batch_idx], q_offsets[batch_idx + 1]
        key_cpu, value_cpu = kv_for_batch(batch_idx)
        out, lse = _ref_out_lse(query_cpu[q_start:q_end], key_cpu, value_cpu, scale, data_type,
                                is_causal, window_size, softcap)
        golden_out[q_start:q_end] = out.reshape(q_end - q_start, num_heads, head_size)
        if golden_lse is not None:
            golden_lse[:, q_start:q_end] = lse

    torch.testing.assert_close(output_npu.cpu(), golden_out, rtol=RTOL, atol=ATOL)
    if softmax_lse_npu is not None:
        torch.testing.assert_close(softmax_lse_npu.cpu(), golden_lse, rtol=RTOL, atol=ATOL)


@pytest.fixture
def metadata_spy(monkeypatch):
    """Spy on get_scheduler_metadata to prove the training interfaces route
    through the AICPU scheduler-metadata path internally (official flash-attn
    only exposes scheduler_metadata on flash_attn_with_kvcache)."""
    from flash_attn_npu import flash_attn_npu_interface as interface
    calls = []
    original = interface.get_scheduler_metadata

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(interface, "get_scheduler_metadata", _spy)
    return calls


FLASH_ATTN_FUNC_CASES = [
    # data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, False),
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, True),
    (torch.float16, 7, 1, 1, 512, 512, 128, False),
]


FLASH_ATTN_VARLEN_CASES = [
    # data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal
    (torch.bfloat16, 1, 1, 1, 512, 1024, 128, True),
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, False),
    (torch.float16, 7, 5, 1, 512, 512, 128, True),
]


FLASH_ATTN_FUNC_SWA_SOFTCAP_CASES = [
    # data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, window_size, softcap
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, False, (256, 256), 0.0),
    (torch.bfloat16, 2, 4, 2, 1024, 1024, 128, True, (512, -1), 0.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, True, (512, 256), 30.0),
    (torch.float16, 2, 2, 2, 512, 512, 128, False, (64, 128), 30.0),
]


FLASH_ATTN_VARLEN_SWA_SOFTCAP_CASES = [
    # data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, window_size, softcap
    (torch.bfloat16, 3, 4, 2, 512, 768, 128, False, (200, 200), 0.0),
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, True, (511, -1), 0.0),
    (torch.bfloat16, 1, 1, 1, 512, 1024, 128, True, (512, 0), 30.0),
    (torch.float16, 1, 2, 2, 512, 512, 128, False, (64, 128), 30.0),
]


KV_CACHE_BSND_CASES = [
    # data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, block_size, is_causal, window_size, softcap
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 128, False, (-1, -1), 0.0),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, 128, True, (-1, -1), 0.0),
    (torch.bfloat16, 2, 4, 4, 512, 1024, 128, 128, False, (256, 256), 0.0),
    (torch.bfloat16, 2, 4, 4, 512, 1024, 128, 128, True, (300, -1), 0.0),
    (torch.bfloat16, 2, 4, 2, 128, 1024, 128, 128, True, (-1, -1), 30.0),
    (torch.float16, 1, 2, 1, 256, 512, 128, 128, False, (128, 128), 50.0),
]


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal",
    FLASH_ATTN_FUNC_CASES,
)
def test_flash_attn_func_metadata_bsnd(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal,
    metadata_spy,
):
    query = _rand_npu((batch_size, q_seqlen, num_heads, head_size), data_type, WIDE_RANGE)
    key = _rand_npu((batch_size, kv_seqlen, kv_heads, head_size), data_type, WIDE_RANGE)
    value = _rand_npu((batch_size, kv_seqlen, kv_heads, head_size), data_type, WIDE_RANGE)
    scale = 1.0 / (head_size ** 0.5)

    output_npu, softmax_lse_npu, _ = flash_attn_func(
        query,
        key,
        value,
        softmax_scale=scale,
        causal=is_causal,
        window_size=WINDOW_SIZE,
        return_attn_probs=True,
    )
    assert len(metadata_spy) == 1

    key_cpu = key.detach().cpu()
    value_cpu = value.detach().cpu()
    _assert_bsnd_matches_ref(
        output_npu,
        softmax_lse_npu,
        query,
        lambda batch_idx: (key_cpu[batch_idx], value_cpu[batch_idx]),
        batch_size=batch_size,
        q_seqlen=q_seqlen,
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        data_type=data_type,
        is_causal=is_causal,
    )


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal",
    FLASH_ATTN_VARLEN_CASES,
)
def test_flash_attn_varlen_func_metadata_tnd(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal,
    metadata_spy,
):
    q_lengths = [q_seqlen] * batch_size
    kv_lengths = [kv_seqlen] * batch_size
    q_offsets = _prefix_sums(q_lengths)
    kv_offsets = _prefix_sums(kv_lengths)
    cu_seqlens_q = _int32_npu(q_offsets)
    cu_seqlens_k = _int32_npu(kv_offsets)

    query = _rand_npu((q_offsets[-1], num_heads, head_size), data_type, WIDE_RANGE)
    key = _rand_npu((kv_offsets[-1], kv_heads, head_size), data_type, WIDE_RANGE)
    value = _rand_npu((kv_offsets[-1], kv_heads, head_size), data_type, WIDE_RANGE)
    scale = 1.0 / (head_size ** 0.5)

    output_npu, softmax_lse_npu, _ = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        q_seqlen,
        kv_seqlen,
        softmax_scale=scale,
        causal=is_causal,
        window_size=WINDOW_SIZE,
        return_attn_probs=True,
    )
    assert len(metadata_spy) == 1

    key_cpu = key.detach().cpu()
    value_cpu = value.detach().cpu()
    _assert_tnd_matches_ref(
        output_npu,
        softmax_lse_npu,
        query,
        lambda batch_idx: (
            key_cpu[kv_offsets[batch_idx]:kv_offsets[batch_idx + 1]],
            value_cpu[kv_offsets[batch_idx]:kv_offsets[batch_idx + 1]],
        ),
        q_offsets=q_offsets,
        batch_size=batch_size,
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        data_type=data_type,
        is_causal=is_causal,
    )


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, window_size, softcap",
    FLASH_ATTN_FUNC_SWA_SOFTCAP_CASES,
)
def test_flash_attn_func_metadata_swa_softcap(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size,
    is_causal, window_size, softcap, metadata_spy,
):
    query = _rand_npu((batch_size, q_seqlen, num_heads, head_size), data_type, WIDE_RANGE)
    key = _rand_npu((batch_size, kv_seqlen, kv_heads, head_size), data_type, WIDE_RANGE)
    value = _rand_npu((batch_size, kv_seqlen, kv_heads, head_size), data_type, WIDE_RANGE)
    scale = 1.0 / (head_size ** 0.5)

    output_npu, softmax_lse_npu, _ = flash_attn_func(
        query,
        key,
        value,
        softmax_scale=scale,
        causal=is_causal,
        window_size=window_size,
        softcap=softcap,
        return_attn_probs=True,
    )
    assert len(metadata_spy) == 1

    key_cpu = key.detach().cpu()
    value_cpu = value.detach().cpu()
    _assert_bsnd_matches_ref(
        output_npu,
        softmax_lse_npu,
        query,
        lambda batch_idx: (key_cpu[batch_idx], value_cpu[batch_idx]),
        batch_size=batch_size,
        q_seqlen=q_seqlen,
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        data_type=data_type,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
    )


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, window_size, softcap",
    FLASH_ATTN_VARLEN_SWA_SOFTCAP_CASES,
)
def test_flash_attn_varlen_func_metadata_swa_softcap(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size,
    is_causal, window_size, softcap, metadata_spy,
):
    q_lengths = [q_seqlen] * batch_size
    kv_lengths = [kv_seqlen] * batch_size
    q_offsets = _prefix_sums(q_lengths)
    kv_offsets = _prefix_sums(kv_lengths)
    cu_seqlens_q = _int32_npu(q_offsets)
    cu_seqlens_k = _int32_npu(kv_offsets)

    query = _rand_npu((q_offsets[-1], num_heads, head_size), data_type, WIDE_RANGE)
    key = _rand_npu((kv_offsets[-1], kv_heads, head_size), data_type, WIDE_RANGE)
    value = _rand_npu((kv_offsets[-1], kv_heads, head_size), data_type, WIDE_RANGE)
    scale = 1.0 / (head_size ** 0.5)

    output_npu, softmax_lse_npu, _ = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        q_seqlen,
        kv_seqlen,
        softmax_scale=scale,
        causal=is_causal,
        window_size=window_size,
        softcap=softcap,
        return_attn_probs=True,
    )
    assert len(metadata_spy) == 1

    key_cpu = key.detach().cpu()
    value_cpu = value.detach().cpu()
    _assert_tnd_matches_ref(
        output_npu,
        softmax_lse_npu,
        query,
        lambda batch_idx: (
            key_cpu[kv_offsets[batch_idx]:kv_offsets[batch_idx + 1]],
            value_cpu[kv_offsets[batch_idx]:kv_offsets[batch_idx + 1]],
        ),
        q_offsets=q_offsets,
        batch_size=batch_size,
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        data_type=data_type,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
    )


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, block_size, is_causal, window_size, softcap",
    KV_CACHE_BSND_CASES,
)
def test_flash_attn_kvcache_metadata_bsnd(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size,
    block_size, is_causal, window_size, softcap
):
    query = _rand_npu((batch_size, q_seqlen, num_heads, head_size), data_type, SMALL_RANGE)
    key_cache, value_cache, block_table = _make_paged_cache(
        batch_size, kv_seqlen, kv_heads, head_size, block_size, data_type
    )
    cache_seqlens = _int32_npu([kv_seqlen] * batch_size)
    scale = 1.0 / (head_size ** 0.5)

    scheduler_metadata = _metadata(
        batch_size=batch_size,
        q_seqlen=q_seqlen,
        kv_seqlen=kv_seqlen,
        num_heads=num_heads,
        kv_heads=kv_heads,
        head_size=head_size,
        cache_seqlens=cache_seqlens,
        data_type=data_type,
        page_size=block_size,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
    )
    output_npu, softmax_lse_npu = flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=None,
        causal=is_causal,
        window_size=window_size,
        softcap=softcap,
        num_splits=0,
        scheduler_metadata=scheduler_metadata,
        return_softmax_lse=True,
    )

    key_cache_cpu = key_cache.detach().cpu()
    value_cache_cpu = value_cache.detach().cpu()
    block_table_cpu = block_table.cpu()
    _assert_bsnd_matches_ref(
        output_npu,
        softmax_lse_npu,
        query,
        lambda batch_idx: _paged_kv_for_batch(
            key_cache_cpu, value_cache_cpu, block_table_cpu, batch_idx, kv_seqlen, block_size
        ),
        batch_size=batch_size,
        q_seqlen=q_seqlen,
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        data_type=data_type,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
    )


class _CoreNode(ctypes.Structure):
    _fields_ = [
        ("startBIdx", ctypes.c_int), ("startN1Idx", ctypes.c_int),
        ("startS1Idx", ctypes.c_int), ("startS2Idx", ctypes.c_int),
        ("endBIdx", ctypes.c_int), ("endN1Idx", ctypes.c_int),
        ("endS1Idx", ctypes.c_int), ("endS2Idx", ctypes.c_int),
        ("firstSplitKVTaskLseOffset", ctypes.c_int64),
        ("firstSplitKVTaskOOffset", ctypes.c_int64),
    ]


class _SplitNode(ctypes.Structure):
    _fields_ = [
        ("batchIdx", ctypes.c_int), ("headStartIdx", ctypes.c_int),
        ("headEndIdx", ctypes.c_int), ("qStartIdx", ctypes.c_int),
        ("qEndIdx", ctypes.c_int), ("splitNum", ctypes.c_int),
        ("lseTaskOffset", ctypes.c_int64), ("oTaskOffset", ctypes.c_int64),
    ]


class _FAInferTilingData(ctypes.Structure):
    """Mirror of csrc/ascend910/flash_attn_npu/tilingdata.h for flag checks."""
    _fields_ = [
        ("numHeads", ctypes.c_uint32), ("embeddingSize", ctypes.c_uint32),
        ("embeddingSizeV", ctypes.c_uint32), ("numBlocks", ctypes.c_uint32),
        ("blockSize", ctypes.c_uint32), ("maxQSeqlen", ctypes.c_uint32),
        ("maxKvSeqlen", ctypes.c_uint32), ("kvHeads", ctypes.c_uint32),
        ("batch", ctypes.c_uint32), ("maxNumBlocksPerBatch", ctypes.c_uint32),
        ("firstBatchTaskNum", ctypes.c_uint32), ("totalTaskNum", ctypes.c_uint32),
        ("maskType", ctypes.c_uint32),
        ("mm1OutSize", ctypes.c_uint64), ("smOnlineOutSize", ctypes.c_uint64),
        ("mm2OutSize", ctypes.c_uint64), ("UpdateSize", ctypes.c_uint64),
        ("workSpaceSize", ctypes.c_uint64),
        ("scaleValue", ctypes.c_float), ("softcapValue", ctypes.c_float),
        ("padding1", ctypes.c_uint64), ("padding2", ctypes.c_uint64),
        ("padding3", ctypes.c_uint32),
        ("windowSizeLeft", ctypes.c_int64), ("windowSizeRight", ctypes.c_int64),
        ("splitLseTotalSize", ctypes.c_uint64), ("splitOTotalSize", ctypes.c_uint64),
        ("totalSplitNodeNum", ctypes.c_uint32), ("needCoreNum", ctypes.c_uint32),
        ("flashDecodeFlag", ctypes.c_uint32),
        ("coreInfo", _CoreNode * 25),
        ("splitInfo", _SplitNode * 25),
    ]


def _tiling_from_metadata(scheduler_metadata, has_mask):
    raw = scheduler_metadata.cpu().numpy()
    mask_bytes = 2048 * 2048 if has_mask else 0
    blob = raw[mask_bytes:mask_bytes + ctypes.sizeof(_FAInferTilingData)]
    tiling = _FAInferTilingData.from_buffer_copy(blob)
    return tiling


@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_attn_kvcache_metadata_flash_decode(is_causal):
    """FD (BSND paged, tiny Q, long KV) must be decided and scheduled on the AICPU."""
    batch_size, num_heads, kv_heads = 2, 4, 1
    q_seqlen, kv_seqlen, head_size, block_size = 1, 4096, 128, 128
    data_type = torch.bfloat16

    query = _rand_npu((batch_size, q_seqlen, num_heads, head_size), data_type, SMALL_RANGE)
    key_cache, value_cache, block_table = _make_paged_cache(
        batch_size, kv_seqlen, kv_heads, head_size, block_size, data_type
    )
    cache_seqlens = _int32_npu([kv_seqlen] * batch_size)
    scale = 1.0 / (head_size ** 0.5)

    scheduler_metadata = _metadata(
        batch_size=batch_size,
        q_seqlen=q_seqlen,
        kv_seqlen=kv_seqlen,
        num_heads=num_heads,
        kv_heads=kv_heads,
        head_size=head_size,
        cache_seqlens=cache_seqlens,
        data_type=data_type,
        page_size=block_size,
        is_causal=is_causal,
    )
    tiling = _tiling_from_metadata(scheduler_metadata, has_mask=is_causal)
    assert tiling.flashDecodeFlag == 1
    assert tiling.needCoreNum > 0
    assert tiling.splitLseTotalSize > 0
    assert tiling.workSpaceSize > 0

    output_npu, softmax_lse_npu = flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=scale,
        causal=is_causal,
        window_size=WINDOW_SIZE,
        num_splits=0,
        scheduler_metadata=scheduler_metadata,
        return_softmax_lse=True,
    )

    key_cache_cpu = key_cache.detach().cpu()
    value_cache_cpu = value_cache.detach().cpu()
    block_table_cpu = block_table.cpu()
    _assert_bsnd_matches_ref(
        output_npu,
        softmax_lse_npu,
        query,
        lambda batch_idx: _paged_kv_for_batch(
            key_cache_cpu, value_cache_cpu, block_table_cpu, batch_idx, kv_seqlen, block_size
        ),
        batch_size=batch_size,
        q_seqlen=q_seqlen,
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        data_type=data_type,
        is_causal=is_causal,
    )


@pytest.mark.parametrize("is_causal, window_size", [(True, (-1, -1)), (False, (128, 128))])
def test_flash_attn_kvcache_metadata_mask_mismatch_rejected(is_causal, window_size):
    """A mask-less metadata buffer must be rejected by a fwd call needing a mask."""
    data_type = torch.bfloat16
    batch_size, num_heads, kv_heads = 1, 2, 2
    q_seqlen, kv_seqlen, head_size, block_size = 512, 512, 128, 128

    query = _rand_npu((batch_size, q_seqlen, num_heads, head_size), data_type, SMALL_RANGE)
    key_cache, value_cache, block_table = _make_paged_cache(
        batch_size, kv_seqlen, kv_heads, head_size, block_size, data_type
    )
    cache_seqlens = _int32_npu([kv_seqlen] * batch_size)

    scheduler_metadata = _metadata(
        batch_size=batch_size,
        q_seqlen=q_seqlen,
        kv_seqlen=kv_seqlen,
        num_heads=num_heads,
        kv_heads=kv_heads,
        head_size=head_size,
        cache_seqlens=cache_seqlens,
        data_type=data_type,
        page_size=block_size,
        is_causal=False,
    )
    with pytest.raises(RuntimeError, match="scheduler_metadata buffer is too small"):
        flash_attn_with_kvcache(
            query,
            key_cache,
            value_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=is_causal,
            window_size=window_size,
            scheduler_metadata=scheduler_metadata,
        )

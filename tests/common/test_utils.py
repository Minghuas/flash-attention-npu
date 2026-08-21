# Copyright (c) 2026, Minghua Shen.

"""Attention 测试共用的数据、分页 KV 与 mask 构造工具。"""

import torch

CASE_SEED = 42


def make_random_tensor(
    shape,
    data_type,
    *,
    low=-5.0,
    high=5.0,
    generator=None,
    device=None,
    requires_grad=False,
):
    """生成测试用随机 tensor，统一 dtype、范围、设备和 grad 设置。

    随机数先在 CPU 上生成，再移动到目标设备，以保持现有测试的随机数
    顺序和可复现性。
    """
    tensor = low + (high - low) * torch.rand(shape, generator=generator)
    tensor = tensor.to(data_type)
    if device is not None:
        if device == "npu":
            tensor = tensor.npu()
        else:
            tensor = tensor.to(device)
    return tensor.requires_grad_(requires_grad)


def make_attention_inputs(
    query_shape,
    key_shape,
    value_shape,
    dout_shape,
    data_type,
    *,
    generator=None,
    device=None,
    requires_grad=(True, True, True),
):
    """统一生成 Attention 的 Q、K、V 和反向输入 dout。

    Q/K/V 默认使用 ``[-5, 5]``，dout 默认使用 ``[-1, 1]``。三个输入的
    shape 由调用方提供，因此 BSND、TND packed 和 paged KV 都可以复用；
    paged KV 的 block table 仍由对应测试单独构造。
    """
    query = make_random_tensor(
        query_shape,
        data_type,
        generator=generator,
        device=device,
        requires_grad=requires_grad[0],
    )
    key = make_random_tensor(
        key_shape,
        data_type,
        generator=generator,
        device=device,
        requires_grad=requires_grad[1],
    )
    value = make_random_tensor(
        value_shape,
        data_type,
        generator=generator,
        device=device,
        requires_grad=requires_grad[2],
    )
    dout = make_random_tensor(
        dout_shape,
        data_type,
        low=-1.0,
        high=1.0,
        generator=generator,
        device=device,
    )
    return query, key, value, dout


def make_cu_seqlens(seqlens):
    """把每个 batch 的序列长度转换为 int32 累积偏移。

    返回长度为 ``len(seqlens) + 1`` 的 CPU tensor，首元素为 0，末元素为
    所有序列长度之和，可直接作为变长 Attention 的 ``cu_seqlens``。
    """
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32)
    for i, seqlen in enumerate(seqlens, start=1):
        cu[i] = cu[i - 1] + int(seqlen)
    return cu


def make_varlen_seqlens(batch_size, max_seqlen_q, max_seqlen_k, seed=CASE_SEED):
    """生成可复现的变长 Q/KV 序列长度。

    按 Tri Dao 测试方式生成接近最大长度的随机有效长度，范围为
    ``[max_seqlen - 20, max_seqlen]``。返回两个 Python list，分别保存 Q 与 KV 长度。
    独立 generator 保证长度不受测试中其他随机 tensor 的生成顺序影响。
    """
    generator = torch.Generator().manual_seed(seed)
    min_q = max(1, max_seqlen_q - 20)
    min_k = max(1, max_seqlen_k - 20)
    seqlens_q = torch.randint(min_q, max_seqlen_q + 1, (batch_size,), generator=generator).tolist()
    seqlens_k = []
    for q_seqlen in seqlens_q:
        # Keep the Tri Dao near-max distribution while preserving the
        # q_seqlen <= kv_seqlen contract used by the NPU varlen causal path.
        k_low = max(min_k, q_seqlen)
        if k_low > max_seqlen_k:
            k_low = max_seqlen_k
        seqlens_k.append(int(torch.randint(k_low, max_seqlen_k + 1, (1,), generator=generator).item()))
    return seqlens_q, seqlens_k


def make_packed_random_tensor(
    seqlens,
    max_seqlen,
    num_heads,
    head_size,
    data_type,
    *,
    generator=None,
    device=None,
    requires_grad=False,
):
    """按 Tri Dao 的 padded -> unpad 流程生成 TND 随机输入。"""
    padded = make_random_tensor(
        (len(seqlens), max_seqlen, num_heads, head_size),
        data_type,
        generator=generator,
    )
    valid = torch.arange(max_seqlen) < torch.tensor(seqlens)[:, None]
    packed = padded[valid]
    if device is not None:
        packed = packed.npu() if device == "npu" else packed.to(device)
    return packed.detach().requires_grad_(requires_grad)


def pad_packed_tensor(packed, seqlens, max_seqlen):
    """按 ``seqlens`` 将 TND packed tensor 恢复为 padded 四维 tensor。"""
    valid = torch.arange(max_seqlen, device=packed.device) < torch.as_tensor(
        seqlens, device=packed.device
    )[:, None]
    padded = torch.zeros(
        (len(seqlens), max_seqlen, *packed.shape[1:]),
        dtype=packed.dtype,
        device=packed.device,
    )
    padded[valid] = packed
    return padded


def make_block_table(batch_size, kv_seqlen, block_size):
    """生成每个 batch 使用连续物理块的分页 KV 索引表。

    返回形状为 ``[batch_size, ceil(kv_seqlen / block_size)]`` 的 int32 CPU
    tensor。每行对应一个 batch，物理块编号连续且不同 batch 之间不重叠。
    """
    blocks_per_sequence = (kv_seqlen + block_size - 1) // block_size
    return torch.arange(
        batch_size * blocks_per_sequence,
        dtype=torch.int32,
    ).reshape(batch_size, blocks_per_sequence)


def gather_paged_kv(key_cache, value_cache, block_table_row, kv_seqlen, block_size):
    """按单个 batch 的分页索引还原有效长度内的 K/V。

    ``key_cache`` 和 ``value_cache`` 的前两维为物理块与块内偏移；
    ``block_table_row`` 给出逻辑块对应的物理块编号。返回的 K/V 第一维均为
    ``kv_seqlen``，向量化索引会保留输入 cache 所需的 autograd 链路。
    """
    block_table_row = block_table_row.to(device=key_cache.device, dtype=torch.long)
    positions = torch.arange(kv_seqlen, device=key_cache.device)
    block_indices = block_table_row[positions // block_size]
    block_offsets = positions % block_size
    return (
        key_cache[block_indices, block_offsets],
        value_cache[block_indices, block_offsets],
    )


def gather_paged_kv_batch(key_cache, value_cache, block_tables, kv_seqlen, block_size):
    """Batch 版本的分页 KV gather，返回 ``[B, kv_seqlen, H, D]``。"""
    block_tables = block_tables.to(device=key_cache.device, dtype=torch.long)
    positions = torch.arange(kv_seqlen, device=key_cache.device)
    block_indices = block_tables[:, positions // block_size]
    block_offsets = positions % block_size
    return (
        key_cache[block_indices, block_offsets[None, :]],
        value_cache[block_indices, block_offsets[None, :]],
    )


def make_local_attention_mask(q_seqlen, kv_seqlen, window_size_left, window_size_right):
    """构造与序列右端对齐的局部 Attention bool mask。

    返回形状为 ``[q_seqlen, kv_seqlen]`` 的 CPU tensor；``True`` 表示该
    Q/KV 位置位于左右窗口之外，需要在 reference 计算中屏蔽。
    """
    left_boundary = kv_seqlen - q_seqlen - window_size_left
    right_boundary = kv_seqlen - q_seqlen + window_size_right
    row = torch.arange(q_seqlen)[:, None]
    col = torch.arange(kv_seqlen)[None, :]
    return ((-row + col) < left_boundary) | ((-row + col) > right_boundary)


def make_golden_attention_mask(q_seqlen, kv_seqlen, is_causal, window_size_left, window_size_right):
    """规范化窗口参数并构造 reference Attention mask。

    ``-1`` 表示对应方向无限；超出 KV 长度的窗口也按无限处理。causal 模式
    会把右窗口规范为 0。返回 ``(mask, is_causal_golden, is_local_golden)``；
    无需 mask 时 ``mask`` 为 ``None``，否则其中 ``True`` 表示被屏蔽位置。
    """
    wl = window_size_left
    wr = window_size_right
    if kv_seqlen > 0 and wl >= kv_seqlen:
        wl = -1
    if kv_seqlen > 0 and wr >= kv_seqlen:
        wr = -1
    if is_causal:
        wr = 0
    is_causal_golden = wl < 0 and wr == 0
    is_local_golden = (wl >= 0 or wr > 0) and not is_causal_golden
    if is_local_golden:
        if wl < 0:
            wl = kv_seqlen
        if wr < 0:
            wr = kv_seqlen
    if is_causal_golden:
        mask = torch.triu(torch.ones(q_seqlen, kv_seqlen), diagonal=kv_seqlen - q_seqlen + 1).to(torch.bool)
    elif is_local_golden:
        mask = make_local_attention_mask(q_seqlen, kv_seqlen, wl, wr)
    else:
        mask = None
    return mask, is_causal_golden, is_local_golden


def make_padded_varlen_mask(
    q_seqlens,
    kv_seqlens,
    max_q_seqlen,
    max_kv_seqlen,
    is_causal,
    window_size_left,
    window_size_right,
):
    """构造 padded TND golden 使用的 batch 级有效位置与 Attention mask。"""
    q_seqlens = torch.as_tensor(q_seqlens)
    kv_seqlens = torch.as_tensor(kv_seqlens)
    q_valid = torch.arange(max_q_seqlen) < q_seqlens[:, None]
    k_valid = torch.arange(max_kv_seqlen) < kv_seqlens[:, None]
    mask = (~q_valid[:, :, None]) | (~k_valid[:, None, :])
    row = torch.arange(max_q_seqlen)[None, :, None]
    col = torch.arange(max_kv_seqlen)[None, None, :]
    if max_kv_seqlen > 0 and window_size_left >= max_kv_seqlen:
        window_size_left = -1
    if max_kv_seqlen > 0 and window_size_right >= max_kv_seqlen:
        window_size_right = -1
    if is_causal:
        window_size_right = 0
    is_causal_golden = window_size_left < 0 and window_size_right == 0
    is_local_golden = (window_size_left >= 0 or window_size_right > 0) and not is_causal_golden
    offsets = (kv_seqlens - q_seqlens)[:, None, None]
    if is_causal_golden:
        mask = mask | (col - row >= offsets + 1)
    elif is_local_golden:
        left = max_kv_seqlen if window_size_left < 0 else window_size_left
        right = max_kv_seqlen if window_size_right < 0 else window_size_right
        diff = col - row
        mask = mask | (diff < offsets - left) | (diff > offsets + right)
    return q_valid, k_valid, mask

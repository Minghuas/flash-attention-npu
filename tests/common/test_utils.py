# Copyright (c) 2026, Minghua Shen.

"""Attention 测试共用的数据、分页 KV 与 mask 构造工具。"""

import torch

CASE_SEED = 42


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

    每个长度至少取对应最大长度的四分之一，并保证同一 batch 中
    ``q_seqlen <= kv_seqlen``。返回两个 Python list，分别保存 Q 与 KV 长度。
    独立 generator 保证长度不受测试中其他随机 tensor 的生成顺序影响。
    """
    min_q = max(1, max_seqlen_q // 4)
    min_k = max(1, max_seqlen_k // 4)
    generator = torch.Generator().manual_seed(seed)
    seqlens_q = []
    seqlens_k = []
    for _ in range(batch_size):
        sk = int(torch.randint(min_k, max_seqlen_k + 1, (1,), generator=generator).item())
        q_hi = min(max_seqlen_q, sk)
        q_lo = min(min_q, q_hi)
        seqlens_q.append(int(torch.randint(q_lo, q_hi + 1, (1,), generator=generator).item()))
        seqlens_k.append(sk)
    return seqlens_q, seqlens_k


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

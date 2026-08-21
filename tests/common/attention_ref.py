# Copyright (c) 2026, Minghua Shen.
import torch
"""
  FlashAttention NPU 公共 reference / golden 实现。

  - 正向golden：
    ref_flash_attention，用于 v2/v3/v4 forward
    测试计算 golden out/lse。
"""

"""
小算子 FlashAttention 正向golden相关实现
"""


def softmax1(
    qk_result,
    is_first,
    gm,
    interm_dtype = torch.float16,
    rescale_threshold = 0.0,
    ):
    sim = qk_result.to(interm_dtype)
    lm = torch.max(sim, dim=-1, keepdims=True)[0]
    if is_first:
        hm = lm
        dm = torch.zeros_like(lm)
    else:
        hm = torch.maximum(gm, lm)
        dm = gm - hm
        if rescale_threshold > 0:
            hm = torch.maximum(gm, lm - rescale_threshold)
            dm = gm - hm
    gm = hm
    sim_sub = sim - hm
    sim_sub = torch.exp(sim_sub.to(interm_dtype))
    row_sum = torch.sum(sim_sub, dim=-1, keepdims=True)
    return sim_sub, row_sum, dm, gm


def qk_scores(query, key, scale, reorder_ops):
    """Compute BSND QK scores without flattening the batch dimension."""
    q = query.permute(0, 2, 1, 3)  # B, H, Q, D
    k = key.permute(0, 2, 3, 1)  # B, Hkv, D, K
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if num_heads == num_kv_heads:
        return torch.matmul(q * scale, k) if not reorder_ops else torch.matmul(q, k * scale)
    group_num = num_heads // num_kv_heads
    q = q.reshape(q.shape[0], num_kv_heads, group_num, q.shape[2], q.shape[3])
    k = k.unsqueeze(2)
    scores = torch.matmul(q * scale, k) if not reorder_ops else torch.matmul(q, k * scale)
    return scores.reshape(scores.shape[0], num_heads, scores.shape[-2], scores.shape[-1])


def pv_out(prob, value):
    """Compute BSND probability-times-value without materializing repeated KV."""
    batch, num_heads, q_len, _ = prob.shape
    num_kv_heads = value.shape[2]
    value = value.permute(0, 2, 1, 3)  # B, Hkv, K, D
    if num_heads == num_kv_heads:
        return torch.matmul(prob, value)
    group_num = num_heads // num_kv_heads
    prob = prob.reshape(batch, num_kv_heads, group_num, q_len, prob.shape[-1])
    out = torch.matmul(prob, value.unsqueeze(2))
    return out.reshape(batch, num_heads, q_len, value.shape[-1])


def softmax_with_sink(scores, sink_matrix, value_dtype):
    """Compute attention probabilities and LSE with an extra sink term."""
    sink_matrix = torch.as_tensor(sink_matrix, device=scores.device)
    if sink_matrix.dim() == 3:
        sink_matrix = sink_matrix.unsqueeze(0)
    expected_shape = (scores.shape[0], scores.shape[1], scores.shape[2], 1)
    if tuple(sink_matrix.shape) != expected_shape:
        raise ValueError(
            f"sink_matrix shape {tuple(sink_matrix.shape)} must be {expected_shape}"
        )
    sink_matrix = sink_matrix.to(scores.dtype)
    row_max = torch.maximum(scores.amax(dim=-1, keepdim=True), sink_matrix)
    row_max_high = row_max.to(torch.float64)
    score_exp = torch.exp(scores.to(torch.float64) - row_max_high)
    sink_exp = torch.exp(sink_matrix.to(torch.float64) - row_max_high)
    denominator = score_exp.sum(dim=-1, keepdim=True) + sink_exp
    probability = (score_exp / denominator).to(value_dtype)
    lse = (torch.log(denominator) + row_max_high).squeeze(-1)
    return probability, lse


def ref_flash_attention(
    query,
    key,
    value,
    scale,
    mask,
    data_type,
    softcap=0.0,
    rescale_threshold=0.0,
    upcast=True,
    reorder_ops=False,
    sink_matrix=None,
):
    """BSND reference implementation that computes all batches together."""
    dtype_og = query.dtype
    if upcast:
        query, key, value = query.float(), key.float(), value.float()
    interm_dtype = value.dtype
    scale = torch.tensor(scale, device=query.device, dtype=query.dtype)
    qk_result = qk_scores(query, key, scale, reorder_ops).to(interm_dtype)

    if softcap > 0.0:
        qk_result = torch.tanh(qk_result / softcap) * softcap
    if mask is not None:
        mask = mask.to(device=qk_result.device, dtype=torch.bool)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        qk_result = qk_result.masked_fill(mask[:, None, :, :], -1e4)

    if sink_matrix is not None and rescale_threshold and rescale_threshold > 0.0:
        raise NotImplementedError("sink_matrix is not supported with rescale_threshold")

    if rescale_threshold and rescale_threshold > 0.0:
        context_size = 512
        gm = None
        gl = None
        go = None
        for kv_start in range(0, qk_result.shape[-1], context_size):
            qk_chunk = qk_result[..., kv_start:kv_start + context_size]
            p_chunk, row_sum, dm, gm = softmax1(
                qk_chunk,
                kv_start == 0,
                gm,
                interm_dtype,
                rescale_threshold,
            )
            lo = pv_out(p_chunk.to(value.dtype), value[:, kv_start:kv_start + context_size])
            if kv_start == 0:
                gl = row_sum
                go = lo.to(interm_dtype)
            else:
                dm = torch.exp(dm)
                gl = gl * dm + row_sum
                go = go * dm + lo.to(interm_dtype)
        out = (go / gl).permute(0, 2, 1, 3)
        lse = torch.squeeze(torch.log(gl) + gm, dim=-1).to(torch.float32)
    else:
        if sink_matrix is None:
            lse = torch.logsumexp(qk_result.to(torch.float32), dim=-1)
            prob = torch.softmax(qk_result, dim=-1).to(value.dtype)
        else:
            prob, lse = softmax_with_sink(qk_result, sink_matrix, value.dtype)
        out = pv_out(prob, value).permute(0, 2, 1, 3)
    return out.to(dtype_og), lse

def ref_flash_attention_pair(
    query,
    key,
    value,
    scale,
    mask,
    data_type,
    softcap=0.0,
    rescale_threshold=None,
    sink_matrix=None,
):
    """Return the two BSND golden references used by the comparator."""
    kwargs = {} if rescale_threshold is None else {"rescale_threshold": rescale_threshold}
    out_ref, lse_ref = ref_flash_attention(
        query, key, value, scale, mask, data_type, softcap,
        upcast=True, reorder_ops=False, sink_matrix=sink_matrix, **kwargs,
    )
    out_pt, lse_pt = ref_flash_attention(
        query, key, value, scale, mask, data_type, softcap,
        upcast=False, reorder_ops=True, sink_matrix=sink_matrix, **kwargs,
    )
    return out_ref, lse_ref, out_pt, lse_pt

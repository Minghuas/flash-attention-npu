# Copyright (c) 2026, Minghua Shen.
import torch
"""
  FlashAttention NPU 公共 reference / golden 实现。

  - 正向golden：
    ref_flash_attention / ref_masked_attention，用于 v2/v3/v4 forward 测
    试计算 golden out/lse。
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
    """计算 QK scores。

    MHA 直接使用 torch.matmul；GQA/MQA 使用 grouped einsum，避免真实
    repeat K heads。
    """
    if query.shape[0] == key.shape[0]:
        return torch.matmul(query, key * scale) if reorder_ops else torch.matmul(query * scale, key)
    group_num = query.shape[0] // key.shape[0]
    query_grouped = query.reshape(key.shape[0], group_num, query.shape[1], query.shape[2])
    if reorder_ops:
        scores = torch.einsum("hgtd,hds->hgts", query_grouped, key * scale)
    else:
        scores = torch.einsum("hgtd,hds->hgts", query_grouped * scale, key)
    return scores.reshape(query.shape[0], query.shape[1], key.shape[2])


def pv_out(prob, value):
    """计算 attention * V。

    MHA 直接使用 torch.matmul；GQA/MQA 使用 grouped einsum，避免真实
    repeat V heads。
    """
    if prob.shape[0] == value.shape[0]:
        return torch.matmul(prob, value)
    group_num = prob.shape[0] // value.shape[0]
    prob_grouped = prob.reshape(value.shape[0], group_num, prob.shape[1], prob.shape[2])
    out = torch.einsum("hgts,hsd->hgtd", prob_grouped, value)
    return out.reshape(prob.shape[0], prob.shape[1], value.shape[2])


def ref_flash_attention(
    query,
    key,
    value,
    scale,
    mask,
    data_type,
    softcap=0.0,
    rescale_threshold = 0.0,
    upcast=True,
    reorder_ops=False,
    ):
    dtype_og = query.dtype
    if upcast:
        query, key, value = query.float(), key.float(), value.float()
    interm_dtype = value.dtype
    query = query.permute(1, 0, 2)
    key = key.permute(1, 2, 0)
    value = value.permute(1, 0, 2)
    scale = torch.tensor(scale, device=query.device, dtype=query.dtype)

    if rescale_threshold <= 0:
        qk_result = qk_scores(query, key, scale, reorder_ops).to(interm_dtype)
        if softcap > 0.0:
            qk_result = torch.tanh(qk_result / softcap) * softcap
        if mask is not None:
            mask = mask[: qk_result.shape[-2], : qk_result.shape[-1]]
            qk_result = qk_result.masked_fill(mask.to(torch.bool).to(qk_result.device), -1e4)

        lse = torch.logsumexp(qk_result.to(torch.float32), dim=-1).to(torch.float32)
        p_result = torch.softmax(qk_result, dim=-1).to(value.dtype)
        out = pv_out(p_result, value).permute(1, 0, 2)
        return out.to(dtype_og), lse

    context_len = key.shape[2]
    context_size = 512
    gm = None
    gl = None
    go = None
    if mask is not None:
        mask = mask.to(query.device)
    for kv_start in range(0, context_len, context_size):
        sub_len = min(context_size, context_len - kv_start)
        sub_key = key[:, :, kv_start: kv_start + sub_len]
        sub_mask = None
        if mask is not None:
            sub_mask = mask[:query.shape[1], kv_start : kv_start + sub_len].to(interm_dtype) * (-1e4)
        sub_value = value[:, kv_start: kv_start + sub_len, :]
        qk_result = qk_scores(query, sub_key, scale, reorder_ops).to(interm_dtype)
        if softcap > 0.0:
            qk_result = torch.tanh(qk_result / softcap) * softcap
        if sub_mask is not None:
            qk_result = qk_result + sub_mask
        p_result, row_sum, dm, gm = softmax1(qk_result, kv_start == 0, gm, interm_dtype, rescale_threshold)
        p_result = p_result.to(value.dtype)
        lo = pv_out(p_result, sub_value).to(interm_dtype)
        if kv_start == 0:
            gl = row_sum
            go = lo
        else:
            dm = torch.exp(dm)
            gl = gl * dm + row_sum
            go = go * dm + lo
    go = go / gl
    go = go.permute(1, 0, 2)
    lse = torch.squeeze((torch.log(gl) + gm), dim=-1).to(torch.float32)
    return go.to(dtype_og), lse


def ref_flash_attention_pair(
    query,
    key,
    value,
    scale,
    mask,
    data_type,
    softcap=0.0,
    rescale_threshold=None,
):
    """调用两种 reference，返回 Tri Dao 双基准所需的四个结果。

    ``ref`` 使用 float32 中间计算和原始运算顺序，``pt`` 使用输入 dtype
    中间计算并调整运算顺序。``rescale_threshold`` 为 ``None`` 时使用普通
    reference；需要分块 rescale 的版本可以传入对应阈值。
    """
    kwargs = {} if rescale_threshold is None else {"rescale_threshold": rescale_threshold}
    out_ref, lse_ref = ref_flash_attention(
        query,
        key,
        value,
        scale,
        mask,
        data_type,
        softcap,
        upcast=True,
        reorder_ops=False,
        **kwargs,
    )
    out_pt, lse_pt = ref_flash_attention(
        query,
        key,
        value,
        scale,
        mask,
        data_type,
        softcap,
        upcast=False,
        reorder_ops=True,
        **kwargs,
    )
    return out_ref, lse_ref, out_pt, lse_pt


def masked_attention_sink(sim_high, sink_matrix, value_dtype):
    """计算带 sink token 分母的 attention 概率和 lse。"""
    sink_matrix = torch.as_tensor(sink_matrix, device=sim_high.device)
    assert sink_matrix.shape == sim_high[..., :1].shape, (
        f"sink_matrix 形状 {sink_matrix.shape} 与 row_max 形状 {sim_high[..., :1].shape} 不一致！"
    )
    row_max = torch.maximum(torch.amax(sim_high, dim=-1, keepdim=True), sink_matrix.to(sim_high.dtype))
    sim_sub = torch.exp((sim_high - row_max).to(sim_high.dtype))
    row_sum = torch.sum(sim_sub, dim=-1, keepdim=True)
    sink_exp = torch.exp((sink_matrix.to(row_max.dtype) - row_max).to(sim_high.dtype))
    p_high = (sim_sub / (row_sum + sink_exp)).to(value_dtype)

    row_max_high = row_max.to(torch.float64)
    sim_sub_high = torch.exp(sim_high.to(torch.float64) - row_max_high)
    row_sum_high = torch.sum(sim_sub_high, dim=-1, keepdim=True)
    sink_exp_high = torch.exp(sink_matrix.to(torch.float64) - row_max_high)
    lse_high = torch.squeeze(torch.log(row_sum_high + sink_exp_high) + row_max_high, dim=-1)
    return p_high, lse_high.cpu().numpy()


def ref_masked_attention(
            query,  # (q_seqlen, num_heads, head_size)
            key,    # (k_seqlen, kv_heads, head_size)
            value,
            scale: float,
            mask,    # (q_seqlen, k_seqlen)
            sink_matrix,
            upcast=True,
            reorder_ops=False,
):
    dtype_og = query.dtype
    if upcast:
        query, key, value = query.float(), key.float(), value.float()
    query = query.permute(1, 0, 2)
    key = key.permute(1, 2, 0)
    value = value.permute(1, 0, 2)
    scale = torch.tensor(scale, device=query.device, dtype=query.dtype)
    sim_high = qk_scores(query, key, scale, reorder_ops)
    if mask is not None:
        sim_high = sim_high + (
            mask[:sim_high.shape[-2], :sim_high.shape[-1]]
            ).to(sim_high.device).to(sim_high.dtype) * (-1e4)

    if sink_matrix is not None:
        p_high, lse_high = masked_attention_sink(sim_high, sink_matrix, value.dtype)
    else:
        p_high = torch.softmax(sim_high, dim=-1).to(value.dtype)
        lse_high = torch.logsumexp(sim_high.to(torch.float32), dim=-1).cpu().numpy()

    out_high = pv_out(p_high, value)
    out_high = out_high.permute(1, 0, 2)
    return out_high.to(dtype_og), lse_high

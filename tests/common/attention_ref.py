# Copyright (c) 2026, Minghua Shen.
import torch
"""
  FlashAttention NPU 公共 reference / golden 实现。

  - 正向golden：
    ref_flash_attention / ref_masked_attention，用于 v2/v3/v4 forward 测
    试计算 golden out/lse。
  - 反向golden：
    golden_bsnd_bwd_from_fwd / golden_tnd_bwd_from_fwd，传入 FA forward
    的 out/lse，仅计算 backward golden。
"""

"""
小算子 FlashAttention 正向golden相关实现
"""

def group_matmul(head, kv_head, left, right):
    group_num = head // kv_head
    score = None
    for i in range(kv_head):
        group_score = torch.matmul(left[i * group_num:(i + 1) * group_num, :, :],
                                    right[i:(i + 1), :, :])
        if score is None:
            score = group_score
        else:
            score = torch.cat((score, group_score), 0)
    return score


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

def qkMM1(
    query,
    key
    ):
    result = None
    qk_k = key.shape[1]
    qk_k_split = 128
    qk_k_loop = (qk_k + 127) // 128
    for qk_k_loop_idx in range(qk_k_loop):
        sub_k = 128 if qk_k_loop_idx != (qk_k_loop - 1) else (qk_k - qk_k_loop_idx * 128)
        partial_Query = query[:, :, qk_k_loop_idx * 128: qk_k_loop_idx * 128 + sub_k]
        partial_Key = key[:, qk_k_loop_idx * 128: qk_k_loop_idx * 128 + sub_k, :]
        result_split = group_matmul(partial_Query.shape[0], partial_Key.shape[0], partial_Query, partial_Key)
        if result is None:
            result = result_split
        else:
            result = result + result_split
    return result

def pvMM2(
    p,
    value
    ):
    result = None
    pv_k = value.shape[1]
    pv_k_split = 128
    pv_k_loop = (pv_k + 127) // 128
    for pv_k_loop_idx in range(pv_k_loop):
        sub_k = 128 if pv_k_loop_idx != (pv_k_loop - 1) else (pv_k - pv_k_loop_idx * 128)
        partial_P = p[:, :, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k]
        partial_Value = value[:, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k, :]
        result_split = group_matmul(partial_P.shape[0], partial_Value.shape[0], partial_P, partial_Value)
        if result is None:
            result = result_split
        else:
            result = result + result_split
    return result

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
    scale = torch.tensor(scale).to(query.dtype)
    context_len = key.shape[2]
    context_size = 512
    group_num = query.shape[0] // key.shape[0]
    gl = None
    gl_high = None
    go = None
    go_high = None
    if mask is not None:
        mask = mask.cpu()
    for kv_start in range(0, context_len, context_size):
        sub_len = context_size
        if kv_start + context_size > context_len:
            sub_len = context_len - kv_start
        sub_key = key[:, :, kv_start: kv_start + sub_len]
        sub_mask = None
        if mask is not None:
            sub_mask = mask[:query.shape[1], kv_start : kv_start + sub_len].to(interm_dtype) * (-1e4)
        sub_value = value[:, kv_start: kv_start + sub_len, :]
        if reorder_ops:
            qk_result = qkMM1(query, sub_key * scale).to(interm_dtype)
        else:
            qk_result = qkMM1(query * scale, sub_key).to(interm_dtype)
        if softcap > 0.0:
            qk_result = softcap * torch.tanh(qk_result / softcap)
        if mask is not None:
            qk_result += sub_mask
        if kv_start == 0:
            gm = None
        p_result, row_sum, dm, gm = softmax1(qk_result, kv_start == 0, gm, interm_dtype, rescale_threshold)
        p_result = p_result.to(value.dtype)
        if kv_start == 0:
            gm_high = None
        lo = pvMM2(p_result, sub_value).to(interm_dtype)
        if kv_start == 0:
            gl = row_sum
            go = lo
        else:
            dm = torch.exp(dm)
            gl = gl * dm
            gl = gl + row_sum
            go = go * dm
            go = go + lo
    go = go / gl
    go = go.permute(1, 0, 2)
    lse = torch.squeeze((torch.log(gl) + gm), dim=-1).to(torch.float32)
    return go.to(dtype_og), lse

def softmax_numpy(sim, sink_matrix):
    import numpy as np

    if isinstance(sim, torch.Tensor):
        sim = sim.detach().cpu().numpy()
    if sink_matrix is not None and isinstance(sink_matrix, torch.Tensor):
        sink_matrix = sink_matrix.detach().cpu().numpy()
    row_max = np.max(sim, axis=-1, keepdims=True)
    valid_row_mask = ~np.isneginf(row_max)
    # add sink rowmax
    if sink_matrix is not None:
        assert sink_matrix.shape == row_max.shape, \
            f"sink_matrix 形状 {sink_matrix.shape} 与 row_max 形状 {row_max.shape} 不一致！"
        # 更新含sink的rowmax
        # row_max = np.maximum(row_max, sink_matrix)
        row_max[valid_row_mask] = np.maximum(
            row_max[valid_row_mask],
            sink_matrix[valid_row_mask]
        )

    sim_sub = sim - row_max
    sim_sub_high = sim.astype(np.float64) - row_max.astype(np.float64)

    sim_sub = np.exp(sim_sub)
    sim_sub_high = np.exp(sim_sub_high)
    row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
    row_sum_high = np.sum(sim_sub_high, axis=-1, keepdims=True)

    if sink_matrix is not None:
        sink_exp = np.exp(sink_matrix - row_max)
        sink_exp_high = np.exp(sink_matrix.astype(np.float64) - row_max.astype(np.float64))
        row_sum = row_sum + sink_exp
        row_sum_high = row_sum_high + sink_exp_high

    soft_res = sim_sub / row_sum
    lse = np.squeeze((np.log(row_sum_high) + row_max.astype(np.float64)), axis=-1)
    # lse = np.squeeze((np.log(row_sum) + row_max), axis=-1)

    return soft_res, lse, row_max

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
    scale = torch.tensor(scale).to(query.dtype)
    if reorder_ops:
        sim_high = group_matmul(query.shape[0], key.shape[0], query, key * scale)  # (head_num, q_seqlen, k_seqlen)
    else:
        sim_high = group_matmul(query.shape[0], key.shape[0], query * scale, key)  # (head_num, q_seqlen, k_seqlen)
    if mask is not None:
        sim_high = sim_high + (
            mask[:sim_high.shape[-2], :sim_high.shape[-1]]
            ).to(sim_high.dtype) * (-1e4)

    if upcast or sink_matrix is not None:
        import numpy as np

        p_high, lse_high, gm = softmax_numpy(sim_high, sink_matrix)
        lse_high = lse_high.astype(np.float64)
        p_high = torch.from_numpy(p_high).to(value.dtype)
    else:
        p_high = torch.softmax(sim_high, dim=-1).to(value.dtype)
        lse_high = torch.logsumexp(sim_high.to(torch.float32), dim=-1).cpu().numpy()

    out_high = group_matmul(query.shape[0], key.shape[0], p_high, value)
    out_high = out_high.permute(1, 0, 2)
    return out_high.to(dtype_og), lse_high

"""
小算子 FlashAttention 反向golden相关实现
"""

def broadcast_kv_single(num_heads, num_kv_heads, kv_tensor, dtype):
    factor = num_heads // num_kv_heads
    b, _, s, d = kv_tensor.shape
    kv_res = torch.zeros([b, num_heads, s, d], dtype=dtype)
    for i in range(num_heads):
        j = i // factor
        kv_res[:, i : i + 1, :, :] = kv_tensor[:, j : j + 1, :, :]
    return kv_res


def normalize_window(seq_q, seq_k, is_causal, window_size_left, window_size_right):
    wl = window_size_left
    wr = window_size_right
    if wl >= seq_k - 1:
        wl = -1
    if wr >= seq_q - 1:
        wr = -1
    if is_causal:
        wr = 0
    is_causal_out = wl < 0 and wr == 0
    is_local = (wl >= 0 or wr >= 0) and not is_causal_out
    return wl, wr, is_causal_out, is_local


def make_window_atten_mask(
    seq_q,
    seq_k,
    is_causal=False,
    window_size_left=-1,
    window_size_right=-1,
):
    wl, wr, is_causal_out, is_local = normalize_window(
        seq_q, seq_k, is_causal, window_size_left, window_size_right
    )
    if not is_causal_out and not is_local:
        return torch.tensor(0)

    offset = seq_k - seq_q
    shape = (seq_q, seq_k)
    if is_causal_out:
        return torch.triu(torch.ones(shape), diagonal=offset + 1)

    mask = torch.zeros(shape)
    if wr >= 0:
        mask = mask + torch.triu(torch.ones(shape), diagonal=wr + 1 + offset)
    if wl >= 0:
        mask = mask + torch.tril(torch.ones(shape), diagonal=-wl - 1 + offset)
    return mask


def _resolve_atten_mask(q, k, atten_mask, is_causal, window_size_left, window_size_right):
    if atten_mask is not None and len(atten_mask.shape) != 0:
        return atten_mask
    seq_q = q.shape[2]
    seq_k = k.shape[2]
    return make_window_atten_mask(
        seq_q, seq_k, is_causal, window_size_left, window_size_right
    )


def tsoftmax_grad(dp, softmax_res):
    muls = dp * softmax_res
    muls_r = muls.sum(dim=-1, keepdims=True)
    return (dp - muls_r) * softmax_res


def _keep_prob(dropout_p):
    return 1.0 - dropout_p


def _softcap_backward_factor(q, k, scale, softcap, compute_dtype, *, gtype=torch.float64):
    if softcap <= 0.0:
        return None
    qb = q.to(compute_dtype)
    kb = k.to(compute_dtype)
    qk = torch.matmul(qb, kb.permute(0, 1, 3, 2)).to(torch.float32).mul(scale)
    tanh_qk = torch.tanh(qk.to(gtype) / softcap)
    return 1.0 - tanh_qk * tanh_qk


def sum_gqa_grad(dk_or_dv, nheads, nheads_k, batch, seq_k, headdim):
    if nheads == nheads_k:
        return dk_or_dv
    g = nheads // nheads_k
    return (
        torch.sum(
            dk_or_dv.reshape(batch, nheads_k, g, seq_k, headdim),
            dim=2,
            keepdim=True,
        ).reshape(batch, nheads_k, seq_k, headdim)
    )


def softmax_res_from_fa_lse_bsnd(
    q_bn,
    k_bn,
    softmax_lse,
    scale,
    softcap,
    is_causal,
    window_size_left,
    window_size_right,
    compute_dtype,
    *,
    gtype=torch.float64,
):
    """q_bn / k_bn: (B, N, S, D)。softmax_lse: FA BSND (B, N, S_q)。"""
    atten_mask = _resolve_atten_mask(
        q_bn, k_bn, None, is_causal, window_size_left, window_size_right
    )
    qb = q_bn.to(compute_dtype)
    kb = k_bn.to(compute_dtype)
    qk = torch.matmul(qb, kb.permute(0, 1, 3, 2)).to(torch.float32).mul(scale)
    if softcap > 0.0:
        qk = softcap * torch.tanh(qk / softcap)
    if atten_mask is not None and len(atten_mask.shape) != 0:
        qk = qk + atten_mask.to(torch.float32) * (-40000.0)
    lse = softmax_lse.to(torch.float32).to(gtype)
    if lse.dim() != 3:
        raise ValueError(f"BSND softmax_lse 期望 3 维 (B,N,S)，实际 {tuple(lse.shape)}")
    lse = lse.unsqueeze(-1)
    softmax_res = torch.exp(qk.to(gtype) - lse)
    if atten_mask is not None and len(atten_mask.shape) != 0:
        softmax_res[atten_mask.bool().broadcast_to(softmax_res.shape)] = 0
    return softmax_res


def softmax_res_from_fa_lse_tnd_slice(
    q_bn,
    k_bn,
    softmax_lse_nt_slice,
    scale,
    softcap,
    is_causal,
    window_size_left,
    window_size_right,
    compute_dtype,
    *,
    gtype=torch.float64,
):
    """TND 单 batch 切片：FA varlen softmax_lse 切片 (N, S_q) NT。"""
    atten_mask = _resolve_atten_mask(
        q_bn, k_bn, None, is_causal, window_size_left, window_size_right
    )
    qb = q_bn.to(compute_dtype)
    kb = k_bn.to(compute_dtype)
    qk = torch.matmul(qb, kb.permute(0, 1, 3, 2)).to(torch.float32).mul(scale)
    if softcap > 0.0:
        qk = softcap * torch.tanh(qk / softcap)
    if atten_mask is not None and len(atten_mask.shape) != 0:
        qk = qk + atten_mask.to(torch.float32) * (-40000.0)
    lse = softmax_lse_nt_slice.to(torch.float32).to(gtype).unsqueeze(0).unsqueeze(-1)
    softmax_res = torch.exp(qk.to(gtype) - lse)
    if atten_mask is not None and len(atten_mask.shape) != 0:
        softmax_res[atten_mask.bool().broadcast_to(softmax_res.shape)] = 0
    return softmax_res


def tbackward_bsnd(dx, q, k, v, softmax_res, drop_mask, scale, softcap, dropout_p):
    keep_prob = _keep_prob(dropout_p)
    dp = torch.matmul(dx, v.permute(0, 1, 3, 2))
    if drop_mask is None or len(drop_mask.shape) == 0:
        drop_res = softmax_res.permute(0, 1, 3, 2)
        dp_drop = dp
    else:
        drop_res = softmax_res.mul(drop_mask).mul(1.0 / keep_prob).permute(0, 1, 3, 2)
        dp_drop = dp * drop_mask * (1.0 / keep_prob)
    dv = torch.matmul(drop_res, dx)
    softmax_grad_res = tsoftmax_grad(dp_drop, softmax_res) * scale
    softcap_factor = _softcap_backward_factor(q, k, scale, softcap, q.dtype)
    if softcap_factor is not None:
        softmax_grad_res = softmax_grad_res * softcap_factor
    dq = torch.matmul(softmax_grad_res, k)
    dk = torch.matmul(softmax_grad_res.permute(0, 1, 3, 2), q)
    return dq, dk, dv


def tbackward_tnd(dx, q, k, v, softmax_res, drop_mask, scale, softcap, dropout_p):
    keep_prob = _keep_prob(dropout_p)
    if drop_mask is None or len(drop_mask.shape) == 0:
        drop_res = softmax_res.permute(0, 1, 3, 2)
        dp_drop = torch.matmul(dx, v.permute(0, 1, 3, 2))
    else:
        drop_res = softmax_res.mul(drop_mask).mul(1.0 / keep_prob).permute(0, 1, 3, 2)
        dp = torch.matmul(dx, v.permute(0, 1, 3, 2))
        dp_drop = dp * drop_mask * (1.0 / keep_prob)
    dv = torch.matmul(drop_res, dx)
    softmax_grad_res = tsoftmax_grad(dp_drop, softmax_res) * scale
    softcap_factor = _softcap_backward_factor(q, k, scale, softcap, q.dtype)
    if softcap_factor is not None:
        softmax_grad_res = softmax_grad_res * softcap_factor
    dq = torch.matmul(softmax_grad_res, k)
    dk = torch.matmul(softmax_grad_res.permute(0, 1, 3, 2), q)
    return dq, dk, dv


def golden_bsnd_bwd_from_fwd(
    q,
    k,
    v,
    dout,
    out,
    softmax_lse,
    nheads,
    nheads_k,
    scale,
    softcap,
    dropout_p,
    is_causal,
    window_size_left,
    window_size_right,
    *,
    gtype=torch.float64,
    drop_mask=None,
):
    """BSND 反传标杆。softmax_lse layout: FA (B, N, S_q)。"""
    del out
    batch, seq_q, _, headdim = q.shape
    seq_k = k.shape[1]
    compute_dtype = q.dtype

    q_bn = q.detach().cpu().permute(0, 2, 1, 3).to(gtype)
    k_bn = k.detach().cpu().permute(0, 2, 1, 3).to(gtype)
    v_bn = v.detach().cpu().permute(0, 2, 1, 3).to(gtype)
    dx_bn = dout.detach().cpu().permute(0, 2, 1, 3).to(gtype)
    lse_cpu = softmax_lse.detach().cpu().to(torch.float32)

    if nheads == nheads_k:
        k_new, v_new = k_bn, v_bn
    else:
        k_new = broadcast_kv_single(nheads, nheads_k, k_bn, gtype)
        v_new = broadcast_kv_single(nheads, nheads_k, v_bn, gtype)

    softmax_res = softmax_res_from_fa_lse_bsnd(
        q_bn,
        k_new,
        lse_cpu,
        scale,
        softcap,
        is_causal,
        window_size_left,
        window_size_right,
        compute_dtype,
        gtype=gtype,
    )
    if drop_mask is None:
        drop_mask = torch.tensor(1)
    dq_bn, dk_bn, dv_bn = tbackward_bsnd(
        dx_bn, q_bn, k_new, v_new, softmax_res, drop_mask, scale, softcap, dropout_p
    )
    dk_bn = sum_gqa_grad(dk_bn, nheads, nheads_k, batch, seq_k, headdim)
    dv_bn = sum_gqa_grad(dv_bn, nheads, nheads_k, batch, seq_k, headdim)

    dq = dq_bn.permute(0, 2, 1, 3).to(compute_dtype)
    dk = dk_bn.permute(0, 2, 1, 3).to(compute_dtype)
    dv = dv_bn.permute(0, 2, 1, 3).to(compute_dtype)
    return dq, dk, dv


def golden_tnd_bwd_from_fwd(
    q,
    k,
    v,
    dout,
    out,
    softmax_lse,
    nheads,
    nheads_k,
    seqlens_q,
    seqlens_k,
    scale,
    softcap,
    dropout_p,
    is_causal,
    window_size_left,
    window_size_right,
    *,
    gtype=torch.float64,
    drop_mask=None,
):
    """TND 反传标杆。softmax_lse layout: FA varlen (N, total_q) NT。"""
    del out
    seqlens_q = list(seqlens_q)
    seqlens_k = list(seqlens_k)
    cu_q = [0]
    cu_k = [0]
    for sq, sk in zip(seqlens_q, seqlens_k):
        cu_q.append(cu_q[-1] + int(sq))
        cu_k.append(cu_k[-1] + int(sk))
    headdim = q.shape[-1]
    compute_dtype = q.dtype
    lse_nt = softmax_lse.detach().cpu().to(torch.float32)
    if lse_nt.dim() != 2:
        raise ValueError(f"TND softmax_lse 期望 NT (N, total_q)，实际 {tuple(lse_nt.shape)}")

    dq_golden = torch.empty_like(q, dtype=compute_dtype)
    dk_golden = torch.empty_like(k, dtype=compute_dtype)
    dv_golden = torch.empty_like(v, dtype=compute_dtype)
    if drop_mask is None:
        drop_mask = torch.tensor(1)

    for i, sq in enumerate(seqlens_q):
        sk = seqlens_k[i]
        if sq == 0 or sk == 0:
            continue

        qi = q[cu_q[i] : cu_q[i + 1]].detach().cpu().unsqueeze(0).permute(0, 2, 1, 3).to(gtype)
        ki = k[cu_k[i] : cu_k[i + 1]].detach().cpu().unsqueeze(0).permute(0, 2, 1, 3).to(gtype)
        vi = v[cu_k[i] : cu_k[i + 1]].detach().cpu().unsqueeze(0).permute(0, 2, 1, 3).to(gtype)
        dxi = dout[cu_q[i] : cu_q[i + 1]].detach().cpu().unsqueeze(0).permute(0, 2, 1, 3).to(gtype)
        lse_i = lse_nt[:, cu_q[i] : cu_q[i + 1]]

        if nheads == nheads_k:
            ki_new, vi_new = ki, vi
        else:
            ki_new = broadcast_kv_single(nheads, nheads_k, ki, gtype)
            vi_new = broadcast_kv_single(nheads, nheads_k, vi, gtype)

        softmax_res_i = softmax_res_from_fa_lse_tnd_slice(
            qi,
            ki_new,
            lse_i,
            scale,
            softcap,
            is_causal,
            window_size_left,
            window_size_right,
            compute_dtype,
            gtype=gtype,
        )
        drop_mask_i = (
            drop_mask if drop_mask.dim() == 0 else drop_mask[i][:, :sq, :sk]
        )
        dqi, dki, dvi = tbackward_tnd(
            dxi, qi, ki_new, vi_new, softmax_res_i, drop_mask_i, scale, softcap, dropout_p
        )
        dki = sum_gqa_grad(dki, nheads, nheads_k, 1, sk, headdim)
        dvi = sum_gqa_grad(dvi, nheads, nheads_k, 1, sk, headdim)

        dq_golden[cu_q[i] : cu_q[i + 1]] = (
            dqi.permute(0, 2, 1, 3).reshape(sq, nheads, headdim).to(compute_dtype)
        )
        dk_golden[cu_k[i] : cu_k[i + 1]] = (
            dki.permute(0, 2, 1, 3).reshape(sk, nheads_k, headdim).to(k.dtype)
        )
        dv_golden[cu_k[i] : cu_k[i + 1]] = (
            dvi.permute(0, 2, 1, 3).reshape(sk, nheads_k, headdim).to(v.dtype)
        )

    return dq_golden, dk_golden, dv_golden

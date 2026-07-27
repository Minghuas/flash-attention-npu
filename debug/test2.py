#!/usr/bin/env python3
# debug/test2.py
# RUNTIME-DEBUG script for ALiBi position-parameter verification.
#
# Unlike test1.py (Q=K=0, only the bias is visible), this script uses ENCODED
# Q/K tensors so that the Score matrix S = Q @ K^T itself carries the
# (batch, head, q_seq, k_seq) position info.
#
# IMPORTANT — bfloat16 precision constraint:
#   bfloat16 has only 7 mantissa bits (8 effective), so it can represent
#   integers EXACTLY only up to 256. Above that, values get rounded (e.g.
#   10000 -> 9984 because spacing is 64 in that range). So we must NOT pack
#   batch/head/q_seq into one large number (the old prompt.md scheme fails on
#   bfloat16). Instead we use SEPARATE embedding channels, each holding a
#   small value (< 256), and a 4-token K "probe" group per logical key.
#
# Encoding (all values < 256, bfloat16-exact):
#     Q[..., 0] = q_seq      (requires Sq <= 256)
#     Q[..., 1] = head_idx   (requires H  <= 256)
#     Q[..., 2] = batch_idx  (requires B  <= 256)
#     Q[..., 3] = 1          (probe dimension for k_seq)
#
#     For each logical key position j, 4 K tokens:
#         K[4*j]   = [1, 0, 0, 0]   -> S[row, 4*j]   = q_seq
#         K[4*j+1] = [0, 1, 0, 0]   -> S[row, 4*j+1] = head_idx
#         K[4*j+2] = [0, 0, 1, 0]   -> S[row, 4*j+2] = batch_idx
#         K[4*j+3] = [0, 0, 0, j]   -> S[row, 4*j+3] = k_seq (= j)
#
# At the breakpoint in ApplyAlibiRows, read the score tile BEFORE bias:
#     col % 4 == 0 -> q_seq       (verify vs qSBlockBaseIdx + token)
#     col % 4 == 1 -> head_idx    (verify vs qNBlockBaseIdx + absRow/qSBlockSize)
#     col % 4 == 2 -> batch_idx   (verify vs current batch)
#     col % 4 == 3 -> k_seq = col // 4   (verify vs kvSStartIdx region)
#
# Run: python debug/test2.py [--batch B] [--heads H] [--kv-h KV] [--sq SQ] [--sk SK] [--causal]
# NOTE: Sk must be a MULTIPLE OF 4 (logical_sk = Sk // 4), and Sq, logical_sk <= 256.

import sys
import argparse
import torch
import torch_npu
from flash_attn_npu_v2 import flash_attn_func

# ============================================================================
# debug Q/K encoding  (bfloat16-safe: one channel per field, all values < 256)
# ============================================================================
def decode_score_col(col_value, col_idx):
    """
    Given a score value S[row, col] (float, may be read from the float32 score
    tile in the kernel) and its column index, decode what it represents:
        col % 4 == 0 -> q_seq      (value == q_seq of this row's Q token)
        col % 4 == 1 -> head_idx   (value == head index of this row)
        col % 4 == 2 -> batch_idx  (value == batch index of this row)
        col % 4 == 3 -> k_seq      (value == col // 4 = logical key position)
    Returns (kind, value).
    """
    v = int(round(float(col_value)))
    kind = ["q_seq", "head_idx", "batch_idx", "k_seq"][col_idx % 4]
    return kind, v

def make_debug_qk(B, Sq, Sk, H, KVH, D, dtype):
    """
    Build encoded Q [B, Sq, H, D] and K [B, Sk, KVH, D].
    Sk must be a multiple of 4. logical key length = Sk // 4.

    All stored values are < 256 so they are exact in bfloat16.

    Q[..., 0] = q_seq ; Q[..., 1] = head_idx ; Q[..., 2] = batch_idx ; Q[..., 3] = 1
    K 4-token probe group per logical key j:
        K[b, 4j,   kvh, :] = [1, 0, 0, 0]
        K[b, 4j+1, kvh, :] = [0, 1, 0, 0]
        K[b, 4j+2, kvh, :] = [0, 0, 1, 0]
        K[b, 4j+3, kvh, :] = [0, 0, 0, j]
    => S[h, q, 4j]   = q_seq(b,q,h)
       S[h, q, 4j+1] = head_idx(h)
       S[h, q, 4j+2] = batch_idx(b)
       S[h, q, 4j+3] = k_seq = j
    """
    assert Sk % 4 == 0, f"Sk must be a multiple of 4 for 4-token probe K, got Sk={Sk}"
    logical_sk = Sk // 4
    # assert Sq <= 256, f"Sq must be <= 256 for bfloat16-exact q_seq, got Sq={Sq}"
    # assert logical_sk <= 256, f"logical_sk (Sk//4) must be <= 256, got {logical_sk}"

    q = torch.zeros(B, Sq, H, D, dtype=dtype)
    k = torch.zeros(B, Sk, KVH, D, dtype=dtype)

    # Q encoding: channel 0=q_seq, 1=head, 2=batch, 3=1(probe)
    for b in range(B):
        for h in range(H):
            for qs in range(Sq):
                q[b, qs, h, 0] = qs
                q[b, qs, h, 1] = h
                q[b, qs, h, 2] = b
                q[b, qs, h, 3] = 1

    # K encoding: 4-token probe group per logical key position j
    for b in range(B):
        for kvh in range(KVH):
            for j in range(logical_sk):
                base = 4 * j
                k[b, base,     kvh, 0] = 1          # probe q_seq
                k[b, base + 1, kvh, 1] = 1          # probe head_idx
                k[b, base + 2, kvh, 2] = 1          # probe batch_idx
                k[b, base + 3, kvh, 3] = base + 3          # carry k_seq

    return q, k


# ---------- slopes (deterministic, different per head) ----------
def make_alibi_slopes(batch_size, num_heads):
    """Simple integer slopes 1, 2, 3, ... per head for easy debugging."""
    _h = torch.tensor([h + 1 for h in range(num_heads)], dtype=torch.float32)
    return _h.unsqueeze(0).repeat(batch_size, 1)

# ---------- reference helpers (from test_flash_attn_npu_v2.py) ----------
def group_matmul(head, kv_head, left, right, high_prec=1):
    group_num = head // kv_head
    score = None
    for i in range(kv_head):
        if high_prec == 0:
            group_score = torch.matmul(left[i * group_num:(i + 1) * group_num, :, :].to(torch.float32),
                                        right[i:(i + 1), :, :].to(torch.float32)).to(torch.float32)
        else:
            group_score = torch.matmul(left[i * group_num:(i + 1) * group_num, :, :].to(torch.float32),
                                        right[i:(i + 1), :, :].to(torch.float32))
        if score is None:
            score = group_score
        else:
            score = torch.cat((score, group_score), 0)
    return score

def softmax1(qk_result, is_first, gm, interm_dtype=torch.float16):
    sim = qk_result.to(interm_dtype)
    lm = torch.max(sim, dim=-1, keepdims=True)[0]
    if is_first:
        hm = lm
        dm = 0
    else:
        hm = torch.maximum(gm, lm)
        dm = gm - hm
    gm = hm
    sim_sub = sim - hm
    sim_sub = torch.exp(sim_sub.to(interm_dtype))
    row_sum = torch.sum(sim_sub, dim=-1, keepdims=True)
    return sim_sub, row_sum, dm, gm

def qkMM1(query, key):
    result = None
    qk_k = key.shape[1]
    qk_k_split = 128
    qk_k_loop = (qk_k + 127) // 128
    for qk_k_loop_idx in range(qk_k_loop):
        sub_k = 128 if qk_k_loop_idx != (qk_k_loop - 1) else (qk_k - qk_k_loop_idx * 128)
        partial_Query = query[:, :, qk_k_loop_idx * 128: qk_k_loop_idx * 128 + sub_k]
        partial_Key = key[:, qk_k_loop_idx * 128: qk_k_loop_idx * 128 + sub_k, :]
        result_split = group_matmul(partial_Query.shape[0], partial_Key.shape[0], partial_Query, partial_Key, 0)
        if result is None:
            result = result_split
        else:
            result = result + result_split
    return result

def pvMM2(p, value):
    result = None
    pv_k = value.shape[1]
    pv_k_split = 128
    pv_k_loop = (pv_k + 127) // 128
    for pv_k_loop_idx in range(pv_k_loop):
        sub_k = 128 if pv_k_loop_idx != (pv_k_loop - 1) else (pv_k - pv_k_loop_idx * 128)
        partial_P = p[:, :, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k]
        partial_Value = value[:, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k, :]
        result_split = group_matmul(partial_P.shape[0], partial_Value.shape[0], partial_P, partial_Value, 0)
        if result is None:
            result = result_split
        else:
            result = result + result_split
    return result

# ---------- reference flash attention ----------
def ref_flash_attention(
    query,
    key,
    value,
    scale,
    mask,
    data_type,
    softcap,
    alibi_slopes=None,
    is_causal=False,
    ):
    inner_prec = 0
    interm_dtype = torch.float16 if inner_prec == 1 else torch.float32
    query = query.permute(1, 0, 2)
    key = key.permute(1, 2, 0)
    value = value.permute(1, 0, 2)
    scale = torch.tensor(scale)
    scale = scale.to(torch.float16) if inner_prec == 1 else scale.to(torch.float32)
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
        qk_result = qkMM1(query, sub_key).to(interm_dtype)
        qk_result = qk_result * scale
        if softcap > 0.0:
            qk_result = softcap * torch.tanh(qk_result / softcap)
        if alibi_slopes is not None:
            # Unified ALiBi bias: -slope_h * |i_abs - j_abs| for ALL mask types.
            sk_chunk = qk_result.shape[2]
            j_abs = kv_start + torch.arange(sk_chunk, dtype=torch.float32)
            slopes = alibi_slopes.to(torch.float32).reshape(-1, 1, 1)  # [nheads, 1, 1]
            i_abs = (context_len - qk_result.shape[1]) + torch.arange(qk_result.shape[1], dtype=torch.float32)
            dist = torch.abs(i_abs.reshape(-1, 1) - j_abs.reshape(1, -1))  # [Sq, sk]
            qk_result = qk_result + (-slopes) * dist.unsqueeze(0)
        if mask is not None:
            qk_result += sub_mask
        if kv_start == 0:
            gm = None
        p_result, row_sum, dm, gm = softmax1(qk_result, kv_start == 0, gm, interm_dtype)
        p_result = p_result.to(data_type)
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
    return go.to(data_type), lse

# ---------- main ----------
def main(args):
    B, H, KVH, Sq, Sk, D = args.batch, args.heads, args.kv_heads, args.sq, args.sk, args.hdim
    dtype = torch.bfloat16
    # scale = 1.0 / (D ** 0.5)
    scale = 1.0 

    # ---- encoded Q/K (bfloat16-safe, 4-token probe) ----
    q_cpu, k_cpu = make_debug_qk(B, Sq, Sk, H, KVH, D, dtype)
    # V = logical key position (col//4) so output = weighted-avg logical key position
    logical_sk = Sk // 4
    v_cpu = torch.arange(Sk, dtype=dtype).div(4, rounding_mode='floor').reshape(1, Sk, 1, 1).expand(B, Sk, KVH, D).contiguous().clone()

    q = q_cpu.clone().npu()
    k = k_cpu.clone().npu()
    v = v_cpu.clone().npu()
    slopes = make_alibi_slopes(B, H).npu()
    slopes_cpu = make_alibi_slopes(B, H)

    print(f"==> ENCODED Q/K (runtime debug), batch={B} H={H} kvH={KVH} Sq={Sq} Sk={Sk} (logical_sk={logical_sk}) D={D} causal={args.causal}")
    print(f"    slopes per head: {slopes_cpu.tolist()[0]}")
    print()
    print(f"    >>> SET BREAKPOINT in alibi.hpp ApplyAlibiRows<NO_MASK> (~line 124)")
    print(f"    >>> READ scoreUb/lsUbTensor BEFORE AddBiasToRow, decode by column % 4:")
    print(f"        col % 4 == 0 -> q_seq     (== qSBlockBaseIdx + token)")
    print(f"        col % 4 == 1 -> head_idx  (== qNBlockBaseIdx + absRow/qSBlockSize)")
    print(f"        col % 4 == 2 -> batch_idx (== current batch)")
    print(f"        col % 4 == 3 -> k_seq     (== col // 4, in kvSStartIdx region)")
    print()

    # sanity: show the expected score pattern for (b=0, h=0, q=0..2), cols 0..11
    print(f"    ---- expected S pattern (b=0, h=0, first 3 q rows, cols 0..11 = 3 probe groups) ----")
    print(f"    col:       0    1    2    3    4    5    6    7    8    9   10   11")
    print(f"    means:   qseq head bidx kseq qseq head bidx kseq qseq head bidx kseq")
    for qs in range(min(3, Sq)):
        vals = []
        for j in range(3):  # 3 logical key positions
            vals += [qs, 0, 0, j]
        print(f"    q_seq={qs}: {vals}")

    # ---- kernel call ----
    out, lse, _ = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale,
                                  causal=args.causal, window_size=(-1, -1),
                                  softcap=0.0, alibi_slopes=slopes, return_attn_probs=True)

    # ---- reference call ----
    data_type_cpu = torch.bfloat16
    causal_mask = None
    if args.causal:
        causal_mask = torch.triu(torch.ones(Sq, Sk), diagonal=Sk - Sq + 1).to(torch.bool)

    ref_out_list = []
    ref_lse_list = []
    for b in range(B):
        ref_o, ref_l = ref_flash_attention(
            q_cpu[b], k_cpu[b], v_cpu[b], scale, causal_mask, data_type_cpu,
            softcap=0.0, alibi_slopes=slopes_cpu[b], is_causal=args.causal)
        ref_out_list.append(ref_o)
        ref_lse_list.append(ref_l)

    ref_out = torch.stack(ref_out_list, dim=0)  # [B, Sq, H, D]
    ref_lse = torch.stack(ref_lse_list, dim=0)  # [B, H, Sq]

    # ---- compare ----
    out_cpu = out.cpu()
    lse_cpu = lse.cpu()

    out_diff = (out_cpu.float() - ref_out.float()).abs()
    lse_diff = (lse_cpu.float() - ref_lse.float()).abs()

    print()
    print(f"    kernel out.shape={tuple(out.shape)}  lse.shape={tuple(lse.shape)}")
    print(f"    ref    out.shape={tuple(ref_out.shape)}  lse.shape={tuple(ref_lse.shape)}")
    print(f"    out  range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    print(f"    lse  range: [{lse.min().item():.4f}, {lse.max().item():.4f}]")
    print()
    print(f"    ---- comparison against ref_flash_attention ----")
    print(f"    out  max diff: {out_diff.max().item():.6f}  mean: {out_diff.mean().item():.6f}")
    print(f"    lse  max diff: {lse_diff.max().item():.6f}  mean: {lse_diff.mean().item():.6f}")

    print('---------------------------------------')
    print(f"    ---- per-head overview ----")
    for h in range(H):
        out_diff_h = (out_cpu[0, :, h, :].float() - ref_out[0, :, h, :].float()).abs()
        lse_diff_h = (lse_cpu[0, h, :].float() - ref_lse[0, h, :].float()).abs()
        o_max = out_diff_h.max().item()
        l_max = lse_diff_h.max().item()
        marker = "⚠" if (o_max > 1e-2 or l_max > 1e-2) else "✓"
        print(f"    head={h} slope={slopes_cpu[0,h].item():.0f}: out_maxdiff={o_max:.4f}  lse_maxdiff={l_max:.4f}  {marker}")

    print()
    print(f"    ---- head 0, first 4 rows ----")
    print(f"    Kernel out[0, 0:4, 0, 0:4]:")
    print(f"    {out_cpu[0, :4, 0, :4]}")
    print(f"    Ref    out[0, 0:4, 0, 0:4]:")
    print(f"    {ref_out[0, :4, 0, :4]}")
    print(f"    Kernel lse[0, 0, :8]:")
    print(f"    {lse_cpu[0, 0, :8]}")
    print(f"    Ref    lse[0, 0, :8]:")
    print(f"    {ref_lse[0, 0, :8]}")

    if H > KVH:
        group_size = H // KVH
        for kv_h in range(KVH):
            qh_start = kv_h * group_size
            print(f"    ---- KV-head={kv_h} → Q-heads {qh_start}:{qh_start+group_size}, rows 0:3 ----")
            for qh in range(qh_start, qh_start + group_size):
                print(f"    Q-head={qh}: kernel={out_cpu[0, :3, qh, 0].tolist()}")
                print(f"    Q-head={qh}: ref    ={ref_out[0, :3, qh, 0].tolist()}")

    if out_diff.max().item() > 1e-2 or lse_diff.max().item() > 1e-2:
        print(f"    ⚠ MISMATCH DETECTED!", flush=True)
        print(f"    out   max diff: {out_diff.max().item():.6f}  mean: {out_diff.mean().item():.6f}", flush=True)
        print(f"    lse   max diff: {lse_diff.max().item():.6f}  mean: {lse_diff.mean().item():.6f}", flush=True)
    else:
        print(f"    ✅ PASS — kernel matches reference!", flush=True)
    print('---------------------------------------')

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ALiBi runtime debug: encoded Q/K, breakpoint in ApplyAlibiRows")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--kv-h", "--kv-heads", dest="kv_heads", type=int, default=2)
    p.add_argument("--sq", type=int, default=16, help="query seqlen (must be <= 256)")
    p.add_argument("--sk", type=int, default=32, help="key seqlen (MUST be multiple of 4; logical_sk = sk//4, must be <= 256)")
    p.add_argument("--hdim", type=int, default=128)
    p.add_argument("--causal", action="store_true")
    args = p.parse_args()

    # # 调试
    args.batch = 2
    args.heads = 4
    args.kv_heads = 2
    args.sq = 512
    args.sk = 1024
    args.hdim = 128
    torch_npu.npu.set_device(5)
    
    # torch_npu.npu.set_device(0)
    
    main(args)

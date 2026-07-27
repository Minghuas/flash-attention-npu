#!/usr/bin/env python3
# debug/test1.py
# Standalone debug script for ALiBi. Q=K=V=0 => score (Q@K^T) starts at 0,
# so the output of ApplyAlibiRows IS the alibi bias itself. Run with:
#   python debug/test1.py [--causal] [--batch B] [--heads H] [--kv-h KV] [--sq SQ] [--sk SK]
# Set a breakpoint in ApplyAlibiRows<NO_MASK> (alibi.hpp:~124) to inspect the bias
# computation in the actual kernel context.

import sys
import argparse
import torch
import torch_npu
from flash_attn_npu_v2 import flash_attn_func

torch_npu.npu.set_device(0)

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
            # i_abs includes diffS = kv_seqlen - q_seqlen (right-aligned query);
            # causal mask (j>i -> -inf) is applied separately by the `mask` block below.
            sk_chunk = qk_result.shape[2]
            j_abs = kv_start + torch.arange(sk_chunk, dtype=torch.float32)
            slopes = alibi_slopes.to(torch.float32).reshape(-1, 1, 1)  # [nheads, 1, 1]
            i_abs = (context_len - qk_result.shape[1]) + torch.arange(qk_result.shape[1], dtype=torch.float32)
            dist = torch.abs(i_abs.reshape(-1, 1) - j_abs.reshape(1, -1))  # [Sq, sk]
            qk_result = qk_result + (-slopes) * dist.unsqueeze(0)  # [H,Sq,sk] + [H,1,1]*[1,Sq,sk] -> [H,Sq,sk]
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

# ---------- reference bias ----------
def ref_bias(slope, i_q, kv_start, ncols):
    """For one head, one row: -slope * |i_q - (kv_start + col)|."""
    cols = torch.arange(kv_start, kv_start + ncols, dtype=torch.float32)
    return -slope * (torch.tensor(i_q, dtype=torch.float32) - cols).abs()

# ---------- main ----------
def main(args):
    B, H, KVH, Sq, Sk, D = args.batch, args.heads, args.kv_heads, args.sq, args.sk, args.hdim
    dtype = torch.bfloat16
    scale = 1.0 / (D ** 0.5)

    # Q=K=0 => score = pure bias; V = position index => output = weighted avg position
    q = torch.zeros(B, Sq, H, D, dtype=dtype).npu()
    k = torch.zeros(B, Sk, KVH, D, dtype=dtype).npu()
    v_cpu = torch.arange(Sk, dtype=dtype).reshape(1, Sk, 1, 1).expand(B, Sk, KVH, D).contiguous().clone()
    # print(v_cpu[0, :4, :4, :4])
    v = v_cpu.clone().npu()
    slopes = make_alibi_slopes(B, H).npu()
    slopes_cpu = make_alibi_slopes(B, H)

    print(f"==> Q=K=0 V=[0..Sk-1], batch={B} H={H} kvH={KVH} Sq={Sq} Sk={Sk} D={D} causal={args.causal}")
    print(f"    slopes per head: {slopes_cpu.tolist()[0]}")
    print()

    # ----- kernel call -----
    out, lse, _ = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale,
                                  causal=args.causal, window_size=(-1, -1),
                                  softcap=0.0, alibi_slopes=slopes, return_attn_probs=True)

    # ----- reference call -----
    # ref_flash_attention expects 3D inputs: [seqlen, heads, head_dim]
    data_type_cpu = torch.bfloat16
    causal_mask = None
    if args.causal:
        # causal mask: j > i -> mask out
        causal_mask = torch.triu(torch.ones(Sq, Sk), diagonal=Sk - Sq + 1).to(torch.bool)

    ref_out_list = []
    ref_lse_list = []
    for b in range(B):
        q_cpu = q[b].cpu()  # [Sq, H, D]
        k_cpu = k[b].cpu()  # [Sk, KVH, D]
        v_cpu = v[b].cpu()  # [Sk, KVH, D]
        s_cpu = slopes_cpu[b]  # [H]

        ref_o, ref_l = ref_flash_attention(
            q_cpu, k_cpu, v_cpu, scale, causal_mask, data_type_cpu,
            softcap=0.0, alibi_slopes=s_cpu, is_causal=args.causal)
        # ref_o: [Sq, H, D], ref_l: [H, Sq]
        ref_out_list.append(ref_o)
        ref_lse_list.append(ref_l)

    ref_out = torch.stack(ref_out_list, dim=0)  # [B, Sq, H, D]
    ref_lse = torch.stack(ref_lse_list, dim=0)  # [B, H, Sq]

    # ----- compare -----
    out_cpu = out.cpu()
    lse_cpu = lse.cpu()

    print(f"    kernel out.shape={tuple(out.shape)}  lse.shape={tuple(lse.shape)}")
    print(f"    ref    out.shape={tuple(ref_out.shape)}  lse.shape={tuple(ref_lse.shape)}")

    out_diff = (out_cpu.float() - ref_out.float()).abs()
    lse_diff = (lse_cpu.float() - ref_lse.float()).abs()
    print(f"    out  range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    print(f"    lse  range: [{lse.min().item():.4f}, {lse.max().item():.4f}]")
    print()

    print(f"    ---- comparison against ref_flash_attention ----")
    print(f"    out  max diff: {out_diff.max().item():.6f}")
    print(f"    out  mean diff: {out_diff.mean().item():.6f}")
    print(f"    lse  max diff: {lse_diff.max().item():.6f}")
    print(f"    lse  mean diff: {lse_diff.mean().item():.6f}")

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

    # If GQA fails, also print head that shares this KV head
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
        # Dump full diff stats for analysis
        print(f"    out   max diff: {out_diff.max().item():.6f}  mean: {out_diff.mean().item():.6f}", flush=True)
        print(f"    lse   max diff: {lse_diff.max().item():.6f}  mean: {lse_diff.mean().item():.6f}", flush=True)
    else:
        print(f"    ✅ PASS — kernel matches reference!", flush=True)
    print('---------------------------------------')

    # print()
    # print(f"    ---- reference bias for head-0 (what kernel should produce) ----")
    # s0 = slopes_cpu[0, 0].item()
    # diffS = max(0, Sk - Sq)  # non-causal right-aligned
    # for i in range(min(4, Sq)):
    #     i_q = diffS + i
    #     b = ref_bias(s0, i_q, kv_start=0, ncols=Sk)
    #     print(f"    i_q={i_q} slope={s0}: {b.tolist()}")

    # print()
    # print("    Since Q=K=V=0, the output is softmax(bias)@0 ≈ 0.")
    # print("    To see the bias: set a breakpoint in alibi.hpp ApplyAlibiRows<NO_MASK>")
    # print("    after BuildAbsBiasRow, inspect workUb (the bias row).")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ALiBi debug: Q=K=V=0, score=bias")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--kv-h", "--kv-heads", dest="kv_heads", type=int, default=2)
    p.add_argument("--sq", type=int, default=512, help="query seqlen")  # (1024,1024)组合报错
    p.add_argument("--sk", type=int, default=512, help="key   seqlen")
    p.add_argument("--hdim", type=int, default=128)
    p.add_argument("--causal", action="store_true")
    args = p.parse_args()
    main(args)

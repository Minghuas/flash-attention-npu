#!/usr/bin/env python3
# S3 判刀 W（devlog #44.7）：tile 间/行间/列间全区分的结构化输入。
#   Q[b,s,h,d] = (h+1)*(s+1)     —— tile（head）间、行（s）间全不同
#   K[j,d]     = j               —— 列（kv 位置）间不同 → P 非均匀、sum 非平凡
#   V[k]       = k               —— 列加权 → O 对 P 的列模式敏感
# 关键性质：tile h 与 tile h-1 的 S/P/sum 全部不同 →
#   任何"坏行读到上一 tile 对应行的旧值"型竞态都直接显现（与全 1/V=1 判刀的
#   掩盖性相反）。正确值由 CPU 参考给出，坏行数值可反推错读来源。
# 用法：conda activate FA2 && python debug/test_splitb_s3_stage3.py
#       （不带 DUMP 先测撞率；撞到 FAIL 后同输入加 DUMP=1 抓现场）
import os
os.environ["FLASH_ATTN_FORCE_SPLITB"] = "1"

import torch
import torch_npu
from flash_attn_npu import flash_attn_func
torch.npu.set_device(1)

# B, Sq, Sk, H, D = 2, 64, 64, 8, 128    # B=2：槽 0/槽 1 各用一次的最小复现形态
B, Sq, Sk, H, D = 1, 64, 64, 2, 16    # B=2：槽 0/槽 1 各用一次的最小复现形态
SCALE = 1.0 / (D ** 0.5)


def make_inputs(dtype):
    """结构化输入（fp32 构造后 cast，保证 fp16/bf16 下数值确定性）。
    Q=(h+1)(s+1)/512 ∈ [0.002, 1]、K=(j+1)/64 ∈ [0.016, 1] →
    S 范围 ≈ [2.8e-4, 11.3]（scale 后）——softmax 有区分度且不溢出。"""
    q = torch.zeros(B, Sq, H, D, dtype=torch.float32)
    for s in range(Sq):
        for h in range(H):
            q[:, s, h, :] = (h + 1) * (s + 1) / 512.0
    k = torch.zeros(B, Sk, H, D, dtype=torch.float32)
    for j in range(Sk):
        k[:, j, :, :] = (j + 1) / 64.0
    v = torch.zeros(B, Sk, H, D, dtype=torch.float32)
    for j in range(Sk):
        v[:, j, :, :] = j + 1
    return q.to(dtype).npu(), k.to(dtype).npu(), v.to(dtype).npu()


def cpu_ref(q, k, v):
    """BSND → BHND 参考（与 test_splitb_s3.py torch_ref 同款）。
    返回 (o, scores, p_unorm, maxv, sumv, otmp)：与 kernel 各段产物一一对照"""
    qf = q.float().transpose(1, 2)
    kf = k.float().transpose(1, 2)
    vf = v.float().transpose(1, 2)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * SCALE   # [B,H,Sq,Sk]（kernel S 段 = ×SCALE 前？）
    scores_raw = scores / SCALE                               # [B,H,Sq,Sk] = QK 原始内积（kernel gS）
    maxv = scores.max(dim=-1, keepdim=True).values            # [B,H,Sq,1]
    p_unorm = torch.exp(scores - maxv)                        # kernel P（未归一 exp，fp16/bf16）
    sumv = p_unorm.sum(dim=-1)                                # [B,H,Sq] kernel stats sum
    p = p_unorm / sumv.unsqueeze(-1)
    o = torch.matmul(p, vf).transpose(1, 2)                   # [B,Sq,H,D]
    otmp = torch.matmul(p_unorm, vf).transpose(1, 2)          # [B,Sq,H,D] kernel OTmp（未除 sum）
    return o, scores_raw, p_unorm.transpose(1, 2), maxv.transpose(1, 2).squeeze(-1), sumv.transpose(1, 2), otmp


def run_once(q, k, v, o_ref, itr):
    out = flash_attn_func(q, k, v, 0.0, SCALE, False)
    torch.npu.synchronize()
    # 参考值 cast 到输出 dtype 再还原：消除 bf16 输出粒度（33.44→33.5 是正确舍入）
    # 的假阳性，只留真实错误
    err = (out.float().cpu() - o_ref.to(torch.bfloat16).float()).abs()   # [B,Sq,H,D]
    maxerr = err.max().item()
    ok = maxerr < 2e-2
    print(f"[itr{itr}] max_err={maxerr:.4f} => {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        am = torch.unravel_index(err.argmax(), err.shape)
        rows = err.amax(dim=(0, 2, 3))                        # per-s
        bad = (rows > 1e-2).nonzero().flatten().tolist()
        b0, h0 = int(am[0]), int(am[2])
        print(f"    argmax=(b={b0} h={h0} s={am[1]} d={am[3]}) err={err[b0, am[1], h0, am[3]].item():.4f}", flush=True)
        # 重点：argmax 行的完整样本（out vs ref 前 8 列）+ 参考 bf16 舍入值
        s_am = int(am[1])
        oa = out[b0, s_am, h0, :8].float().cpu().tolist()
        ra = o_ref[b0, s_am, h0, :8].tolist()
        rb = o_ref[b0, s_am, h0, :8].to(torch.bfloat16).float().tolist()
        print(f"    s={s_am} out={['%.4f' % x for x in oa]}", flush=True)
        print(f"          ref={['%.4f' % x for x in ra]}", flush=True)
        print(f"          ref_bf16={['%.4f' % x for x in rb]}", flush=True)
        # 相邻行对照（s_am-1, s_am+1 的 out）
        for s2 in (max(0, s_am - 1), min(Sq - 1, s_am + 1)):
            oa2 = out[b0, s2, h0, :4].float().cpu().tolist()
            print(f"    s={s2} out={['%.4f' % x for x in oa2]}", flush=True)
        print(f"    bad s rows: {bad[:40]}{'...' if len(bad) > 40 else ''} ({len(bad)} rows)", flush=True)
    return ok


if __name__ == "__main__":
    dtype = torch.bfloat16
    q, k, v = make_inputs(dtype)
    o_ref, scores_raw, p_unorm, maxv, sumv, otmp = cpu_ref(q.cpu(), k.cpu(), v.cpu())
    print(f"ref O(b0,0:4,h0,:4) = {o_ref[0, :4, 0, :4].tolist()}")  # 参考值预览
    # 坏区中间量参考（b2 h0 s48-63 = t26/t27 实证坏行区，与 dump 探针 desc=111/212/223/224/321/412 对齐）
    print(f"ref S_raw(b0,h0,s48) = {scores_raw[0, 0, 48, :8].tolist()}")
    print(f"ref P_unorm(b0,h0,s63,:8) = {p_unorm[0, 63, 0, :8].tolist()}")
    print(f"ref max(b0,h0,s60:64) = {maxv[0, 60:64, 0].tolist()}")
    print(f"ref sum(b0,h0,s60:64) = {sumv[0, 60:64, 0].tolist()}")
    print(f"ref OTmp(b0,h0,s63,:4) = {otmp[0, 63, 0, :4].tolist()}")
    print(f"ref O(b0,h0,s63,:4) = {o_ref[0, 63, 0, :4].tolist()}")
    npass = 0
    for itr in range(1):      # 1MB 预算 = 单次 kernel 调用，迭代次数无关（聚焦版每轮 ≤64KB）
        npass += run_once(q, k, v, o_ref, itr)
    print(f"\n{npass}/1 PASS")

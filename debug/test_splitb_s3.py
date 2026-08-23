#!/usr/bin/env python3
# S3 golden 对比：SplitB 路径（FORCE_SPLITB=1）vs 纯 torch 参考 vs 旧路径
# 用法：conda activate FA2 && python debug/test_splitb_s3.py
import os
os.environ["FLASH_ATTN_FORCE_SPLITB"] = "1"   # 必须在 import 前设置

import math
import torch
import torch_npu
from flash_attn_npu import flash_attn_func
torch.npu.set_device(4)

def torch_ref(q, k, v, scale):
    """纯 fp32 参考（BSND [B,Sq,H,D]；GQA 按 q 头 h → kv 头 h//G 展开，与 kernel 映射一致）"""
    qf = q.float().transpose(1, 2)   # B,H,Sq,D
    kf = k.float().transpose(1, 2)   # B,Hkv,Sk,D
    vf = v.float().transpose(1, 2)
    g = qf.shape[1] // kf.shape[1]
    if g > 1:
        kf = kf.repeat_interleave(g, dim=1)   # [k0,k0,k1,k1..]：q 头 h 用 kv 头 h//g
        vf = vf.repeat_interleave(g, dim=1)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    p = torch.softmax(scores, dim=-1)
    o = torch.matmul(p, vf)
    return o.transpose(1, 2), scores  # B,Sq,H,D


def run_case(B, Sq, Sk, H, Hkv, D, dtype, tag):
    print('---------------------\n')
    print(f"[{tag}] B={B} Sq={Sq} Sk={Sk} H={H} Hkv={Hkv} D={D} {str(dtype).split('.')[-1]:8s}", flush=True)
    print('---------------------\n')
    torch.manual_seed(42 + B + Sq * 7 + H)
    q = (torch.randn(B, Sq, H, D) * 0.5).to(dtype).npu()
    k = (torch.randn(B, Sk, Hkv, D) * 0.5).to(dtype).npu()
    v = (torch.randn(B, Sk, Hkv, D) * 0.5).to(dtype).npu()
    scale = 1.0 / math.sqrt(D)

    # SplitB 路径（本环境变量强制触发）
    out_new = flash_attn_func(q, k, v, 0.0, scale, False)
    # 每用例同步：避免前用例 kernel 未跑完时下一用例已启动（两代 kernel 并发
    # 竞争 GM 总线 + ffts 全局区 → -O2 时序窗口重现，t19 全量序列前半 FAIL 而
    # 单用例 PASS 的根因判别实验，devlog #44.7）
    torch.npu.synchronize()

    # 参考（旧路径正确性已由 193 用例回归背书，不在此重复对比）
    o_ref, _ = torch_ref(q.cpu(), k.cpu(), v.cpu(), scale)

    d_new = (out_new.float().cpu() - o_ref).abs().max().item()
    ok = d_new < 2e-2
    print(f"[{tag}] B={B} Sq={Sq} Sk={Sk} H={H} Hkv={Hkv} D={D} {str(dtype).split('.')[-1]:8s} "
          f"| splitb_max_err={d_new:.4f} => {'PASS' if ok else 'FAIL'}", flush=True)
    if _os.environ.get("FLASH_ATTN_SPLITB_ERRMAP") == "1":
        err = (out_new.float().cpu() - o_ref).abs()          # [B,Sq,H,D]
        am = torch.unravel_index(err.argmax(), err.shape)
        b0, h0 = int(am[0]), int(am[2])
        pb = err.amax(dim=(1, 2, 3)).tolist()                # per-batch max err
        ph = err.amax(dim=(0, 1, 3)).tolist()                # per-head max err
        rows = err[b0, :, h0, :].amax(dim=-1)                # 最差 (b,h) 的逐 s 行误差
        bad = rows > 1e-2
        rowpat = ''.join('X' if v else '.' for v in bad.tolist())
        print(f"[errmap] argmax=(b={b0} h={h0} s={int(am[1])} d={int(am[3])}) err={err[b0, int(am[1]), h0, int(am[3])]:.4f}", flush=True)
        print(f"[errmap] per_batch: " + ' '.join(f"b{i}={v:.3f}" for i, v in enumerate(pb)), flush=True)
        print(f"[errmap] per_head:  " + ' '.join(f"h{i}={v:.3f}" for i, v in enumerate(ph)), flush=True)
        print(f"[errmap] rows(b={b0},h={h0}) bad={bad.sum().item()}/{Sq} pattern={rowpat}", flush=True)
        print(f"[errmap] rows halves: [{bad[:Sq//2].sum().item()}/{Sq//2}, {bad[Sq//2:].sum().item()}/{Sq - Sq//2}] (AIV0/AIV1)", flush=True)
    return ok


if __name__ == "__main__":
    # 用例过滤：FLASH_ATTN_SPLITB_CASES="1,64,64,8,8,128" （B,Sq,Sk,H,Hkv,D 精确匹配，逗号分隔多个）
    import os as _os
    _flt = _os.environ.get("FLASH_ATTN_SPLITB_CASES")
    cases = [
        # S3 目标配置：NO_MASK / fp16 / D=128 / MHA（先小后大）
        (4,   64,  64,  8, 8, 128, torch.float16, "S3-core"),
        (1,   64,  64,  8, 8, 128, torch.float16, "S3-b1"),
        (8, 64,  64,  8, 8, 128, torch.float16, "S3-b8"),
        (12, 64,  64,  8, 8, 128, torch.float16, "S3-b12"),
        (20, 64,  64,  8, 8, 128, torch.float16, "S3-b20"),
        (128, 64,  64,  8, 8, 128, torch.float16, "S3-b128"),
        (1024, 64, 64,  8, 8, 128, torch.float16, "S3-b1024"),
        # s1Base 尾块 / 非 16 对齐 Sq / s1Outer>1
        (4,   48,  64,  8, 8, 128, torch.float16, "S3-tail48"),
        (4,   200, 64,  8, 8, 128, torch.float16, "S3-sq200(2 s1blk)"),
        (4,   64,  32,  8, 8, 128, torch.float16, "S3-sk32"),
        # GQA
        (4,   64,  64,  8, 4, 128, torch.float16, "S3-gqa2"),
        (4,   64,  64,  8, 1, 128, torch.float16, "S3-gqa8"),
        # D=64
        (4,   64,  64,  8, 8,  64, torch.float16, "S3-d64"),
        # bf16
        (4,   64,  64,  8, 8, 128, torch.bfloat16, "S3-bf16"),
    ]
    if _flt:
        keys = {tuple(int(x) for x in k.split(",")) for k in _flt.split(";")}
        cases = [c for c in cases if tuple(int(x) for x in c[:6]) in keys]
    print('==============================')
    print('Cases to run:')
    for c in cases:
        print(f'  {c[7]}: {c[0]}x{c[1]}x{c[2]}x{c[3]}x{c[4]}x{c[5]}')
    print('==============================')
    npass = sum(run_case(*c) for c in cases)
    print(f"\n{'='*60}\n{npass}/{len(cases)} PASS")

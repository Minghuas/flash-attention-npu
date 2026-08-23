#!/usr/bin/env python3
# S3 判刀样例（devlog #44.7）：Q=K 随机 + V 全 1 → 正确 O 处处 = 1.0（softmax 行和恒 1）。
# 任何坏行直接显现；坏行数值可反推破坏源：
#   P 坏（读旧 S/P）→ O = ΣP̃ ≠ 1；stats sum 坏 → O = OTmp/sum_wrong ≠ 1；
#   OTmp 坏 → O ≠ 1。配合 DumpTensor 聚焦坏区 s16-31（探针 desc=111/211/221/222/311/411）。
# bf16 单核撞率约 1/5，循环 10 次高概率抓到 FAIL。
# 用法：conda activate FA2 && FLASH_ATTN_SPLITB_DUMP=1 python debug/test_splitb_s3_stage2.py
import os
os.environ["FLASH_ATTN_FORCE_SPLITB"] = "1"

import torch
import torch_npu
from flash_attn_npu import flash_attn_func
torch.npu.set_device(1)


def run_once(B, Sq, Sk, H, D, dtype, seed, itr):
    torch.manual_seed(seed)
    q = (torch.randn(B, Sq, H, D) * 0.5).to(dtype).npu()
    k = (torch.randn(B, Sk, H, D) * 0.5).to(dtype).npu()
    v = torch.ones(B, Sk, H, D).to(dtype).npu()
    scale = 1.0 / (D ** 0.5)
    out = flash_attn_func(q, k, v, 0.0, scale, False)
    torch.npu.synchronize()
    err = (out.float().cpu() - 1.0).abs()                # 期望全 1
    maxerr = err.max().item()
    ok = maxerr < 2e-2
    print(f"[itr{itr}] max_err={maxerr:.4f} => {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        # 坏区定位：per-s 行最大误差（[B,Sq,H,D] → amax dims (0,2,3)）
        rows = err.amax(dim=(0, 2, 3))
        bad = (rows > 1e-2).nonzero().flatten().tolist()
        print(f"    bad s rows: {bad[:40]}{'...' if len(bad) > 40 else ''} ({len(bad)} rows)", flush=True)
        per_head = err.amax(dim=(0, 1, 3)).tolist()
        print(f"    per_head: " + ' '.join(f"h{i}={v:.3f}" for i, v in enumerate(per_head)), flush=True)
    return ok


if __name__ == "__main__":
    B, Sq, Sk, H, D = 4, 64, 64, 8, 128
    dtype = torch.bfloat16
    seed = 42 + B + Sq * 7 + H                      # 与 t16 core 同种子（固定输入，纯时序判别）
    npass = 0
    for itr in range(10):
        npass += run_once(B, Sq, Sk, H, D, dtype, seed, itr)
    print(f"\n{npass}/10 PASS")

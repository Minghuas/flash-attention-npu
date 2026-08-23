#!/usr/bin/env python3
# S3 分阶段判别实验 1：全 1 / 结构化输入，反推 h7（最后 tile）损坏公式。
#
# 背景（devlog #44）：-O2 下 h7 误差 ≈2-3 且 b1 确定性复现。全 1 输入下每阶段期望值
# 可手算，h7 输出值唯一标识损坏模式：
#   O = 1.0        → 正确
#   O = 64.0       → P≡1 且 divisor≈1（SM 读到 S=0，且 stats sum 错）
#   O = 1.0/64     → P≡1 但 divisor=64（stats 正确）→ 其实 = 1/64×ΣV = 1（V 全 1 时）⚠
#   O = 11.3137    → P=rawS 未归一（÷64）
#   O = 724.08     → P=rawS 且 divisor=1
#   O = 2.0 / 0.5  → divisor 减半/翻倍
# 用 V 全 1（ΣV=64）与 V=s+1（ΣV=2080）双配置消除歧义。
#
# 用法：conda activate FA2 && python debug/test_splitb_s3_stage.py
import os
os.environ["FLASH_ATTN_FORCE_SPLITB"] = "1"   # 必须在 import 前设置

import math
import torch
import torch_npu
from flash_attn_npu import flash_attn_func
torch.npu.set_device(1)


def run_stage_case(tag, q, k, v, expect_desc):
    """expect_desc: dict h->期望输出张量 [Sq,D] 或标量（每 head 期望值）"""
    B, Sq, H, D = q.shape
    scale = 1.0 / math.sqrt(D)
    out = flash_attn_func(q, k, v, 0.0, scale, False)
    torch.npu.synchronize()
    oc = out.float().cpu()
    print(f"\n[{tag}] B={B} Sq={Sq} H={H} D={D}")
    for h in range(H):
        oh = oc[0, :, h, :]                       # [Sq,D]
        exp = expect_desc[h]
        if isinstance(exp, torch.Tensor):
            err = (oh - exp).abs().max().item()
            refv = exp.abs().mean().item()
        else:
            err = (oh - exp).abs().max().item()
            refv = abs(exp)
        ok = err < 2e-2
        print(f"  h{h}: mean={oh.mean().item():+.4f} min={oh.min().item():+.4f} "
              f"max={oh.max().item():+.4f} | err={err:.4f} (期望量级 {refv:.4f}) {'OK' if ok else '  <== 偏离'}")
    return oc


if __name__ == "__main__":
    torch.manual_seed(0)

    # ============ 配置 A：Q=K=V 全 1 ============
    # 期望链条（scale 在 SM 段 ScaleS 才乘，QK 段 S 是原始内积）：
    #   S = Σ_d 1×1 = 128；ScaleS → 128/√128 = 11.3137；max=11.3137
    #   P（fp16 未归一 exp） = exp(11.3137−11.3137) = 1.0；sum = 64
    #   OTmp = Σ P·V = 64；O = 64/64 = 1.0
    B, Sq, Sk, H, D = 1, 64, 64, 8, 128
    q1 = torch.ones(B, Sq, H, D).half().npu()
    k1 = torch.ones(B, Sk, H, D).half().npu()
    v1 = torch.ones(B, Sk, H, D).half().npu()
    run_stage_case("A: Q=K=V=1", q1, k1, v1, {h: 1.0 for h in range(H)})

    # # ============ 配置 B：Q=K 全 1，V=s+1 ============
    # # ΣV = Σ(s+1) = 2080（每 d 相同）；正确 O = 2080/64 = 32.5；
    # # 损坏 P≡1&div≈1 → O = 2080；P=rawS&div=1 → O = 11.3137×2080
    # v2 = torch.arange(1, Sk + 1, dtype=torch.float32).view(1, Sk, 1, 1).expand(B, Sk, H, D).half().npu()
    # run_stage_case("B: Q=K=1, V=s+1", q1, k1, v2, {h: 32.5 for h in range(H)})

    # # ============ 配置 C：Q=K 全 1，V 随机（ΣV/64 参考） ============
    # torch.manual_seed(42 + B + Sq * 7 + H)   # 与 t16 core 同种子
    # v3 = (torch.randn(B, Sk, H, D) * 0.5).half().npu()
    # sv = v3.float().cpu().sum(dim=1)          # [B,H,D]
    # exp_c = (sv[:, None, :, :] / Sk).expand(B, Sq, H, D).contiguous()
    # run_stage_case("C: Q=K=1, V=rand", q1, k1, v3, {h: exp_c[0, :, h, :] for h in range(H)})

    # # ============ 配置 D：Q=K 随机（t16 同款），V 全 1 ============
    # # 正确 O = P·1 = 1（softmax 行和=1）；损坏 P≡1&div≈1 → O=64
    # torch.manual_seed(42 + B + Sq * 7 + H)
    # q4 = (torch.randn(B, Sq, H, D) * 0.5).half().npu()
    # k4 = (torch.randn(B, Sk, H, D) * 0.5).half().npu()
    # run_stage_case("D: Q=K=rand, V=1", q4, k4, v1, {h: 1.0 for h in range(H)})

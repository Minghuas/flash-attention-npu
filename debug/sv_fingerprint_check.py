#!/usr/bin/env python3
"""纯 CPU 复现 t16 输入，检验 h7 错误指纹假设（P≡1? divisor≈1?）。

背景（devlog #44）：-O2 下 S3 的 h7（最后 tile）错误 ≈ 2-3，指纹候选：
  - cand1 O=ΣV      : P≡1 且 divisor≈1（SM 读到 S=0 或 PV 读到 P=1，且 stats 也错）
  - cand2 O=ΣV/64   : P≡1 但 divisor=64（stats 正确计算 exp(0-0)=64）
  - cand3 正确 O
  - cand4 O=ΣrawS·V : P 被替换为未归一化的原始 S（fp16 视图错位等）
比较各候选与观测误差，确定 h7 的输出到底等于哪个公式。

用法：conda activate FA2 && python debug/sv_fingerprint_check.py
"""
import torch, math

observed = {"core": 2.0401, "b1": 2.9198, "sk32": 1.5614, "bf16": 2.0342}


def gen(B, Sq, Sk, H, Hkv, D, dtype):
    torch.manual_seed(42 + B + Sq * 7 + H)          # 与 test_splitb_s3.py 完全一致
    q = (torch.randn(B, Sq, H, D) * 0.5).to(dtype)
    k = (torch.randn(B, Sk, Hkv, D) * 0.5).to(dtype)
    v = (torch.randn(B, Sk, Hkv, D) * 0.5).to(dtype)
    return q, k, v


def main():
    cases = {
        "core": (4, 64, 64, 8, 8, 128, torch.float16, observed["core"]),
        "b1":   (1, 64, 64, 8, 8, 128, torch.float16, observed["b1"]),
        "sk32": (4, 64, 32, 8, 8, 128, torch.float16, observed["sk32"]),
        "bf16": (4, 64, 64, 8, 8, 128, torch.bfloat16, observed["bf16"]),
    }
    for tag, (B, Sq, Sk, H, Hkv, D, dt, obs) in cases.items():
        q, k, v = gen(B, Sq, Sk, H, Hkv, D, dt)
        vf = v.float()
        sv = vf.sum(dim=1)                           # [B,Hkv,D] = Σ_kv V
        scale = 1.0 / math.sqrt(D)
        # 照 torch_ref：q/k/v 先转 BHND 布局再算（test_splitb_s3.py torch_ref 同款）
        qf = q.float().transpose(1, 2)               # [B,H,Sq,D]
        kf = k.float().transpose(1, 2)               # [B,Hkv,Sk,D]
        vf_bh = vf.transpose(1, 2)                   # [B,Hkv,Sk,D]
        scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale   # [B,H,Sq,Sk]
        p = torch.softmax(scores, dim=-1)
        o_ref = torch.matmul(p, vf_bh).transpose(1, 2)            # [B,Sq,H,D]

        cands = {
            "ΣV":    sv[:, None, 7, :].expand(B, Sq, D),
            "ΣV/64": sv[:, None, 7, :].expand(B, Sq, D) / Sk,
            "ΣrawS·V": torch.einsum('bsj,bjd->bsd',
                                    scores[:, 7, :, :], vf_bh[:, 7, :, :]),
            "correct": o_ref[:, :, 7, :],
        }
        print(f"[{tag}] B={B} Sk={Sk} {str(dt).split('.')[-1]}  obs_h7_err={obs:.4f}")
        for name, c in cands.items():
            e = (c - o_ref[:, :, 7, :]).abs().max().item()
            mark = "  <== 匹配" if abs(e - obs) < 0.5 else ""
            print(f"    |{name:10s} − ref_7|max = {e:.4f}{mark}")
        # 决定性检验 1：假设 O_actual = α·ΣrawS·V + (1−α)·ref，拟合 α
        rsv = cands["ΣrawS·V"]
        ref7 = o_ref[:, :, 7, :]
        # 每元素最小二乘：err(α) = |(α·rsv + (1−α)·ref7) − obs_actual|，
        # 无 obs_actual 具体值，改用一致性检验 2
        # 决定性检验 2：0.5 混合的误差 = 0.5·|rsv−ref7|，其 max 与 argmax 位置
        e_half = 0.5 * (rsv - ref7).abs()
        am = torch.unravel_index(e_half.argmax(), e_half.shape)
        print(f"    [0.5mix] max_err={e_half.max().item():.4f} argmax=(b={am[0]} s={am[1]} d={am[2]})  "
              f"(obs argmax 见 t16 日志: core=(b=1 s=36 d=86), b1=(b=0 s=18 d=35), sk32=(b=0 s=58 d=15))")
        print()


if __name__ == "__main__":
    main()

# ---- 追加：观测 argmax 位置处各候选的误差值（决定性） ----
def check_obs_positions():
    obs_am = {"core": (1, 36, 86), "b1": (0, 18, 35), "sk32": (0, 58, 15), "bf16": (1, 36, 86)}
    cases = {
        "core": (4, 64, 64, 8, 8, 128, torch.float16),
        "b1":   (1, 64, 64, 8, 8, 128, torch.float16),
        "sk32": (4, 64, 32, 8, 8, 128, torch.float16),
        "bf16": (4, 64, 64, 8, 8, 128, torch.bfloat16),
    }
    for tag, (B, Sq, Sk, H, Hkv, D, dt) in cases.items():
        q, k, v = gen(B, Sq, Sk, H, Hkv, D, dt)
        scale = 1.0 / math.sqrt(D)
        qf = q.float().transpose(1, 2)
        kf = k.float().transpose(1, 2)
        vf = v.float().transpose(1, 2)
        scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale   # [B,H,Sq,Sk]
        p = torch.softmax(scores, dim=-1)
        o_ref = torch.matmul(p, vf).transpose(1, 2)               # [B,Sq,H,D]
        rsv = torch.einsum('bsj,bjd->bsd', scores[:, 7, :, :], vf[:, 7, :, :])
        ref7 = o_ref[:, :, 7, :]
        sv = v.sum(dim=1).float()
        b, s, d = obs_am[tag]
        for name, c in {"ΣV": sv[b, 7, d], "ΣV/64": sv[b, 7, d]/Sk,
                        "ΣrawS·V": rsv[b, s, d], "ref": ref7[b, s, d]}.items():
            print(f"  {tag:5s} @obs({b},{s},{d}) {name:9s} = {c:+.4f}   (|c−ref|={abs(c - ref7[b,s,d].item()):.4f})")
        e_half = 0.5 * (rsv - ref7).abs()
        print(f"  {tag:5s} @obs 0.5mix err = {e_half[b,s,d].item():.4f}   "
              f"obs_err={observed[tag]:.4f}   比值={observed[tag]/e_half[b,s,d].item():.4f}")
        print()

check_obs_positions()

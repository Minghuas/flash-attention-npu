#!/usr/bin/env python3
"""解析判刀 W 的 DumpTensor 日志并与 CPU 参考逐段逐行对比。

用法：
  conda activate FA2 && python debug/parse_dump_compare.py debug/log/t28_judgeW_probe.log

对比对象（与 t28 探针 desc 对齐，聚焦 b1/h0 的 s48-63 坏行区（t28b 实证错误在 b1））：
  desc=111  S      (16x64 float32)  vs ref S_raw
  desc=212  P      (16x64 bf16)     vs ref P_unorm（bf16 舍入后）
  desc=223  max    (16 float32)     vs ref max
  desc=224  sum    (16 float32)     vs ref sum
  desc=321  OTmp   (16x128 float32) vs ref OTmp
  desc=412  O      (16x1024 bf16, 取每行前 128 = head0) vs ref O（bf16 舍入后）
输出每段前 4 行明细 + 出错行清单（|dump-ref| > 阈值）。
"""
import re
import sys
import torch

# ---- 与 test_splitb_s3_stage3.py 相同的结构化输入（非随机） ----
B, Sq, Sk, H, D = 2, 64, 64, 8, 128    # B=2 最小复现形态
SCALE = 1.0 / (D ** 0.5)


def make_ref():
    """CPU 参考中间量（与 kernel 各段产物一一对应）"""
    q = torch.zeros(B, Sq, H, D)
    for s in range(Sq):
        for h in range(H):
            q[:, s, h, :] = (h + 1) * (s + 1) / 512.0
    k = torch.zeros(B, Sk, H, D)
    for j in range(Sk):
        k[:, j, :, :] = (j + 1) / 64.0
    v = torch.zeros(B, Sk, H, D)
    for j in range(Sk):
        v[:, j, :, :] = j + 1
    qf = q.transpose(1, 2)
    kf = k.transpose(1, 2)
    vf = v.transpose(1, 2)
    scores_raw = torch.matmul(qf, kf.transpose(-1, -2))          # kernel gS（scale 前）
    scores = scores_raw * SCALE
    maxv = scores.max(dim=-1, keepdim=True).values
    p_unorm = torch.exp(scores - maxv).to(torch.bfloat16).float()  # kernel P（bf16 量化模拟）
    sumv = p_unorm.sum(dim=-1)
    otmp = torch.matmul(p_unorm, vf)                              # kernel OTmp（未除 sum）
    o = torch.matmul(p_unorm / sumv.unsqueeze(-1), vf)            # kernel O
    # O 的 BSND 展平视图：o[b1, s, h, d] → [64, 1024]（行主序 s×strideO 布局）
    o_flat = o.transpose(1, 2)[1].contiguous().reshape(64, -1).to(torch.bfloat16).float()
    return {
        "S": scores_raw.transpose(1, 2)[1, 48:64, 0, :],          # [16,64] b1 h0 s48-63
        "S_h": scores_raw.transpose(1, 2)[1, 48:64, :, :],        # [16,8,64] 按 head
        "P": p_unorm.transpose(1, 2)[1, 48:64, 0, :].to(torch.bfloat16).float(),
        "max": maxv.transpose(1, 2).squeeze(-1)[1, 48:64, 0],
        "sum": sumv.transpose(1, 2)[1, 48:64, 0],
        "sum_h": sumv.transpose(1, 2)[1, 48:64, :],               # [16,8] 按 head
        "sum_aiv1": sumv.transpose(1, 2)[1, 32:64, 0],            # AIV1 半区 32 行
        "OTmp": otmp.transpose(1, 2)[1, 32:64, 0, :],             # [32,128] AIV1 半区
        "OTmp_h": otmp.transpose(1, 2)[1, 60:64, :, :],           # [4,8,128] 按 head s60-63
        "O": o_flat,                                              # [64,1024] 全宽（8 头）
    }


def parse_dump(path):
    """提取每个 desc 的数值数组"""
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    dumps = {}
    cur = None
    for ln in lines:
        m = re.search(r"DumpTensor: desc=(\d+),.*dump_size=(\d+)", ln)
        if m:
            cur = (int(m.group(1)), int(m.group(2)))
            dumps.setdefault(cur[0], []).append([])
            continue
        if cur is not None and (ln.startswith("[AIV") or ln.startswith("[AIC") or "DumpTensor" in ln):
            cur = None
            continue
        if cur is not None:
            dumps[cur[0]][-1] += [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", ln)]
    return dumps


def compare_section(name, desc, dump_vals, ref, tol, shape, row_base=48):
    print(f"\n=== desc={desc} {name} ===")
    if desc not in dump_vals:
        print("  (无 dump 数据)")
        return
    vals = dump_vals[desc][0]
    n_elems = shape[0] * (shape[1] if len(shape) > 1 else 1)
    if len(vals) < n_elems:
        print(f"  (数据不足: {len(vals)} < {n_elems})")
        return
    a = torch.tensor(vals[:n_elems]).reshape(shape)
    r = ref
    is1d = len(shape) == 1 or shape[1] == 1
    if is1d:
        a2 = a.reshape(-1, 1)
        r2 = r.reshape(-1, 1)
    else:
        a2, r2 = a, r
    diff = (a2 - r2).abs()
    bad_rows = (diff.amax(dim=-1) > tol).nonzero().flatten().tolist()
    print(f"  行数={shape[0]} 阈值={tol}  出错行: {bad_rows if bad_rows else '无'}")
    for i in range(min(4, shape[0])):
        row = row_base + i
        d = diff[i].max().item()
        mark = "  <== BAD" if d > tol else ""
        if is1d:
            print(f"  s{row}: dump={a2[i, 0].item():.4f}  ref={r2[i, 0].item():.4f}  diff={d:.4f}{mark}")
        else:
            n = min(8, shape[1])
            av = [f"{x:.3f}" for x in a[i, :n].tolist()]
            rv = [f"{x:.3f}" for x in r[i, :n].tolist()]
            print(f"  s{row}: dump={av}")
            print(f"        ref ={rv}  maxdiff={d:.4f}{mark}")
    if bad_rows:
        for i in bad_rows[:6]:
            dmax = diff[i].max().item()
            amax = diff[i].argmax().item()
            print(f"  BAD s{row_base+i}: maxdiff={dmax:.4f} @col{amax} dump={a2[i, amax].item():.4f} ref={r2[i, amax].item():.4f}")


def compare_o_heatmap(name, descs, dumps, ref_flat, tol, n_rows, row_base):
    """O 的 64 行 × 8 头坏点图：每行每头的 maxdiff，坏点高亮"""
    for desc in descs:
        if desc not in dumps:
            continue
        vals = dumps[desc][0]
        n_elems = n_rows * 1024
        if len(vals) < n_elems:
            print(f"  (desc={desc} 数据不足: {len(vals)} < {n_elems})")
            continue
        a = torch.tensor(vals[:n_elems]).reshape(n_rows, 1024)
        r = ref_flat[row_base - 48: row_base - 48 + n_rows] if False else ref_flat
        r = ref_flat[0: n_rows] if row_base == 0 else ref_flat[32: 32 + n_rows]
        diff = (a - r).abs().reshape(n_rows, 8, 128).amax(dim=-1)   # [rows, 8 heads]
        bad = (diff > tol).nonzero()
        print(f"=== desc={desc} {name} ({n_rows} 行 × 8 头, 阈值={tol}) ===")
        print(f"  坏点总数: {bad.shape[0]}")
        for r_i, h_i in bad[:24].tolist():
            s = row_base + r_i
            print(f"  BAD s{s} h{h_i}: maxdiff={diff[r_i, h_i].item():.4f}")
        if bad.shape[0] == 0:
            print("  全对")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "debug/log/t28_judgeW_probe.log"
    ref = make_ref()
    dumps = parse_dump(path)
    print(f"解析 {path}：desc 集合 = {sorted(dumps.keys())}")
    # 全 batch × tile0/2/3 覆盖（数据 batch 间相同，ref 复用 b1 的按-head 参考）
    for bo in range(2):
        for ti in (0, 2, 3):
            compare_section(f"S b{bo} tile{ti} (QK 原始内积)", 111 + bo * 10 + ti,
                            dumps, ref["S_h"][:, ti, :], 1e-2, (16, 64))
    # sum 覆盖全部 8 tile × 2 batch（按 head 参考）
    for bo in range(2):
        for ti in range(8):
            compare_section(f"stats sum b{bo} tile{ti} (AIV1 s48-63)", 224 + 10 * bo + ti,
                            dumps, ref["sum_h"][:, ti], 1e-1, (16, 1))
    # 段4 前（divout 读前）的 stats 对照：与 224 同地址同 ref——若此处坏 → 段间被破坏
    for bo in range(2):
        for ti in range(8):
            compare_section(f"stats sum S4前 b{bo} tile{ti}", 424 + 10 * bo + ti,
                            dumps, ref["sum_h"][:, ti], 1e-1, (16, 1))
    for bo in range(2):
        for ti in (0, 2, 3):
            compare_section(f"OTmp b{bo} tile{ti} (s60-63)", 321 + bo * 10 + ti,
                            dumps, ref["OTmp_h"][:, ti, :], 0.01, (4, 128), row_base=60)
    compare_o_heatmap("O AIV0 (s0-31)", [411], dumps, ref["O"], 2e-2, 32, 0)
    compare_o_heatmap("O AIV1 (s32-63)", [412], dumps, ref["O"], 2e-2, 32, 32)
    # ---- divout 内部三点链（devlog #44.11）----
    # 参考：全行 sum 与 O（fp32 与 bf16）
    import torch as _t
    _q = _t.zeros(B, Sq, H, D)
    for s in range(Sq):
        for h in range(H):
            _q[:, s, h, :] = (h + 1) * (s + 1) / 512.0
    _k = _t.zeros(B, Sk, H, D)
    for j in range(Sk):
        _k[:, j, :, :] = (j + 1) / 64.0
    _v = _t.arange(1, Sk + 1, dtype=_t.float32).view(1, Sk, 1, 1).expand(B, Sk, H, D)
    _sc = _t.matmul(_q.transpose(1, 2), _k.transpose(1, 2).transpose(-1, -2)) * SCALE
    _pu = _t.exp(_sc - _sc.max(dim=-1, keepdim=True).values).to(_t.bfloat16).float()  # bf16 P 模拟
    sum_full = _pu.sum(-1).transpose(1, 2)[1]                          # [64,8]
    _o_fp = _t.matmul(_pu, _v.transpose(1, 2)).transpose(1, 2) / \
        _pu.sum(-1).transpose(1, 2).unsqueeze(-1)                      # [B,Sq,H,D]

    # 点1：glUb = divout 读到的 sum（每条 = 一次 LoadStats 调用；前 8 条 = b0 的 tile0-7）
    for desc, aiv in ((521, 0), (522, 1)):
        if desc in dumps:
            for ti in range(min(8, len(dumps[desc]))):
                vals = dumps[desc][ti]
                a = _t.tensor(vals[:32])
                r = sum_full[32 * aiv: 32 * (aiv + 1), ti]
                diff = (a - r).abs()
                bad = (diff > 1e-2).nonzero().flatten().tolist()
                print(f"=== desc={desc} glUb AIV{aiv} b0 tile{ti} (divout 读到的 sum) ===")
                print(f"  出错行: {bad if bad else '无'}")
                if bad:
                    for i in bad[:4]:
                        s = 32 * aiv + i
                        print(f"  s{s}: dump={a[i].item():.4f} ref={r[i].item():.4f} diff={diff[i].item():.4f}")

    # 点2/3：Div 后 goUb32 / Cast 后 goUb16（前 8 条 = b0 tile0-7）
    for desc, aiv in ((531, 0), (532, 1), (541, 0), (542, 1)):
        if desc in dumps:
            for ti in range(min(8, len(dumps[desc]))):
                vals = dumps[desc][ti]
                a = _t.tensor(vals[:32])
                s_row = 32 * aiv
                is_bf16 = desc >= 540
                r = _o_fp[1, s_row, ti, :32]
                if is_bf16:
                    r = r.to(_t.bfloat16).float()
                diff = (a - r).abs()
                tol = 1e-2
                bad = (diff > tol).nonzero().flatten().tolist()
                name = "Cast后 goUb16" if is_bf16 else "Div后 goUb32"
                print(f"=== desc={desc} {name} AIV{aiv} b0 tile{ti} s={s_row} 前32列 ===")
                print(f"  出错列: {bad if bad else '无'}")
                if bad:
                    for i in bad[:4]:
                        print(f"  c{i}: dump={a[i].item():.4f} ref={r[i].item():.4f} diff={diff[i].item():.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""从 test_splitb_s3.py 的长日志中提取每用例 PASS/FAIL 结果摘要。

用法:
  python debug/extract_results.py LOG [LOG2 ...]
  --diff       跨日志按 tag 对比 (verdict 翻转 / err 漂移 -> 时序敏感信号)
  --ctx N      每个结果行后附 N 行原始上下文 (用于看 probe 标记)
  --tags A,B   (与 --ctx 合用) 只对指定 tag 打印上下文
"""
import argparse
import re
import sys

RESULT_RE = re.compile(
    r'^\[(?P<tag>[^\]]+)\].*\| splitb_max_err=(?P<err>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) => (?P<v>PASS|FAIL)'
)


def parse(path):
    lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
    results = []
    for i, ln in enumerate(lines):
        m = RESULT_RE.match(ln)
        if m:
            results.append((i + 1, m.group('tag'), float(m.group('err')), m.group('v')))
    return lines, results


def fmt_row(tag, err, v, err_w=12):
    return f"  {tag:<22} err={err:>{err_w}}  {v}"


def dump_log(path, ctx=0, only_tags=None):
    lines, results = parse(path)
    print(f"=== {path} ({len(results)} cases) ===")
    for lineno, tag, err, v in results:
        print(fmt_row(tag, err, v))
        if ctx > 0 and (only_tags is None or tag in only_tags):
            for c in lines[lineno:lineno + ctx]:
                print(f"      | {c}")
    npass = sum(1 for r in results if r[3] == 'PASS')
    nfail = len(results) - npass
    print(f"--- {npass}/{len(results)} PASS, {nfail} FAIL ---")
    if npass:
        print("PASS:", ', '.join(r[1] for r in results if r[3] == 'PASS'))
    if nfail:
        print("FAIL:", ', '.join(r[1] for r in results if r[3] == 'FAIL'))
    print()
    return results


def diff_logs(paths):
    parsed = [(p, parse(p)[1]) for p in paths]
    tags = {}
    for p, rs in parsed:
        for _, tag, err, v in rs:
            tags.setdefault(tag, []).append((p, err, v))
    print(f"=== verdict diff: {' vs '.join(paths)} ===")
    for tag, recs in sorted(tags.items()):
        base = recs[0]
        cells = [f"{p.split('/')[-1]}:{v}({err:.4f})" for p, err, v in recs]
        flip = len({v for _, _, v in recs}) > 1
        drift = max(abs(recs[i][1] - recs[0][1]) for i in range(1, len(recs))) > 1e-6
        mark = '  <-- FLIP' if flip else ('  <-- err drift' if drift else '')
        print(f"  {tag:<22} {'  '.join(cells)}{mark}")
    print()


def main():
    ap = argparse.ArgumentParser(description='Extract PASS/FAIL summary from splitb test logs')
    ap.add_argument('logs', nargs='+', help='log file paths')
    ap.add_argument('--diff', action='store_true', help='cross-log verdict comparison')
    ap.add_argument('--ctx', type=int, default=0, help='print N raw lines after each result line')
    ap.add_argument('--tags', default=None, help='comma list of tags to show context for')
    args = ap.parse_args()

    all_results = []
    for p in args.logs:
        all_results.append(dump_log(p, ctx=args.ctx, only_tags=set(args.tags.split(',')) if args.tags else None))
    if args.diff and len(all_results) >= 2:
        diff_logs(args.logs)

    ok = all(r[3] == 'PASS' for rs in all_results for r in rs)
    if ok:
        print('ALL PASS')
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()

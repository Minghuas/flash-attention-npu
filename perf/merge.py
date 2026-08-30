#!/usr/bin/env python3
"""
合并两份性能数据，按 batch+seqlen 对齐，并重新计算加速比。
speedup(1) = base_ms(1) / fa_ms(1)
speedup(2) = base_ms(1) / fa_ms(2)
speedup     = fa_ms(2) / fa_ms(1)   # 方案1相对方案2的加速比
输出列：batch, seqlen, fa_ms(1), fa_ms(2), base_ms(1), speedup(1), speedup(2), speedup

用法：
    python merge_bench.py result1.csv result2.csv -o merged.csv
"""

import argparse
import pandas as pd

def detect_sep(filepath):
    """自动检测分隔符：逗号、制表符或空白"""
    with open(filepath, 'r') as f:
        first = f.readline()
        if ',' in first:
            return ','
        elif '\t' in first:
            return '\t'
        elif '  ' in first or '\t' in first:
            return '\s+'
        else:
            return ','   # 默认

def main():
    parser = argparse.ArgumentParser(description="合并两份benchmark数据并重新计算speedup")
    parser.add_argument('--file1', help='第一份数据文件', default='./perf/data/bench3.csv')
    parser.add_argument('--file2', help='第二份数据文件', default='./perf/data/bench2.csv')
    parser.add_argument('-o', '--output', default='./perf/data/merged2.csv', help='输出CSV路径')
    parser.add_argument('--sep1', help='指定文件1分隔符，默认自动检测')
    parser.add_argument('--sep2', help='指定文件2分隔符，默认自动检测')
    args = parser.parse_args()

    sep1 = args.sep1 if args.sep1 else detect_sep(args.file1)
    sep2 = args.sep2 if args.sep2 else detect_sep(args.file2)

    df1 = pd.read_csv(args.file1, sep=sep1, engine='python', skipinitialspace=True)
    df2 = pd.read_csv(args.file2, sep=sep2, engine='python', skipinitialspace=True)

    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    # 合并（外连接）
    merged = pd.merge(df1, df2, on=['batch', 'seqlen'], how='outer', suffixes=('(1)', '(2)'))

    # 重新计算加速比
    # 注意：分母可能为0或缺失，需处理
    merged['speedup(1)'] = merged['base_ms(1)'] / merged['fa_ms(1)']
    merged['speedup(2)'] = merged['base_ms(1)'] / merged['fa_ms(2)']

    # 格式化为两位小数加 'x'（可选，也可保留数值）
    merged['speedup(1)'] = merged['speedup(1)'].map(lambda x: f'{x:.2f}x' if pd.notnull(x) else None)
    merged['speedup(2)'] = merged['speedup(2)'].map(lambda x: f'{x:.2f}x' if pd.notnull(x) else None)

    # 方案1相对方案2的加速比
    merged['speedup'] = merged['fa_ms(2)'] / merged['fa_ms(1)']
    merged['speedup'] = merged['speedup'].map(lambda x: f'{x:.2f}x' if pd.notnull(x) else None)

    # 选择输出列
    out_cols = ['batch', 'seqlen', 'fa_ms(1)', 'fa_ms(2)', 'base_ms(1)', 'speedup(1)', 'speedup(2)', 'speedup']
    for col in out_cols:
        if col not in merged.columns:
            merged[col] = None

    merged = merged.sort_values(['batch', 'seqlen']).reset_index(drop=True)

    # 输出：各列按最大宽度对齐，便于直接阅读
    out_df = merged[out_cols].astype(str).replace({'nan': '', 'None': ''})
    widths = {c: max(len(str(c)), out_df[c].str.len().max()) for c in out_cols}
    with open(args.output, 'w') as f:
        f.write(', '.join(str(c).ljust(widths[c]) for c in out_cols) + '\n')
        for _, row in out_df.iterrows():
            f.write(', '.join(row[c].ljust(widths[c]) for c in out_cols) + '\n')
    print(f"合并完成，结果保存至 {args.output}")

if __name__ == '__main__':
    main()
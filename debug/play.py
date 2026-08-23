
def build(first, N=10, slope=2):
    vec = [0] * N
    # part1: first < 0
    idx = 0
    while idx < -first and idx < N:
        vec[idx] = -first - idx
        idx += 1
    # part2: first >= 0
    while idx < N:
        vec[idx] = idx + first
        idx += 1
    
    vec = [-slope * v for v in vec]
    return vec

def build2(first, N=10, slope=2):
    # 在循环内直接累加slope，不用再乘slope
    vec = [0] * N
    idx = 0
    value = slope * first
    while idx < -first and idx < N:      # 改 <= 为 <
        vec[idx] = value
        value += slope
        idx += 1
    value = -slope * (idx + first)         # 1 次乘法，不进循环，左臂执行时算出 0（无开销感）
    while idx < N:
        vec[idx] = value
        value -= slope
        idx += 1
    return vec

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--first', type=int, default=0)
    parser.add_argument('--N', type=int, default=512)
    parser.add_argument('--slope', type=float, default=1)
    args = parser.parse_args()
    vec = build(args.first, args.N, args.slope)
    vec2 = build2(args.first, args.N, args.slope)
    print(vec)
    print(vec2)
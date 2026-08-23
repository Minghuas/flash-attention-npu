"""
请实现一个用于调试 FlashAttention v2 forward kernel 的 Q/K debug tensor
生成器。该生成器的目标不是进行模型推理，而是构造特殊输入，使得
kernel 内部 QK GEMM 计算得到的 Score 矩阵能够直接反映每个元素对应的
Q token 和 K token 位置信息，从而帮助分析 TaskId、Q block、KV block
以及 sequence offset 的映射关系。

背景说明：

在 FlashAttention v2 的 forward kernel 中，Q 和 KV sequence 会被划分成
多个 block 进行计算。对于每个 TaskId，kernel 会加载一个 Q block 和一个
K/V block，然后执行矩阵乘法：

    S = Q @ K^T

其中 S 是当前 attention score block。

在后续 online softmax 阶段，我们需要实现 ALiBi 功能。ALiBi 计算需要知道
当前 score 元素对应的：

    batch index
    head index
    query sequence index
    key sequence index

即：

    score[b, h, q_seq, k_seq]

但是由于 FlashAttention 内部会对 batch、head、sequence 进行重新组织，
仅通过当前 S block 中的 rowIdx 和 columnIdx 很难直接确认其真实含义。

因此，需要设计特殊的 Q/K 输入，使得 GEMM 输出的 S 矩阵自身携带这些
位置信息，从而可以在 kernel debug 时直接读取 S 的数值并反推出对应的
Q/K 坐标。


设计思想：

普通的矩阵乘计算：

    S[row][col] = Q[row] · K[col]

其中 row 由 Q token 决定，col 由 K token 决定。

因此，希望构造 Q 和 K embedding，使得：

1. S 的偶数列输出当前 Q token 的身份信息；
2. S 的奇数列输出当前 K token 的 sequence index。

最终，一个 S row 应该呈现如下形式：

    [
        q_info,
        k_seq_1,
        q_info,
        k_seq_3,
        q_info,
        k_seq_5,
        ...
    ]

这样：

- 读取任意偶数 column，可以得到当前 row 对应的 Q token的信息；
- 读取任意奇数 column，可以得到当前 column 对应的 K sequenceIdx。


Q tensor 设计：

假设输入 Q 的 shape 为：

    [batch_size, seq_q, num_heads, head_dim]

要求：

    head_dim >= 2

Q embedding 的前两个维度用于 debug 编码，其余维度保持为 0。

对于每一个 Q token：

    Q[b, q_seq, head]
其有隐藏层embed，

设置其隐藏层编码信息：

    Q[..., 0] = q_info
    Q[..., 1] = 1
    其余维度保持为 0

其中 q_info 用于唯一编码 batch、head 和 query sequence：

    q_info =
        batch_idx * 1000000
        + head_idx * 10000
        + q_seq


选择该编码方式是因为测试范围为：

    batch_idx: 0 ~ 3
    head_idx : 0 ~ 15
    q_seq    : 0 ~ 1023

三个字段之间不会产生进位冲突。


Q tensor 生成逻辑：

    for b in range(batch_size):
        for h in range(num_heads):
            for q in range(seq_q):

                q_info = (
                    b * 1000000
                    + h * 10000
                    + q
                )

                Q[b][q][h][0] = q_info
                Q[b][q][h][1] = 1

                remaining dimensions are zero


K tensor 设计：

为了让不同的 Score column 输出不同的信息，需要让不同的 K column
承担不同的 probe 功能。

在一个K序列中，第j、j+1个K token的编码是
k[2*j] = [1, 0, ..., 0]
k[2*j+1] = [0, j+1, ..., 0] 
j\in(0,1,2,3, seq_k/2)

这样计算得到的S块中，第2*j列输出当前 Q token 的身份信息；第2*j+1列输出当前 K token 的 sequence index。

即：
    S[row][2*j]     = q_info
    S[row][2*j + 1] = key sequence index


K tensor 生成逻辑：

    for b in range(batch_size):
        for h in range(num_heads):
            for k in range(seq_k):

                K[b][2*k][h][0] = 1
                K[b][2*k][h][1] = 0

                K[b][2*k+1][h][0] = 0
                K[b][2*k+1][h][1] = k

                remaining dimensions are zero


Score 矩阵预期结果：

假设：

    batch = 2
    head = 3
    q_seq = 128

则：

    q_info =
        2 * 1000000
        + 3 * 10000
        + 128

    = 2030128


假设当前 K block 包含：

    k_seq = [0,1,2,3]

那么对应的 Score row 应该为：

    [
        2030128,
        0,
        2030128,
        1,
        2030128,
        2,
        2030128,
        3
    ]


在 kernel 中解析时：

通过：

    S[row][0]

恢复 Q token：

    q_info = S[row][0]

解析：

    batch_idx = q_info // 1000000

    remain = q_info % 1000000

    head_idx = remain // 10000

    q_seq = remain % 10000


通过：

    S[row][2*i+1]

直接获得：

    k_seq = i


实现要求：

请实现对应的 Q/K debug tensor 生成代码，并保证：

1. 支持 batch_size = 1~4；
2. 支持 num_heads = 1~16；
3. 支持 seq_q 最大 1024；
4. 支持任意 head_dim >= 2；
5. 保持 tensor dtype 与 FlashAttention 输入一致；
6. 生成的数据布局与 FlashAttention 输入布局一致；
7. 代码中提供清晰注释，说明每个维度编码的含义。


注意：

该方案只用于 FlashAttention kernel 内部调试。

由于 debug K 会将真实 KV sequence 长度扩大为两倍：

    debug_seq_k = real_seq_k * 2

因此该输入不能用于性能测试，也不能用于验证真实模型 accuracy。

它的用途是帮助定位：

    TaskId
        ->
    Q block / K block
        ->
    (batch, head, seq_q, seq_k)

之间的实际映射关系。
"""
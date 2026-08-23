import sys
import argparse
import torch
import torch_npu
torch_npu.npu.set_compile_mode(jit_compile=False)
# from flash_attn_npu_v3 import flash_attn_with_kvcache
from flash_attn_npu_v2 import flash_attn_with_kvcache


def run_test(batch, q_seqlen, kv_seqlen, head_num, kv_head, 
             head_dim_qk, head_dim_v, dtype, causal, paged, block_size, alibi_slopes):
    """运行 FlashAttention 测试"""
    
    # 参数验证
    if head_num % kv_head != 0:
        raise ValueError(f"head_num ({head_num}) 必须能被 kv_head ({kv_head}) 整除")
    
    # 创建输入
    query = torch.randn(batch, q_seqlen, head_num, head_dim_qk, dtype=dtype).npu()
    cache_seqlens = torch.full((batch,), kv_seqlen, dtype=torch.int32).npu()
    
    if paged:
        # 分页缓存模式
        num_blocks = max(10, (kv_seqlen * batch + block_size - 1) // block_size)
        key_cache = torch.randn(num_blocks, block_size, kv_head, head_dim_qk, dtype=dtype).npu()
        value_cache = torch.randn(num_blocks, block_size, kv_head, head_dim_v, dtype=dtype).npu()
    else:
        # 标准缓存模式
        key_cache = torch.randn(batch, kv_seqlen, kv_head, head_dim_qk, dtype=dtype).npu()
        value_cache = torch.randn(batch, kv_seqlen, kv_head, head_dim_v, dtype=dtype).npu()
    
    # 运行 FlashAttention
    output = flash_attn_with_kvcache(
        query, key_cache, value_cache,
        cache_seqlens=cache_seqlens,
        alibi_slopes=alibi_slopes,
        causal=causal
    )
    
    print(f"[SUCCESS]: 测试通过！")
    return output


def main():
    parser = argparse.ArgumentParser(description="FlashAttention 测试")

    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--q_seqlen', type=int, default=1024, help='Q序列长度')
    parser.add_argument('--kv_seqlen', type=int, default=1024, help='KV序列长度')
    parser.add_argument('--head_num', type=int, default=8, help='注意力头数')
    parser.add_argument('--kv_head', type=int, default=1, help='KV头数')
    parser.add_argument('--head_dim_qk', type=int, default=128, help='Q/K头维度')
    parser.add_argument('--head_dim_v', type=int, default=128, help='V头维度')
    parser.add_argument('--block_size', type=int, default=128, help='块大小')

    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16'])
    parser.add_argument('--causal', action='store_true', help='使用因果注意力')
    parser.add_argument('--paged', action='store_true', help='使用分页缓存')
    
    args = parser.parse_args()
    
    dtype = torch.float16 if args.dtype == 'float16' else torch.bfloat16
    
    alibi_slopes = torch.randn(args.batch, args.head_num, dtype=dtype).npu()
    
    try:
        run_test(args.batch, args.q_seqlen, args.kv_seqlen, args.head_num, 
                args.kv_head, args.head_dim_qk, args.head_dim_v, dtype, 
                args.causal, args.paged, args.block_size, alibi_slopes)
        return 0
    except Exception as e:
        print(f"[ERROR]: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

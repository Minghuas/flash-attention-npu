import sys
import torch
import torch_npu
# torch_npu.npu.set_device(5)
from flash_attn_npu_v2 import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache


def make_alibi_slopes(num_heads, batch_size=None):
    """Fixed deterministic ALiBi slopes: 0.5/(2^h) per head. Same per batch."""
    _h = torch.tensor([0.5 / (2 ** h) for h in range(num_heads)], dtype=torch.float32)
    if batch_size is not None:
        _h = _h.unsqueeze(0).repeat(batch_size, 1)
    return _h.npu()


def run_fwd():
    """
    标准 Flash Attention 前向推理
    输入形状: (batch_size, seqlen, nheads, headdim)
    """
    batch_size, seqlen, nheads, headdim = 2, 512, 8, 128
    
    # 随机初始化 Q, K, V
    q = torch.randn(batch_size, seqlen, nheads, headdim, device='npu', dtype=torch.float16, requires_grad=False)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device='npu', dtype=torch.float16, requires_grad=False)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device='npu', dtype=torch.float16, requires_grad=False)
    
    # 构建 ALiBi slopes
    alibi_slopes = make_alibi_slopes(nheads, batch_size)
    
    # 调用接口
    output = flash_attn_func(
        q=q,
        k=k,
        v=v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=alibi_slopes,
        deterministic=False,
        return_attn_probs=False
    )
    return output

def run_varlen_fwd():
    """
    变长序列 Flash Attention 前向推理
    输入形状: (total_tokens, nheads, headdim)
    """
    max_seqlen = 128
    nheads, headdim = 8, 64
    
    # 模拟变长序列长度，例如 [100, 128]
    seqlens = [100, 128]
    total_tokens = sum(seqlens)
    
    # 构建 cu_seqlens: [0, 100, 228]
    cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32, device='npu')
    
    # 随机初始化 Q, K, V (packed format)
    q = torch.randn(total_tokens, nheads, headdim, device='npu', dtype=torch.float16, requires_grad=False)
    k = torch.randn(total_tokens, nheads, headdim, device='npu', dtype=torch.float16, requires_grad=False)
    v = torch.randn(total_tokens, nheads, headdim, device='npu', dtype=torch.float16, requires_grad=False)
    
    # 构建 ALiBi slopes
    alibi_slopes = make_alibi_slopes(nheads)
    
    # 调用接口
    output = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True, # 变长通常用于解码，开启 causal
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=alibi_slopes,
        deterministic=False,
        return_attn_probs=False
    )
    return output

def run_kv_cache():
    """
    带 KV Cache 的推理模式
    用于自回归生成，更新 cache 并计算 attention
    """
    batch_size, seqlen_q, nheads, headdim = 2, 1, 8, 64 # seqlen_q=1 表示单步解码
    seqlen_k_cache = 1024 # 缓存的最大长度
    
    # 1. 初始化 Cache (预先分配好内存)
    # 形状: (batch_size, max_seqlen, nheads, headdim)
    k_cache = torch.zeros(batch_size, seqlen_k_cache, nheads, headdim, device='npu', dtype=torch.float16)
    v_cache = torch.zeros(batch_size, seqlen_k_cache, nheads, headdim, device='npu', dtype=torch.float16)
    
    # 2. 初始化当前 Step 的输入
    q = torch.randn(batch_size, seqlen_q, nheads, headdim, device='npu', dtype=torch.float16)
    k = torch.randn(batch_size, seqlen_q, nheads, headdim, device='npu', dtype=torch.float16)
    v = torch.randn(batch_size, seqlen_q, nheads, headdim, device='npu', dtype=torch.float16)
    
    # 3. 记录当前缓存的序列长度 (用于指示写入位置)
    # 假设当前缓存里已经有 10 个 token
    cache_seqlens = torch.tensor([10, 10], dtype=torch.int32, device='npu')
    
    # 构建 ALiBi slopes
    alibi_slopes = make_alibi_slopes(nheads, batch_size)
    
    output = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k, # 提供 k/v 会触发 inplace 更新 cache
        v=v,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=None,
        cache_leftpad=None,
        block_table=None,
        softmax_scale=None,
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
        rotary_cos=None,
        rotary_sin=None,
        rotary_interleaved=True,
        alibi_slopes=alibi_slopes,
        num_splits=0,
        return_softmax_lse=False
    )
    return output

def main(func):
    name=func.__name__
    try:
        start_event = torch_npu.npu.Event(enable_timing=True)
        end_event = torch_npu.npu.Event(enable_timing=True)

        for _ in range(10): # warm-up runs
            func()

        torch.npu.synchronize()
        start_event.record()

        for _ in range(100): # timed runs
            func()

        end_event.record()
        end_event.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Average time for 100 runs of {name}: {elapsed_time_ms / 1000:.3f} ms")
        
        return 0
    except Exception as e:
        print(f"[ERROR]: {e}")
        return 1


if __name__ == '__main__':
    torch_npu.npu.set_device(3) 
    # main(run_varlen_fwd)
    # main(run_fwd)
    main(run_kv_cache)

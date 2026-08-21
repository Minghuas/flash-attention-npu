# Copyright (c) 2026, Minghua Shen.

import torch
import torch_npu
import pytest
from tests.common.attention_ref import ref_flash_attention_pair, ref_masked_attention
from tests.common.compare import assert_fa_close
from tests.common.test_utils import gather_paged_kv, make_block_table, make_local_attention_mask
if "Ascend950" in torch_npu.npu.get_device_name():
    from flash_attn_npu_4 import flash_attn_varlen_func
else:
    from flash_attn_npu_4 import flash_attn_varlen_func

def build_cann_causal_mask():
    """Fixed [2048, 2048] causal mask for npu_fused_infer_attention_score."""
    return torch.triu(torch.ones(2048, 2048), diagonal=1).bool().npu()


test_cases = [
    # (data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode,
    #  block_size, is_causal, layout, is_varied, window_size_left, window_size_right)
    (torch.bfloat16, 2, 6, 2, 2, 1024, 128, 1, 128, False, "BSND", False, -1, -1),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 1, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, 1, 128, True, "TND", False, -1, -1),
    (torch.float16, 7, 1, 1, 512, 512, 128, 1, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 1, 128, False, "BSND", False, -1, -1),
    (torch.bfloat16, 2, 1, 1, 1024, 1024, 128, 1, 128, False, "BSND", False, -1, -1),
    (torch.bfloat16, 2, 1, 1, 1024, 1024, 128, 1, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, 1, 128, True, "BSND", False, -1, -1),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, 1, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 1, 1, 1, 1, 1024, 128, 1, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 1, 1, 1, 1, 1024, 128, 1, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 1, 1, 1, 1, 1024, 128, 1, 128, True, "BSND", False, -1, -1),
    (torch.bfloat16, 1, 1, 1, 1, 1024, 128, 1, 128, False, "BSND", False, -1, -1),
    # kv=4096 -> 8 S2 blocks: num_splits=2 -> 2 segs (4 blk each), num_splits=4 -> 4 segs (2 blk each).
    (torch.bfloat16, 1, 1, 1, 1, 4096, 128, 1, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 2, 1, 1, 1, 2048, 128, 1, 128, False, "TND", False, -1, -1),
    (torch.float16, 2, 2, 1, 128, 128, 128, 1, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 2, 6, 2, 2, 1024, 128, 1, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 2, 1, 1, 16, 1024, 128, 1, 128, False, "TND", True, -1, -1),
    (torch.bfloat16, 2, 6, 2, 16, 1024, 128, 1, 128, False, "TND", True, -1, -1),
    (torch.bfloat16, 2, 6, 2, 16, 1024, 128, 1, 128, True, "TND", True, -1, -1),
    (torch.bfloat16, 1, 64, 1, 2, 1024, 256, 1, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 2, 1, 1, 16, 1024, 256, 1, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 2, 1, 1, 16, 10240, 128, 1, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 2, 6, 2, 16, 10240, 128, 1, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 6, 1, 1, 16, 10240, 128, 1, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 1, 128, True, "BSND", False, 512, 0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 1, 128, True, "TND", False, 512, 0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 1, 128, False, "TND", False, 0, 256),
    (torch.float16, 2, 1, 1, 512, 512, 128, 1, 128, False, "TND", False, 508, -256),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 1, 128, False, "BSND", False, -128, 1024),
    (torch.float16, 2, 2, 2, 512, 512, 128, 0, 128, False, "TND", False, 64, 128),
    # SWA + large GQA decode: rowLoopNum>1 must not hang (EVENT_ID0 order in online_softmax)
    (torch.float16, 1, 64, 1, 1, 1024, 128, 0, 128, True, "BSND", False, 542, 647),
    (torch.float16, 1, 128, 1, 1, 1024, 128, 0, 128, True, "BSND", False, 542, 647),
    (torch.float16, 1, 512, 1, 1, 1024, 128, 0, 128, True, "BSND", False, 542, 647),
    (torch.bfloat16, 1, 128, 1, 1, 1024, 128, 0, 128, True, "TND", False, 64, 0),
    (torch.float16, 1, 512, 1, 1, 1024, 128, 0, 128, True, "TND", False, 542, 647),
    # Sq>>Sk SWA: left window collapses to -1; golden must zero fully-masked q rows via mask
    (torch.float16, 2, 16, 8, 1024, 128, 128, 0, 128, False, "BSND", False, 497, 265),

    # ===== MHA + BF16 + BSND (causal & non-causal) =====
    (torch.bfloat16, 2, 8, 8, 512, 512, 128, 0, 128, True, "BSND", False, -1, -1),
    (torch.bfloat16, 4, 16, 16, 128, 256, 128, 0, 128, False, "BSND", False, -1, -1),
    (torch.bfloat16, 1, 32, 32, 128, 128, 128, 0, 128, True, "BSND", False, -1, -1),
    (torch.bfloat16, 3, 4, 4, 256, 512, 128, 0, 128, False, "BSND", False, -1, -1),
    (torch.bfloat16, 8, 4, 4, 128, 128, 128, 0, 128, True, "BSND", False, -1, -1),

    # ===== MHA + BF16 + TND (causal & non-causal) =====
    (torch.bfloat16, 3, 4, 4, 64, 1024, 128, 0, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 2, 8, 8, 128, 512, 128, 0, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 1, 16, 16, 64, 1024, 128, 0, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 4, 4, 4, 256, 512, 128, 0, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 2, 8, 8, 1, 512, 128, 0, 128, True, "TND", False, -1, -1),

    # ===== MHA + FP16 + BSND (causal & non-causal) =====
    (torch.float16, 3, 8, 8, 128, 512, 128, 0, 128, True, "BSND", False, -1, -1),
    (torch.float16, 2, 4, 4, 256, 256, 128, 0, 128, False, "BSND", False, -1, -1),
    (torch.float16, 1, 16, 16, 128, 128, 128, 0, 128, True, "BSND", False, -1, -1),
    (torch.float16, 4, 8, 8, 256, 512, 128, 0, 128, False, "BSND", False, -1, -1),
    (torch.float16, 2, 4, 4, 512, 512, 128, 0, 128, True, "BSND", False, -1, -1),

    # ===== MHA + FP16 + TND (causal & non-causal) =====
    (torch.float16, 4, 8, 8, 64, 1024, 128, 0, 128, True, "TND", False, -1, -1),
    (torch.float16, 2, 16, 16, 128, 256, 128, 0, 128, False, "TND", False, -1, -1),
    (torch.float16, 8, 4, 4, 64, 1024, 128, 0, 128, True, "TND", False, -1, -1),
    (torch.float16, 3, 8, 8, 256, 512, 128, 0, 128, False, "TND", False, -1, -1),
    (torch.float16, 2, 2, 2, 128, 1024, 128, 0, 128, True, "TND", False, -1, -1),

    # ===== GQA + BF16 + BSND (causal & non-causal) =====
    (torch.bfloat16, 2, 8, 2, 512, 512, 128, 0, 128, True, "BSND", False, -1, -1),
    (torch.bfloat16, 3, 12, 4, 256, 256, 128, 0, 128, False, "BSND", False, -1, -1),
    (torch.bfloat16, 1, 32, 8, 128, 128, 128, 0, 128, True, "BSND", False, -1, -1),
    (torch.bfloat16, 2, 16, 4, 128, 512, 128, 0, 128, False, "BSND", False, -1, -1),
    (torch.bfloat16, 2, 8, 4, 128, 2048, 128, 0, 128, False, "BSND", False, -1, -1),

    # ===== GQA + BF16 + TND (causal & non-causal) =====
    (torch.bfloat16, 2, 8, 2, 64, 1024, 128, 0, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 4, 16, 4, 128, 512, 128, 0, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 2, 24, 6, 64, 1024, 128, 0, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 6, 8, 2, 128, 512, 128, 0, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 2, 8, 2, 1, 512, 128, 0, 128, False, "TND", False, -1, -1),

    # ===== GQA + FP16 + BSND (causal & non-causal) =====
    (torch.float16, 2, 8, 2, 128, 512, 128, 0, 128, True, "BSND", False, -1, -1),
    (torch.float16, 3, 12, 3, 256, 256, 128, 0, 128, False, "BSND", False, -1, -1),
    (torch.float16, 1, 16, 4, 128, 128, 128, 0, 128, True, "BSND", False, -1, -1),
    (torch.float16, 2, 12, 6, 256, 512, 128, 0, 128, False, "BSND", False, -1, -1),
    (torch.float16, 2, 8, 4, 512, 512, 128, 0, 128, True, "BSND", False, -1, -1),

    # ===== GQA + FP16 + TND (causal & non-causal) =====
    (torch.float16, 2, 8, 2, 128, 1024, 128, 0, 128, True, "TND", False, -1, -1),
    (torch.float16, 4, 16, 8, 128, 512, 128, 0, 128, False, "TND", False, -1, -1),
    (torch.float16, 2, 12, 4, 64, 1024, 128, 0, 128, True, "TND", False, -1, -1),
    (torch.float16, 3, 8, 2, 128, 512, 128, 0, 128, False, "TND", False, -1, -1),
    (torch.float16, 2, 16, 4, 64, 1024, 128, 0, 128, True, "TND", False, -1, -1),

    # ===== MQA + BF16 + BSND (causal & non-causal) =====
    (torch.bfloat16, 2, 4, 1, 512, 512, 128, 0, 128, True, "BSND", False, -1, -1),
    (torch.bfloat16, 3, 8, 1, 256, 256, 128, 0, 128, False, "BSND", False, -1, -1),
    (torch.bfloat16, 1, 16, 1, 128, 128, 128, 0, 128, True, "BSND", False, -1, -1),
    (torch.bfloat16, 2, 8, 1, 128, 512, 128, 0, 128, False, "BSND", False, -1, -1),
    (torch.bfloat16, 1, 32, 1, 128, 128, 128, 0, 128, True, "BSND", False, -1, -1),

    # ===== MQA + BF16 + TND (causal & non-causal) =====
    (torch.bfloat16, 2, 4, 1, 64, 1024, 128, 0, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 4, 8, 1, 128, 512, 128, 0, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 2, 8, 1, 1, 512, 128, 0, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 3, 4, 1, 256, 512, 128, 0, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 2, 4, 1, 64, 2048, 128, 0, 128, True, "TND", False, -1, -1),

    # ===== MQA + FP16 + BSND/TND (causal & non-causal) =====
    (torch.float16, 2, 4, 1, 128, 512, 128, 0, 128, True, "BSND", False, -1, -1),
    (torch.float16, 3, 8, 1, 64, 1024, 128, 0, 128, True, "TND", False, -1, -1),
    (torch.float16, 2, 8, 1, 256, 256, 128, 0, 128, False, "TND", False, -1, -1),
    (torch.float16, 2, 4, 1, 1, 512, 128, 0, 128, True, "BSND", False, -1, -1),
    (torch.float16, 4, 8, 1, 128, 512, 128, 0, 128, False, "BSND", False, -1, -1),

    # ===== head_size=256 + MHA/GQA/MQA + BF16/FP16 + BSND/TND =====
    (torch.bfloat16, 2, 4, 4, 128, 256, 256, 0, 128, True, "BSND", False, -1, -1),
    (torch.bfloat16, 2, 4, 4, 256, 256, 256, 0, 128, False, "BSND", False, -1, -1),
    (torch.bfloat16, 2, 8, 2, 64, 512, 256, 0, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 2, 4, 1, 128, 256, 256, 0, 128, False, "TND", False, -1, -1),
    (torch.float16, 2, 8, 2, 128, 128, 256, 0, 128, True, "BSND", False, -1, -1),
    (torch.float16, 2, 4, 4, 64, 512, 256, 0, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 1, 8, 8, 128, 256, 256, 0, 128, False, "TND", False, -1, -1),
    (torch.float16, 2, 4, 1, 64, 512, 256, 0, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 2, 8, 4, 128, 256, 256, 0, 128, True, "BSND", False, -1, -1),
    (torch.float16, 2, 4, 4, 128, 256, 256, 0, 128, False, "BSND", False, -1, -1),

    # ===== Paged KV cache (cache_mode=1) + MHA/GQA/MQA + BF16/FP16 + BSND/TND =====
    (torch.bfloat16, 2, 4, 4, 256, 1024, 128, 1, 128, True, "BSND", False, -1, -1),
    (torch.bfloat16, 3, 8, 8, 128, 512, 128, 1, 128, False, "BSND", False, -1, -1),
    (torch.bfloat16, 2, 8, 2, 64, 1024, 128, 1, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 4, 4, 1, 128, 512, 128, 1, 128, False, "TND", False, -1, -1),
    (torch.float16, 2, 8, 4, 128, 512, 128, 1, 128, True, "BSND", False, -1, -1),
    (torch.float16, 2, 4, 4, 64, 1024, 128, 1, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 2, 4, 4, 256, 1024, 128, 1, 128, False, "TND", False, -1, -1),
    (torch.float16, 3, 8, 1, 64, 1024, 128, 1, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 2, 4, 1, 256, 512, 128, 1, 128, False, "BSND", False, -1, -1),
    (torch.float16, 2, 8, 8, 128, 256, 128, 1, 128, False, "BSND", False, -1, -1),

    # ===== head_size=64 + MHA/GQA/MQA + BF16/FP16 + BSND/TND =====
    (torch.bfloat16, 2, 16, 16, 512, 512, 64, 0, 128, True, "BSND", False, -1, -1),
    (torch.bfloat16, 2, 8, 2, 128, 1024, 64, 0, 128, True, "TND", False, -1, -1),
    (torch.float16, 2, 4, 1, 256, 256, 64, 0, 128, True, "BSND", False, -1, -1),
    (torch.float16, 3, 32, 32, 64, 512, 64, 0, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 2, 8, 1, 64, 1024, 64, 0, 128, True, "TND", False, -1, -1),

    # ===== is_varied + TND + MHA/GQA/MQA + BF16/FP16 =====
    (torch.bfloat16, 3, 8, 8, 16, 1024, 128, 0, 128, True, "TND", True, -1, -1),
    (torch.bfloat16, 2, 4, 1, 16, 512, 128, 0, 128, False, "TND", True, -1, -1),
    (torch.float16, 4, 8, 2, 16, 1024, 128, 0, 128, True, "TND", True, -1, -1),
    (torch.bfloat16, 2, 12, 4, 16, 1024, 128, 0, 128, False, "TND", True, -1, -1),
    (torch.float16, 3, 4, 4, 16, 512, 128, 0, 128, True, "TND", True, -1, -1),

    # ===== Mixed: head_size=256 + cache_mode=1 + BSND/TND =====
    (torch.bfloat16, 2, 4, 4, 128, 256, 256, 1, 128, False, "BSND", False, -1, -1),
    (torch.bfloat16, 2, 8, 2, 64, 512, 256, 1, 128, True, "TND", False, -1, -1),
    (torch.float16, 2, 4, 1, 128, 256, 256, 1, 128, True, "TND", False, -1, -1),
    (torch.float16, 2, 4, 4, 64, 256, 256, 1, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 2, 8, 4, 128, 256, 256, 1, 128, True, "BSND", False, -1, -1),

    # ===== Additional coverage: large kv_seqlen, large batch, edge cases =====
    (torch.bfloat16, 2, 4, 4, 128, 2048, 128, 0, 128, False, "BSND", False, -1, -1),
    (torch.float16, 2, 4, 4, 256, 2048, 128, 0, 128, False, "BSND", False, -1, -1),
    (torch.bfloat16, 2, 8, 2, 128, 2048, 128, 0, 128, True, "BSND", False, -1, -1),
    (torch.bfloat16, 2, 4, 1, 64, 2048, 128, 0, 128, True, "TND", False, -1, -1),
    (torch.float16, 2, 8, 8, 128, 2048, 128, 0, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 8, 4, 4, 128, 128, 128, 0, 128, True, "BSND", False, -1, -1),
    (torch.float16, 8, 4, 4, 64, 1024, 128, 0, 128, True, "TND", False, -1, -1),
    (torch.bfloat16, 6, 8, 2, 128, 512, 128, 0, 128, False, "TND", False, -1, -1),
    (torch.bfloat16, 2, 32, 32, 64, 128, 128, 0, 128, True, "BSND", False, -1, -1),
    (torch.float16, 2, 32, 32, 64, 256, 128, 0, 128, False, "TND", False, -1, -1),

    # ===== Non-paged TND varied lengths covered by the fused forward/backward path =====
    (torch.bfloat16, 2, 4, 2, 128, 128, 64, 0, 128, True, "TND", True, -1, -1),
    (torch.bfloat16, 3, 1, 1, 512, 1024, 128, 0, 128, True, "TND", True, -1, -1),
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, 0, 128, False, "TND", True, -1, -1),
    (torch.float16, 7, 5, 1, 512, 512, 128, 0, 128, True, "TND", True, -1, -1),
    (torch.float16, 7, 5, 1, 777, 888, 192, 0, 128, False, "TND", True, -1, -1),
    (torch.float16, 7, 5, 1, 1777, 1888, 256, 0, 128, True, "TND", True, -1, -1),
    (torch.bfloat16, 3, 1, 1, 7777, 8192, 64, 0, 128, True, "TND", True, -1, -1),
    (torch.bfloat16, 7, 5, 1, 711, 8192, 111, 0, 128, True, "TND", True, -1, -1),
    (torch.bfloat16, 3, 16, 16, 562, 562, 96, 0, 128, False, "TND", True, -1, -1),
    (torch.bfloat16, 3, 4, 4, 1024, 1024, 128, 0, 128, True, "TND", True, 512, 0),
    (torch.bfloat16, 3, 1, 1, 512, 1024, 128, 0, 128, True, "TND", True, 512, 0),
    (torch.bfloat16, 3, 1, 1, 512, 1024, 128, 0, 128, False, "TND", True, 0, 256),
    (torch.float16, 3, 2, 2, 512, 512, 128, 0, 128, False, "TND", True, 64, 128),
    (torch.bfloat16, 3, 4, 4, 1024, 1024, 128, 0, 128, True, "TND", True, -128, 864),
    (torch.float16, 3, 12, 3, 1024, 1024, 64, 0, 128, False, "TND", True, -1, 1890),
]

@pytest.mark.parametrize("num_splits", [0, 1, 2])
@pytest.mark.parametrize("data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, is_causal, layout, is_varied, window_size_left, window_size_right", test_cases)
def test_fa_custom_ops(data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, is_causal, layout, is_varied, window_size_left, window_size_right, num_splits):
    # num_splits>1 (active KV split) is currently only wired for paged KV + varlen-q (TND).
    name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
    if num_splits > 1 and not (cache_mode == 1 and layout == "TND"):
        pytest.skip("num_splits>1 requires paged KV cache and TND (varlen-q) layout")
    if "Ascend950" in name and num_splits > 1:
        pytest.skip("Ascend950 does not support num_splits>1")
    if "Ascend950" in name and not (1 <= head_size <= 256):
        pytest.skip("Ascend950 supports head_size in [1, 256]")

    if "Ascend950" in name and (window_size_left != -1 or window_size_right != -1):
        pytest.skip("Ascend950 does not support SWA")
    if is_varied and layout != "TND":
        pytest.skip("is_varied requires TND (varlen-q) layout")
    q_min_range = -5.0
    q_max_range = 5.0
    kv_min_range = -5.0
    kv_max_range = 5.0
    block_size = 128
    max_num_blocks_per_seq = (kv_seqlen + block_size - 1) // block_size
    num_blocks = max(64, max_num_blocks_per_seq * batch_size)
    if is_varied:
        # Per-batch q in [1, q_seqlen], kv in [q, kv_seqlen] (kv>=q so q>kv never occurs).
        # Seeded for reproducibility; does not perturb the query/key/value RNG streams above.
        gen = torch.Generator().manual_seed(1234)
        q_sequences = torch.randint(low=1, high=q_seqlen + 1, size=(batch_size,), generator=gen).tolist()
        kv_sequences = [int(torch.randint(low=min(q, kv_seqlen), high=kv_seqlen + 1, size=(1,), generator=gen))
                        for q in q_sequences]
    else:
        q_sequences = [q_seqlen] * batch_size
        kv_sequences = [kv_seqlen] * batch_size
    t_q_sum = sum(q_sequences)
    t_kv_sum = sum(kv_sequences)
    if layout == "BSND":
        query = (
            q_min_range + (q_max_range - q_min_range) * torch.rand(batch_size, q_seqlen, num_heads, head_size)
        ).to(data_type).npu().requires_grad_(True)
    elif layout == "TND":
        query = (
            q_min_range + (q_max_range - q_min_range) * torch.rand(t_q_sum, num_heads, head_size)
        ).to(data_type).npu().requires_grad_(True)
    key_cache = None
    value_cache = None
    block_tables = None
    if cache_mode == 1:
        key_cache = (
            kv_min_range + (kv_max_range - kv_min_range) * torch.rand(num_blocks, block_size, kv_heads, head_size)
        ).to(data_type).npu().requires_grad_(True)
        value_cache = (
            kv_min_range + (kv_max_range - kv_min_range) * torch.rand(num_blocks, block_size, kv_heads, head_size)
        ).to(data_type).npu().requires_grad_(True)
        block_tables = make_block_table(batch_size, kv_seqlen, block_size).npu()
    else:
        if layout == "BSND":
            key_cache = (
                kv_min_range + (kv_max_range - kv_min_range) * torch.rand(batch_size, kv_seqlen, kv_heads, head_size)
            ).to(data_type).npu().requires_grad_(True)
            value_cache = (
                kv_min_range + (kv_max_range - kv_min_range) * torch.rand(batch_size, kv_seqlen, kv_heads, head_size)
            ).to(data_type).npu().requires_grad_(True)
        else:
            key_cache = (
                kv_min_range + (kv_max_range - kv_min_range) * torch.rand(t_kv_sum, kv_heads, head_size)
            ).to(data_type).npu().requires_grad_(True)
            value_cache = (
                kv_min_range + (kv_max_range - kv_min_range) * torch.rand(t_kv_sum, kv_heads, head_size)
            ).to(data_type).npu().requires_grad_(True)
        block_tables = None
    if layout == "BSND":
        q_seqlen_list = [q_seqlen] * batch_size
        kv_seqlen_list = [kv_seqlen] * batch_size
    else:
        q_seqlen_list = q_sequences
        kv_seqlen_list = kv_sequences
    scale = 1.0 / (head_size ** 0.5)
    kv_seqlen_list = torch.tensor(kv_seqlen_list, dtype=torch.int32).npu()
    new_q_seqlen_list = None
    new_kv_seqlen_list = None
    new_q_seqlen_list_cpu = None
    new_kv_seqlen_list_cpu = None
    window_size_left_golden = window_size_left
    window_size_right_golden = window_size_right
    # Match Tri Dao GPU host: both sides vs kv_seqlen.
    if kv_seqlen > 0 and window_size_left_golden >= kv_seqlen:
        window_size_left_golden = -1
    if kv_seqlen > 0 and window_size_right_golden >= kv_seqlen:
        window_size_right_golden = -1
    if is_causal:
        window_size_right_golden = 0
    is_causal_golden = (window_size_left_golden < 0 and window_size_right_golden == 0)
    is_local_golden = (window_size_left_golden >= 0 or window_size_right_golden > 0) and not is_causal_golden
    if is_local_golden:
        if window_size_left_golden < 0:
            window_size_left_golden = kv_seqlen
        if window_size_right_golden < 0:
            window_size_right_golden = kv_seqlen
    if layout == "TND":
        new_q_seqlen_list_cpu = [0]
        pre_seq_sum = 0
        for i in range(batch_size):
            pre_seq_sum += q_sequences[i]
            new_q_seqlen_list_cpu.append(pre_seq_sum)
        new_q_seqlen_list = torch.tensor(new_q_seqlen_list_cpu, dtype=torch.int32).npu()
        if cache_mode == 0:
            new_kv_seqlen_list_cpu = [0]
            pre_seq_sum = 0
            for i in range(batch_size):
                pre_seq_sum += kv_sequences[i]
                new_kv_seqlen_list_cpu.append(pre_seq_sum)
            new_kv_seqlen_list = torch.tensor(new_kv_seqlen_list_cpu, dtype=torch.int32).npu()
    bwd_supported = layout == "TND" and cache_mode == 0 and num_splits <= 1
    cu_seqlens_k_for_api = new_kv_seqlen_list if bwd_supported else None
    max_seqlen_k_for_api = kv_seqlen if bwd_supported else None
    cache_seqlens_for_api = None if bwd_supported else (
        new_kv_seqlen_list if (layout == "TND" and cache_mode == 0) else kv_seqlen_list
    )
    out_out, softmax_lse, *rest = flash_attn_varlen_func(
        query,
        key_cache,
        value_cache,
        qv=None,
        cu_seqlens_q=new_q_seqlen_list,
        cu_seqlens_k=cu_seqlens_k_for_api,
        max_seqlen_q=q_seqlen,
        max_seqlen_k=max_seqlen_k_for_api,
        seqused_k=cache_seqlens_for_api,
        page_table=block_tables,
        softmax_scale=None,
        causal=is_causal,
        window_size=[window_size_left, window_size_right],  # -1 means infinite context window
        softcap=0.0, # 0.0 means deactivated
        num_splits=num_splits,    # Can be tuned for speed
        pack_gqa=None,   # Can be tuned for speed
        return_lse=True,
    )
    query_ref = query.detach().cpu().requires_grad_(True)
    key_ref = key_cache.detach().cpu().requires_grad_(True)
    value_ref = value_cache.detach().cpu().requires_grad_(True)
    block_tables_cpu = block_tables.cpu() if cache_mode == 1 else None

    golden_out_gpu_ref_list = []
    golden_out_gpu_pt_list = []
    golden_out_plain_list = []
    if layout == "BSND":
        golden_lseL_gpu_ref = torch.empty((batch_size, num_heads, q_seqlen), dtype=torch.float32)
        golden_lseL_gpu_pt = torch.empty_like(golden_lseL_gpu_ref)
        golden_lseL = torch.empty((batch_size, num_heads, q_seqlen), dtype=torch.float32)
    else:
        golden_lseL_gpu_ref = torch.empty((num_heads, t_q_sum), dtype=torch.float32)
        golden_lseL_gpu_pt = torch.empty_like(golden_lseL_gpu_ref)
        golden_lseL = torch.empty((num_heads, t_q_sum), dtype=torch.float32)
    for i in range(batch_size):
        q_seqlen_per_batch = q_sequences[i]
        kv_seqlen_per_batch = kv_sequences[i]
        key_cache_per_batch = None
        value_cache_per_batch = None
        query_cpu_per_batch = None
        atten_mask = None
        if is_causal_golden:
            atten_mask = torch.triu(
                torch.ones(q_seqlen_per_batch, kv_seqlen_per_batch),
                diagonal=(kv_seqlen_per_batch - q_seqlen_per_batch + 1),
            ).bool()
        elif is_local_golden:
            atten_mask = make_local_attention_mask(
                q_seqlen_per_batch,
                kv_seqlen_per_batch,
                window_size_left_golden,
                window_size_right_golden,
            )
        if layout == "BSND":
            query_cpu_per_batch = query_ref[i]
            if cache_mode == 1:
                key_cache_per_batch, value_cache_per_batch = gather_paged_kv(
                    key_ref,
                    value_ref,
                    block_tables_cpu[i],
                    kv_seqlen_per_batch,
                    block_size,
                )
            else:
                key_cache_per_batch = key_ref[i]
                value_cache_per_batch = value_ref[i]
        else:
            query_cpu_per_batch = query_ref[new_q_seqlen_list_cpu[i] : new_q_seqlen_list_cpu[i + 1]]
            if cache_mode == 0:
                key_cache_per_batch = key_ref[new_kv_seqlen_list_cpu[i] : new_kv_seqlen_list_cpu[i + 1]]
                value_cache_per_batch = value_ref[new_kv_seqlen_list_cpu[i] : new_kv_seqlen_list_cpu[i + 1]]
            else:
                key_cache_per_batch, value_cache_per_batch = gather_paged_kv(
                    key_ref,
                    value_ref,
                    block_tables_cpu[i],
                    kv_seqlen_per_batch,
                    block_size,
                )
        query_plain_per_batch = query_cpu_per_batch.detach()
        key_plain_per_batch = key_cache_per_batch.detach()
        value_plain_per_batch = value_cache_per_batch.detach()
        if atten_mask is not None:
            output_gpu_ref, golden_lse_gpu_ref, output_gpu_pt, golden_lse_gpu_pt = ref_flash_attention_pair(
                query_cpu_per_batch, key_cache_per_batch, value_cache_per_batch,
                scale, atten_mask, data_type, rescale_threshold=4.0,
            )
            output, golden_lse = ref_masked_attention(query_plain_per_batch, key_plain_per_batch, value_plain_per_batch, scale, atten_mask, None)
        else:
            output_gpu_ref, golden_lse_gpu_ref, output_gpu_pt, golden_lse_gpu_pt = ref_flash_attention_pair(
                query_cpu_per_batch, key_cache_per_batch, value_cache_per_batch,
                scale, None, data_type, rescale_threshold=4.0,
            )
            output, golden_lse = ref_masked_attention(query_plain_per_batch, key_plain_per_batch, value_plain_per_batch, scale, None, None)
        out_gpu_ref = output_gpu_ref.reshape(q_seqlen_per_batch, num_heads, head_size)
        out_gpu_pt = output_gpu_pt.reshape(q_seqlen_per_batch, num_heads, head_size)
        out_plain = output.reshape(q_seqlen_per_batch, num_heads, head_size)
        lse_plain = torch.from_numpy(golden_lse)
        if is_local_golden and atten_mask is not None:
            # Soft mask still yields finite garbage on fully-masked rows;
            # NPU zeroes them / sets lse=inf. Infinite window (-1) must not go
            # through the numeric pre/nextTokensError heuristics.
            fully_masked = atten_mask.all(dim=-1)
            out_gpu_ref = out_gpu_ref.masked_fill(fully_masked[:, None, None], 0)
            out_gpu_pt = out_gpu_pt.masked_fill(fully_masked[:, None, None], 0)
            golden_lse_gpu_ref[:, fully_masked] = torch.inf
            golden_lse_gpu_pt[:, fully_masked] = torch.inf
            out_plain[fully_masked, :, :] = 0
            lse_plain[:, fully_masked] = torch.inf
        if is_causal_golden and atten_mask is not None:
            fully_masked = atten_mask.all(dim=-1)
            out_gpu_ref = out_gpu_ref.masked_fill(fully_masked[:, None, None], 0)
            out_gpu_pt = out_gpu_pt.masked_fill(fully_masked[:, None, None], 0)
            golden_lse_gpu_ref[:, fully_masked] = torch.inf
            golden_lse_gpu_pt[:, fully_masked] = torch.inf
            out_plain[fully_masked, :, :] = 0
            lse_plain[:, fully_masked] = torch.inf
        if layout == "BSND":
            golden_out_gpu_ref_list.append(out_gpu_ref)
            golden_out_gpu_pt_list.append(out_gpu_pt)
            golden_lseL_gpu_ref[i:i+1] = golden_lse_gpu_ref.reshape(1, num_heads, q_seqlen_per_batch)
            golden_lseL_gpu_pt[i:i+1] = golden_lse_gpu_pt.reshape(1, num_heads, q_seqlen_per_batch)
            golden_out_plain_list.append(out_plain)
            golden_lseL[i:i+1] = lse_plain.reshape(1, num_heads, q_seqlen_per_batch)
        else:
            golden_out_gpu_ref_list.append(out_gpu_ref)
            golden_out_gpu_pt_list.append(out_gpu_pt)
            golden_lseL_gpu_ref[:, new_q_seqlen_list[i] : new_q_seqlen_list[i + 1]] = golden_lse_gpu_ref.reshape(num_heads, q_seqlen_per_batch)
            golden_lseL_gpu_pt[:, new_q_seqlen_list[i] : new_q_seqlen_list[i + 1]] = golden_lse_gpu_pt.reshape(num_heads, q_seqlen_per_batch)
            golden_out_plain_list.append(out_plain)
            golden_lseL[:, new_q_seqlen_list[i] : new_q_seqlen_list[i + 1]] = lse_plain.reshape(num_heads, q_seqlen_per_batch)
    if layout == "BSND":
        golden_out_gpu_ref = torch.stack(golden_out_gpu_ref_list, dim=0)
        golden_out_gpu_pt = torch.stack(golden_out_gpu_pt_list, dim=0)
        golden_out = torch.stack(golden_out_plain_list, dim=0)
    else:
        golden_out_gpu_ref = torch.cat(golden_out_gpu_ref_list, dim=0)
        golden_out_gpu_pt = torch.cat(golden_out_gpu_pt_list, dim=0)
        golden_out = torch.cat(golden_out_plain_list, dim=0)
    assert_fa_close(out_out, golden_out_gpu_ref, golden_out_gpu_pt, name="out")
    if "Ascend910" in name:
        assert_fa_close(softmax_lse, golden_lseL_gpu_ref, golden_lseL_gpu_pt, name="softmax_lse")
    if bwd_supported:
        dout = torch.rand_like(out_out) - 0.5
        dq_ag, dk_ag, dv_ag = torch.autograd.grad(out_out, (query, key_cache, value_cache), dout)
        dq_ref, dk_ref, dv_ref = torch.autograd.grad(
            golden_out_gpu_ref,
            (query_ref, key_ref, value_ref),
            dout.detach().cpu(),
            retain_graph=True,
        )
        dq_pt, dk_pt, dv_pt = torch.autograd.grad(
            golden_out_gpu_pt,
            (query_ref, key_ref, value_ref),
            dout.detach().cpu(),
        )
        assert_fa_close(dq_ag, dq_ref, dq_pt, name="dQ")
        assert_fa_close(dk_ag, dk_ref, dk_pt, name="dK")
        assert_fa_close(dv_ag, dv_ref, dv_pt, name="dV")

@pytest.mark.parametrize("data_type", [torch.bfloat16])
@pytest.mark.parametrize("num_heads", [32])
@pytest.mark.parametrize("kv_heads", [8])
@pytest.mark.parametrize("head_size", [35,64,101,128,151,192,201,256])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("window_size_left", [-1])
@pytest.mark.parametrize("window_size_right", [-1])
@pytest.mark.parametrize("softcap", [0.0])
@pytest.mark.parametrize("batch_size, q_seqlen, kv_seqlen", [
    (1, 256, 128),
    (1, 130, 128),
    (2, 256, 256),
    (4, 128, 256),
    (2, 128, 128),
    (1, 384, 128),
    (1, 256, 384),
    (1, 128, 128),
    (1, 256, 512),
    (1, 256, 192),
])
@pytest.mark.parametrize("num_splits", [0, 1, 2])
@pytest.mark.parametrize("cache_mode", [0, 1])
@pytest.mark.parametrize("layout", ["BSND", "TND"])
@pytest.mark.parametrize("is_causal", [True, False])
def test_fa_custom_ops_with_hd_le_256(data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, is_causal, layout, num_splits, window_size_left, window_size_right, softcap):
    is_varied = layout == 'TND'
    name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
    if "Ascend910" in name:
        pytest.skip("Sq > Sk not support in Ascend910")
    test_fa_custom_ops(data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, is_causal, layout, is_varied, window_size_left, window_size_right, num_splits)

# Copyright (c) 2026, Minghua Shen.

import torch
import torch_npu
import pytest
from tests.common.attention_ref import ref_flash_attention_pair
from tests.common.compare import assert_fa_close
from tests.common.test_utils import (
    gather_paged_kv,
    gather_paged_kv_batch,
    make_attention_inputs,
    make_block_table,
    make_cu_seqlens,
    make_golden_attention_mask,
    make_local_attention_mask,
    make_packed_random_tensor,
    make_padded_varlen_mask,
    pad_packed_tensor,
    make_random_tensor,
    make_varlen_seqlens,
)
if "Ascend950" in torch_npu.npu.get_device_name():
    from flash_attn_npu_3 import flash_attn_with_kvcache
else:
    from flash_attn_npu_3 import flash_attn_with_kvcache, flash_attn_func, flash_attn_varlen_func

test_cases = [
    # (data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode,
    #  block_size, is_causal, layout, is_varied, window_size_left, window_size_right, softcap)
    (torch.float16, 1, 8, 4, 4, 128, 64, 128, True, "BSND", False, 578, 295, 0.0),
    (torch.float16, 1, 16, 2, 8, 4096, 2, 128, True, "TND", True, 746, 16, 0.0),
    (torch.bfloat16, 4, 2, 2, 8, 2048, 16, 128, True, "TND", False, 536, 462, 0.0),
    (torch.bfloat16, 4, 32, 2, 4, 2048, 2, 128, True, "TND", False, 460, 62, 0.0),
    (torch.float16, 4, 16, 2, 4, 8192, 1, 128, False, "TND", False, 59, 571, 0.0),
    (torch.float16, 4, 2, 2, 4, 8192, 4, 128, False, "TND", True, 563, 425, 0.0),
    (torch.bfloat16, 2, 6, 2, 2, 1024, 128, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, 128, True, "TND", False, -1, -1, 0.0),
    (torch.float16, 7, 1, 1, 512, 512, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 1, 1, 1024, 1024, 128, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 1, 1, 1024, 1024, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, 128, True, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, 128, True, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 1, 1, 1, 1024, 128, 128, True, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 1, 1, 1, 1024, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 1, 1, 1, 1024, 128, 128, True, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 1, 1, 1, 1024, 128, 128, False, "BSND", False, -1, -1, 0.0),
    # kv=4096 -> 8 S2 blocks: num_splits=2 -> 2 segs (4 blk each), num_splits=4 -> 4 segs (2 blk each).
    (torch.bfloat16, 1, 1, 1, 1, 4096, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 1, 1, 1, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.float16, 2, 2, 1, 128, 128, 128, 128, True, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 6, 2, 2, 1024, 128, 128, True, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 1, 1, 16, 1024, 128, 128, False, "TND", True, -1, -1, 0.0),
    (torch.bfloat16, 2, 6, 2, 16, 1024, 128, 128, False, "TND", True, -1, -1, 0.0),
    (torch.bfloat16, 2, 6, 2, 16, 1024, 128, 128, True, "TND", True, -1, -1, 0.0),
    (torch.bfloat16, 1, 64, 1, 2, 1024, 256, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 1, 1, 16, 1024, 256, 128, True, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 1, 1, 16, 10240, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 6, 2, 16, 10240, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 6, 1, 1, 16, 10240, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 128, True, "BSND", False, 512, 0, 0.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 128, True, "TND", False, 512, 0, 0.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 128, False, "TND", False, 0, 256, 0.0),
    (torch.float16, 2, 1, 1, 512, 512, 128, 128, False, "TND", False, 508, -256, 0.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 128, False, "BSND", False, -128, 1024, 0.0),
    (torch.float16, 2, 2, 2, 512, 512, 128, 128, False, "TND", False, 64, 128, 0.0),
    # SWA + large GQA decode: rowLoopNum>1 must not hang (EVENT_ID0 order in online_softmax)
    (torch.float16, 1, 64, 1, 1, 1024, 128, 128, True, "BSND", False, 542, 647, 0.0),
    (torch.float16, 1, 128, 1, 1, 1024, 128, 128, True, "BSND", False, 542, 647, 0.0),
    (torch.float16, 1, 512, 1, 1, 1024, 128, 128, True, "BSND", False, 542, 647, 0.0),
    # FD + SWA decode (TND + paged + num_splits=2): narrow left window → early
    # FD S2 segments have empty split∩window (kvStart>=kvEnd early return; host-inited
    # partials 0/-inf must combine correctly). Needs isShortSeq/isLongSeq (B*Hk small, Sk>=1024).
    (torch.bfloat16, 1, 128, 1, 1, 1024, 128, 128, True, "TND", False, 64, 0, 0.0),
    (torch.bfloat16, 1, 32, 4, 1, 4096, 128, 128, True, "TND", False, 256, 0, 0.0),
    (torch.float16, 1, 16, 2, 1, 4096, 128, 128, True, "TND", False, 128, 0, 0.0),
    (torch.bfloat16, 1, 32, 4, 1, 8192, 128, 128, False, "TND", False, 512, 0, 0.0),
    (torch.float16, 1, 512, 1, 1, 1024, 128, 128, True, "TND", False, 542, 647, 0.0),
    # D=4 + causal (SWA hang-repro / causal ADDR_MISALIGN probe)
    (torch.float16, 1, 512, 1, 1, 1024, 4, 128, True, "BSND", False, 542, 647, 0.0),
    (torch.float16, 1, 512, 1, 1, 1024, 4, 128, True, "BSND", False, -1, -1, 0.0),

    (torch.bfloat16, 4, 32, 8, 1, 2048, 128, 128, False, "BSND", False, -1, -1, 0.0), # g=4,decode, qNBlockTile=4
    (torch.bfloat16, 8, 64, 8, 1, 4096, 128, 128, False, "BSND", False, -1, -1, 0.0), # g=8,decode, qNBlockTile=8
    (torch.bfloat16, 4, 64, 16, 16, 1024, 128, 128, True, "BSND", False, -1, -1, 0.0),# g=4,Sq=16,qNBlockTile=4
    (torch.bfloat16, 8, 128, 16, 32, 2048, 128, 128, False, "BSND", False, -1, -1, 0.0), # g=8,Sq=32,qNBlockTile=4
    (torch.bfloat16, 4, 64, 8, 64, 4096, 128, 128, False, "BSND", False, -1, -1, 0.0), # g=8,Sq=64,qNBlockTile=2
    (torch.bfloat16, 8, 128, 8, 1, 4096, 128, 128, False, "BSND", False, -1, -1, 0.0), # g=16, decode, qNBlockTile=16
    (torch.bfloat16, 4, 32, 4, 16, 1024, 256, 128, False, "BSND", False, -1, -1, 0.0), # g=8,Sq=16,D=256
    (torch.bfloat16, 8, 128, 32, 64, 2048, 128, 128, True, "BSND", False, -1, -1, 0.0),# g=4,Sq=64,qNBlockTile=2
    (torch.bfloat16, 4, 64, 4, 32, 512, 128, 128, False, "BSND", False, -1, -1, 0.0),# g=16, Sq=32,qNBlockTile=4
    (torch.bfloat16, 2, 64, 8, 1, 4096, 256, 128, False, "BSND", False, -1, -1, 0.0),# g=8,decode, D=256
    (torch.bfloat16, 4, 32, 8, 32, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),# g=4,Sq=32, qNBlockTile=4
    (torch.bfloat16, 8, 64, 8, 32, 4096, 128, 128, False, "TND", False, -1, -1, 0.0),# g=8,Sq=32, qNBlockTile=4
    (torch.bfloat16, 4, 32, 8, 64, 4096, 128, 128, False, "TND", False, -1, -1, 0.0),# g=4,Sq=64, qNBlockTile=2
    (torch.bfloat16, 8, 64, 8, 64, 4096, 128, 128, True, "TND", False, -1, -1, 0.0), # g=8,Sq=64, qNBlockTile=2
    (torch.bfloat16, 4, 64, 8, 48, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),# g=8,Sq=48, qNBlockTile=2
    (torch.bfloat16, 4, 64, 4, 32, 1024, 128, 128, False, "TND", False, -1, -1, 0.0),# g=16, Sq=32, qNBlockTile=4
    (torch.bfloat16, 4, 32, 8, 64, 2048, 256, 128, False, "TND", False, -1, -1, 0.0),# g=4,Sq=64, D=256
    (torch.bfloat16, 8, 128, 16, 32, 4096, 128, 128, False, "TND", False, -1, -1, 0.0),# g=8,Sq=32, qNBlockTile=4
    (torch.bfloat16, 1, 32, 4, 1, 2048, 128, 128, False, "TND", False, -1, -1, 0.0), # FD decode, g=8,nT=4
    (torch.bfloat16, 1, 64, 4, 1, 4096, 128, 128, False, "TND", False, -1, -1, 0.0), # FD decode, g=16, nT=4
    (torch.bfloat16, 1, 128, 4, 1, 2048, 128, 128, True, "TND", False, -1, -1, 0.0), # FD decode, g=32, nT=4
    (torch.bfloat16, 2, 32, 4, 1, 4096, 128, 128, False, "TND", False, -1, -1, 0.0), # FD decode, g=8,nT=8
    (torch.bfloat16, 2, 16, 2, 1, 2048, 128, 128, True, "TND", False, -1, -1, 0.0),# FD decode, g=8,nT=4
    (torch.bfloat16, 1, 32, 8, 1, 2048, 256, 128, False, "TND", False, -1, -1, 0.0), # FD decode, g=4,nT=8, D=256
    (torch.bfloat16, 1, 32, 4, 4, 2048, 128, 128, False, "TND", False, -1, -1, 0.0), # FD multi, g=8,Sq*g=32,nT=4
    (torch.bfloat16, 2, 16, 2, 4, 4096, 128, 128, False, "TND", False, -1, -1, 0.0), # FD multi, g=8,Sq*g=32,nT=4
    (torch.bfloat16, 1, 64, 4, 8, 2048, 128, 128, True, "TND", False, -1, -1, 0.0),# FD multi, g=16, Sq*g=128, nT=4
    (torch.bfloat16, 1, 32, 4, 16, 4096, 128, 128, False, "TND", False, -1, -1, 0.0),# FD multi, g=8,Sq*g=128, nT=4
    (torch.bfloat16, 1, 32, 4, 3, 2048, 128, 128, False, "TND", False, -1, -1, 0.0), # FD Sq=3,g=8,nT=4  [非2幂]
    (torch.bfloat16, 2, 16, 2, 5, 4096, 128, 128, True, "TND", False, -1, -1, 0.0),# FD Sq=5,g=8,nT=4  [非2幂]
    (torch.bfloat16, 1, 64, 4, 7, 2048, 128, 128, False, "TND", False, -1, -1, 0.0), # FD Sq=7,g=16, nT=4  [非2幂]
    (torch.bfloat16, 1, 32, 4, 11, 4096, 128, 128, False, "TND", False, -1, -1, 0.0),# FD Sq=11, g=8,nT=4  [非2幂]
    (torch.bfloat16, 1, 32, 8, 13, 2048, 256, 128, False, "TND", False, -1, -1, 0.0),# FD Sq=13, g=4,nT=8, D=256 [非2幂]
    (torch.bfloat16, 2, 16, 2, 15, 2048, 128, 128, True, "TND", False, -1, -1, 0.0), # FD Sq=15, g=8,nT=4  [非2幂]
    (torch.bfloat16, 4, 32, 32, 1, 2048, 128, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 8, 64, 64, 1, 4096, 128, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 4, 32, 32, 16, 1024, 128, 128, True, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 4, 64, 64, 32, 2048, 128, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 8, 16, 16, 8, 4096, 128, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 32, 32, 1, 4096, 256, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 32, 8, 65, 2048, 128, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 64, 16, 96, 2048, 128, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 128, 4096, 128, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 16, 4, 256, 2048, 128, 128, True, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 4, 65, 2048, 256, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 4, 32, 32, 32, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 4, 32, 32, 48, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 4, 64, 64, 64, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 32, 8, 65, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 64, 16, 96, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 24, 2, 6, 2048, 128, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 4, 32, 2, 6, 4096, 128, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 40, 2, 6, 2048, 128, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 24, 2, 8, 2048, 128, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 4, 32, 2, 8, 4096, 128, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 24, 2, 10, 2048, 128, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 4, 32, 2, 10, 4096, 128, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 64, 2, 10, 2048, 128, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 24, 2, 6, 2048, 256, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 4, 48, 2, 8, 4096, 128, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 48, 4, 8, 2048, 128, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 4, 64, 4, 10, 4096, 128, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 64, 4, 2, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 32, 4, 2, 4096, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 64, 8, 4, 4096, 128, 128, True, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 4, 7, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 2, 16, 2, 8, 4096, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 4, 13, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 32, 8, 16, 2048, 256, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 6, 2, 6, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 10, 2, 6, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 6, 2, 10, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 10, 2, 10, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 14, 2, 10, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 18, 2, 10, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 6, 2, 14, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 10, 2, 14, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 6, 2, 9, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 6, 2, 11, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 10, 2, 15, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 10, 2, 10, 4096, 256, 128, True, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 8, 2, 8, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 16, 2, 16, 2048, 128, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 3, 1, 128, 2048, 1, 128, False, "BSND", False, -1, -1, 0.0),
    (torch.bfloat16, 1, 3, 1, 128, 2048, 1, 128, False, "TND", False, -1, -1, 0.0),
    (torch.bfloat16, 8, 1024, 16, 8, 640, 1, 128, False, "BSND", False, -1, -1, 0.0),
    # Softcap
    (torch.bfloat16, 4, 64, 4, 32, 512, 128, 128, False, "BSND", False, -1, -1, 30.0),# g=16, Sq=32,qNBlockTile=4
    (torch.bfloat16, 2, 64, 8, 1, 4096, 256, 128, False, "BSND", False, -1, -1, 30.0),# g=8,decode, D=256
    (torch.bfloat16, 1, 32, 4, 1, 2048, 128, 128, False, "TND", False, -1, -1, 30.0), # FD decode, g=8,nT=4
    (torch.bfloat16, 1, 32, 8, 1, 2048, 256, 128, False, "TND", False, -1, -1, 30.0), # FD decode, g=4,nT=8, D=256
    (torch.bfloat16, 1, 32, 4, 4, 2048, 128, 128, False, "TND", False, -1, -1, 30.0), # FD multi, g=8,Sq*g=32,nT=4
    (torch.bfloat16, 1, 64, 4, 8, 2048, 128, 128, True, "TND", False, -1, -1, 30.0),# FD multi, g=16, Sq*g=128, nT=4
    (torch.bfloat16, 1, 32, 8, 13, 2048, 256, 128, False, "TND", False, -1, -1, 30.0),# FD Sq=13, g=4,nT=8, D=256 [非2幂]
    (torch.bfloat16, 1, 32, 8, 128, 4096, 128, 128, False, "BSND", False, -1, -1, 30.0),
    (torch.bfloat16, 2, 16, 4, 256, 2048, 128, 128, True, "BSND", False, -1, -1, 30.0),
    (torch.bfloat16, 1, 32, 4, 65, 2048, 256, 128, False, "BSND", False, -1, -1, 30.0),
    (torch.bfloat16, 4, 32, 32, 32, 2048, 128, 128, False, "TND", False, -1, -1, 30.0),
    (torch.bfloat16, 4, 32, 32, 48, 2048, 128, 128, False, "TND", False, -1, -1, 30.0),
    (torch.bfloat16, 4, 64, 64, 64, 2048, 128, 128, False, "TND", False, -1, -1, 30.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 128, True, "BSND", False, 512, 0, 30.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 128, True, "TND", False, 512, 0, 30.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 128, False, "TND", False, 0, 256, 30.0),
    (torch.float16, 2, 1, 1, 512, 512, 128, 128, False, "TND", False, 508, -256, 30.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 128, False, "BSND", False, -128, 1024, 30.0),
    (torch.float16, 2, 2, 2, 512, 512, 128, 128, False, "TND", False, 64, 128, 30.0),
    (torch.bfloat16, 1, 3, 1, 128, 2048, 1, 128, False, "BSND", False, -1, -1, 30.0),
    (torch.bfloat16, 1, 3, 1, 128, 2048, 1, 128, False, "TND", False, -1, -1, 30.0),
    (torch.bfloat16, 8, 1024, 16, 8, 640, 1, 128, False, "BSND", False, -1, -1, 30.0),
]

@pytest.mark.parametrize("num_splits", [0, 2])
@pytest.mark.parametrize("cache_mode", [0, 1])
@pytest.mark.parametrize("data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, block_size, is_causal, layout, is_varied, window_size_left, window_size_right, softcap", test_cases)
def test_fa_custom_ops(data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, is_causal, layout, is_varied, num_splits, window_size_left, window_size_right, softcap):
    name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
    if num_splits > 1 and not (cache_mode == 1 and layout == "TND"):
        pytest.skip("num_splits>1 requires paged KV cache and TND (varlen-q) layout")
    if "Ascend950" in name and num_splits > 1:
        pytest.skip("Ascend950 does not support num_splits>1")
    if "Ascend950" in name and not (1 <= head_size <= 256):
        pytest.skip("Ascend950 supports head_size in [1, 256]")
    if "Ascend950" in name and (window_size_right != -1 or window_size_left != -1):
        pytest.skip("Ascend950 support SWA")
    if "Ascend950" in name and (softcap != 0.0):
        pytest.skip("Ascend950 support softcap")
    if is_varied and layout != "TND":
        pytest.skip("is_varied requires TND (varlen-q) layout")
    block_size = 128
    max_num_blocks_per_seq = (kv_seqlen + block_size - 1) // block_size
    num_blocks = max(64, max_num_blocks_per_seq * batch_size)
    gen = torch.Generator().manual_seed(1234)
    if is_varied:
        q_sequences, kv_sequences = make_varlen_seqlens(batch_size, q_seqlen, kv_seqlen, seed=1234)
    else:
        q_sequences = [q_seqlen] * batch_size
        kv_sequences = [kv_seqlen] * batch_size
    t_q_sum = sum(q_sequences)
    t_kv_sum = sum(kv_sequences)
    if layout == "BSND":
        query = make_random_tensor((batch_size, q_seqlen, num_heads, head_size), data_type, generator=gen, device="npu")
    elif layout == "TND":
        query = make_packed_random_tensor(q_sequences, q_seqlen, num_heads, head_size, data_type, generator=gen, device="npu")
    key_cache = None
    value_cache = None
    block_tables = None
    if cache_mode == 1:
        key_cache = make_random_tensor((num_blocks, block_size, kv_heads, head_size), data_type, generator=gen, device="npu")
        value_cache = make_random_tensor((num_blocks, block_size, kv_heads, head_size), data_type, generator=gen, device="npu")
        block_tables = make_block_table(batch_size, kv_seqlen, block_size).npu()
    else:
        if layout == "BSND":
            key_cache = make_random_tensor((batch_size, kv_seqlen, kv_heads, head_size), data_type, generator=gen, device="npu")
            value_cache = make_random_tensor((batch_size, kv_seqlen, kv_heads, head_size), data_type, generator=gen, device="npu")
        else:
            key_cache = make_packed_random_tensor(kv_sequences, kv_seqlen, kv_heads, head_size, data_type, generator=gen, device="npu")
            value_cache = make_packed_random_tensor(kv_sequences, kv_seqlen, kv_heads, head_size, data_type, generator=gen, device="npu")
        block_tables = None
    if layout == "BSND":
        q_seqlen_list = [q_seqlen] * batch_size
        kv_seqlen_list = [kv_seqlen] * batch_size
    else:
        q_seqlen_list = q_sequences
        kv_seqlen_list = kv_sequences
    scale = 1.0 / (head_size ** 0.5)
    is_rotary_interleaved = False
    kv_seqlen_list = torch.tensor(kv_seqlen_list, dtype=torch.int32).npu()
    rotary_cos = None
    rotary_sin = None
    cache_batch_idx = None
    leftpad_k = None
    alibi_slopes = None
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
    out_out, softmax_lse, *rest = flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        None,
        None,
        None,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        cache_seqlens=kv_seqlen_list,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=leftpad_k,
        page_table=block_tables,
        cu_seqlens_q=new_q_seqlen_list,
        cu_seqlens_k_new=None,
        max_seqlen_q=q_seqlen,
        rotary_seqlens=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        softmax_scale=None,
        causal=is_causal,
        window_size=[window_size_left, window_size_right],
        attention_chunk=0,
        softcap=softcap,
        rotary_interleaved=is_rotary_interleaved,
        scheduler_metadata=None,
        num_splits=num_splits,
        pack_gqa=None,
        sm_margin=0,
        return_softmax_lse=True
    )

    golden_out_ref = None
    golden_out_pt = None
    if layout == "BSND":
        golden_out_ref = torch.empty((batch_size, q_seqlen, num_heads, head_size), dtype=data_type)
        golden_out_pt = torch.empty_like(golden_out_ref)
        golden_lseL_ref = torch.empty((batch_size, num_heads, q_seqlen), dtype=torch.float32)
        golden_lseL_pt = torch.empty_like(golden_lseL_ref)
    else:
        golden_out_ref = torch.empty((t_q_sum, num_heads, head_size), dtype=data_type)
        golden_out_pt = torch.empty_like(golden_out_ref)
        golden_lseL_ref = torch.empty((num_heads, t_q_sum), dtype=torch.float32)
        golden_lseL_pt = torch.empty_like(golden_lseL_ref)
    query_cpu = query.detach().cpu()
    key_cache_cpu = key_cache.detach().cpu()
    value_cache_cpu = value_cache.detach().cpu()
    block_tables_cpu = block_tables.cpu() if cache_mode == 1 else None
    if layout == "BSND":
        atten_mask = None
        if is_causal_golden:
            atten_mask = torch.triu(
                torch.ones(q_seqlen, kv_seqlen),
                diagonal=(kv_seqlen - q_seqlen + 1),
            ).bool()
        elif is_local_golden:
            atten_mask = make_local_attention_mask(
                q_seqlen,
                kv_seqlen,
                window_size_left_golden,
                window_size_right_golden,
            )
        if cache_mode == 1:
            key_batched, value_batched = gather_paged_kv_batch(
                key_cache_cpu, value_cache_cpu, block_tables_cpu, kv_seqlen, block_size
            )
        else:
            key_batched, value_batched = key_cache_cpu, value_cache_cpu
        golden_out_ref, golden_lseL_ref, golden_out_pt, golden_lseL_pt = ref_flash_attention_pair(
            query_cpu, key_batched, value_batched, scale, atten_mask, data_type, softcap
        )
        if atten_mask is not None:
            fully_masked = atten_mask.all(dim=-1)
            golden_out_ref[:, fully_masked] = 0
            golden_out_pt[:, fully_masked] = 0
            golden_lseL_ref[:, :, fully_masked] = torch.inf
            golden_lseL_pt[:, :, fully_masked] = torch.inf
        assert_fa_close(out_out, golden_out_ref, golden_out_pt, softcap=softcap, name="out")
        if "Ascend910" in name:
            assert_fa_close(softmax_lse, golden_lseL_ref, golden_lseL_pt, softcap=softcap, name="softmax_lse")
        return
    query_padded = pad_packed_tensor(query_cpu, q_sequences, q_seqlen)
    if cache_mode == 1:
        key_padded, value_padded = gather_paged_kv_batch(
            key_cache_cpu, value_cache_cpu, block_tables_cpu, kv_seqlen, block_size
        )
    else:
        key_padded = pad_packed_tensor(key_cache_cpu, kv_sequences, kv_seqlen)
        value_padded = pad_packed_tensor(value_cache_cpu, kv_sequences, kv_seqlen)
    q_valid = torch.arange(q_seqlen) < torch.tensor(q_sequences)[:, None]
    k_valid = torch.arange(kv_seqlen) < torch.tensor(kv_sequences)[:, None]
    atten_mask = (~q_valid[:, :, None]) | (~k_valid[:, None, :])
    row = torch.arange(q_seqlen)[None, :, None]
    col = torch.arange(kv_seqlen)[None, None, :]
    if is_causal_golden:
        atten_mask = atten_mask | (col - row >= (torch.tensor(kv_sequences) - torch.tensor(q_sequences))[:, None, None] + 1)
    elif is_local_golden:
        diff = col - row
        left = (torch.tensor(kv_sequences) - torch.tensor(q_sequences))[:, None, None] - window_size_left_golden
        right = (torch.tensor(kv_sequences) - torch.tensor(q_sequences))[:, None, None] + window_size_right_golden
        atten_mask = atten_mask | (diff < left) | (diff > right)
    golden_out_ref, golden_lse_ref, golden_out_pt, golden_lse_pt = ref_flash_attention_pair(
        query_padded, key_padded, value_padded, scale, atten_mask, data_type, softcap
    )
    fully_masked = atten_mask.all(dim=-1)
    golden_out_ref[fully_masked] = 0
    golden_out_pt[fully_masked] = 0
    golden_lse_ref = golden_lse_ref.masked_fill(fully_masked[:, None, :], torch.inf)
    golden_lse_pt = golden_lse_pt.masked_fill(fully_masked[:, None, :], torch.inf)
    golden_out_ref = golden_out_ref[q_valid]
    golden_out_pt = golden_out_pt[q_valid]
    golden_lse_ref = golden_lse_ref.permute(0, 2, 1)[q_valid].transpose(0, 1)
    golden_lse_pt = golden_lse_pt.permute(0, 2, 1)[q_valid].transpose(0, 1)
    assert_fa_close(out_out, golden_out_ref, golden_out_pt, softcap=softcap, name="out")
    if "Ascend910" in name:
        assert_fa_close(softmax_lse, golden_lse_ref, golden_lse_pt, softcap=softcap, name="softmax_lse")
    return

func_cases = [
    # (data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, return_attn_probs, is_causal, window_size_left, window_size_right, softcap)
    (torch.float16, 1, 1, 1, 1024, 1024, 128, True, False, -1, -1, 0.0),
    (torch.float16, 5, 4, 4, 1024, 1024, 128, True, True, -1, -1, 0.0),
    (torch.float16, 7, 1, 1, 512, 512, 128, True, False, -1, -1, 0.0),
    (torch.float16, 1, 1, 1, 1024, 1024, 128, False, False, -1, -1, 0.0),
    (torch.float16, 5, 4, 4, 1024, 1024, 128, False, True, -1, -1, 0.0),
    (torch.float16, 7, 1, 1, 512, 512, 128, False, False, -1, -1, 0.0),
    (torch.float16, 4, 2, 1, 513, 513, 128, False, False, -1, -1, 0.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, True, False, -1, -1, 0.0),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, True, True, -1, -1, 0.0),
    (torch.bfloat16, 7, 1, 1, 512, 512, 128, True, False, -1, -1, 0.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, False, False, -1, -1, 0.0),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, False, True, -1, -1, 0.0),
    (torch.bfloat16, 7, 1, 1, 512, 512, 128, False, False, -1, -1, 0.0),
    (torch.bfloat16, 4, 2, 1, 513, 513, 128, True, False, -1, -1, 0.0),
    (torch.float16, 1, 1, 1, 1024, 1024, 128, True, False, -1, -1, 30.0),
    (torch.float16, 5, 4, 4, 1024, 1024, 128, True, True, -1, -1, 30.0),
    (torch.float16, 7, 1, 1, 512, 512, 128, False, False, -1, -1, 30.0),
    (torch.float16, 4, 2, 1, 513, 513, 128, False, False, -1, -1, 30.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, True, False, -1, -1, 30.0),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, True, True, -1, -1, 30.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, False, False, -1, -1, 30.0),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, False, True, -1, -1, 30.0),
    (torch.bfloat16, 7, 1, 1, 512, 512, 128, False, False, -1, -1, 30.0),
    (torch.bfloat16, 4, 2, 1, 513, 513, 128, True, False, -1, -1, 30.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, True, True, 512, 0, 0.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, True, True, 512, 256, 0.0),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, True, True, -128, 864, 0.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, True, False, 0, 256, 0.0),
    (torch.float16, 2, 2, 2, 512, 512, 128, True, False, 64, 128, 0.0),
    (torch.bfloat16, 1, 1, 1, 1, 1024, 128, True, True, 512, 0, 0.0),
    (torch.bfloat16, 2, 6, 2, 2, 1024, 128, True, True, -1, -1, 0.0),
    (torch.bfloat16, 2, 6, 2, 2, 1024, 128, True, True, 256, 0, 0.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, True, True, 512, 0, 30.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, True, True, 512, 256, 30.0),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, True, True, -128, 864, 30.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, True, False, 0, 256, 30.0),
    (torch.float16, 2, 2, 2, 512, 512, 128, True, False, 64, 128, 30.0),
    (torch.bfloat16, 1, 1, 1, 1, 1024, 128, True, True, 512, 0, 30.0),
    (torch.bfloat16, 2, 6, 2, 2, 1024, 128, True, True, -1, -1, 30.0),
    (torch.bfloat16, 2, 6, 2, 2, 1024, 128, True, True, 256, 0, 30.0),
]


@pytest.mark.parametrize("data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, return_attn_probs, is_causal, window_size_left, window_size_right, softcap", func_cases)
def test_fa_func_ops(data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, return_attn_probs, is_causal, window_size_left, window_size_right, softcap):
    name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
    if "Ascend910" not in name:
        pytest.skip("flash_attn_func only support Ascend910")
    query, key_cache, value_cache, dout = make_attention_inputs(
        (batch_size, q_seqlen, num_heads, head_size),
        (batch_size, kv_seqlen, kv_heads, head_size),
        (batch_size, kv_seqlen, kv_heads, head_size),
        (batch_size, q_seqlen, num_heads, head_size),
        data_type,
        device="npu",
    )
    scale = 1.0 / (head_size ** 0.5)
    ret = flash_attn_func(
        query,
        key_cache,
        value_cache,
        softmax_scale=scale,
        causal=is_causal,
        window_size=[window_size_left, window_size_right],
        softcap=softcap,
        return_attn_probs=return_attn_probs)
    if not return_attn_probs:
        out_out = ret
    else:
        out_out, softmax_lse = ret

    query_ref = query.detach().cpu().requires_grad_(True)
    key_ref = key_cache.detach().cpu().requires_grad_(True)
    value_ref = value_cache.detach().cpu().requires_grad_(True)
    atten_mask, _, _ = make_golden_attention_mask(
        q_seqlen,
        kv_seqlen,
        is_causal,
        window_size_left,
        window_size_right,
    )
    golden_out_ref, golden_lseL_ref, golden_out_pt, golden_lseL_pt = ref_flash_attention_pair(
        query_ref, key_ref, value_ref, scale, atten_mask, data_type, softcap
    )
    if atten_mask is not None:
        fully_masked = atten_mask.all(dim=-1)
        golden_out_ref[:, fully_masked] = 0
        golden_out_pt[:, fully_masked] = 0
        golden_lseL_ref[:, :, fully_masked] = torch.inf
        golden_lseL_pt[:, :, fully_masked] = torch.inf

    assert_fa_close(out_out, golden_out_ref, golden_out_pt, softcap=softcap, name="out")
    if return_attn_probs:
        assert_fa_close(
            softmax_lse,
            golden_lseL_ref,
            golden_lseL_pt,
            softcap=softcap,
            name="softmax_lse",
        )
    dq_ag, dk_ag, dv_ag = torch.autograd.grad(out_out, (query, key_cache, value_cache), dout)
    dout_ref = dout.detach().cpu()
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        golden_out_ref,
        (query_ref, key_ref, value_ref),
        dout_ref,
        retain_graph=True,
    )
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(
        golden_out_pt,
        (query_ref, key_ref, value_ref),
        dout_ref,
    )
    assert_fa_close(dq_ag, dq_ref, dq_pt, softcap=softcap, name="dQ")
    assert_fa_close(dk_ag, dk_ref, dk_pt, softcap=softcap, name="dK")
    assert_fa_close(dv_ag, dv_ref, dv_pt, softcap=softcap, name="dV")


varlen_base_cases = [
    # (data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, window_size_left, window_size_right, softcap)
    (torch.bfloat16, 1, 1, 1, 512, 1024, 128, True, -1, -1, 0.0),
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, False, -1, -1, 0.0),
    (torch.float16, 7, 5, 1, 512, 512, 128, True, -1, -1, 0.0),
    (torch.float16, 7, 5, 1, 777, 888, 192, False, -1, -1, 0.0),
    (torch.float16, 7, 5, 1, 1777, 1888, 256, True, -1, -1, 0.0),
    (torch.bfloat16, 1, 1, 1, 7777, 8192, 64, True, -1, -1, 0.0),
    (torch.bfloat16, 7, 5, 1, 711, 8192, 111, True, -1, -1, 0.0),
    # SWA
    (torch.bfloat16, 1, 1, 1, 512, 512, 128, True, 512, 0, 0.0),
    (torch.bfloat16, 1, 1, 1, 512, 512, 128, True, 256, 128, 0.0),
    (torch.float16, 2, 4, 4, 256, 256, 128, False, 64, 128, 0.0),
    (torch.bfloat16, 1, 1, 1, 512, 512, 128, False, 0, 256, 0.0),
    (torch.bfloat16, 2, 6, 2, 128, 256, 128, True, 127, 0, 0.0),
    (torch.bfloat16, 2, 4, 4, 128, 512, 128, True, 511, 0, 0.0),
    (torch.float16, 1, 2, 2, 64, 192, 128, False, 32, 64, 0.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, True, 512, 0, 0.0),
    (torch.bfloat16, 1, 1, 1, 1, 1024, 128, True, 512, 0, 0.0),
    (torch.float16, 2, 1, 1, 512, 512, 128, False, 508, -256, 0.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, True, -128, 864, 0.0),
    (torch.bfloat16, 2, 6, 2, 2, 1024, 128, True, 256, 0, 0.0),
    # SWA + large GQA decode (EVENT_ID0 / rowLoopNum>1 hang regression)
    (torch.float16, 1, 64, 1, 1, 1024, 128, True, 542, 647, 0.0),
    (torch.float16, 1, 128, 1, 1, 1024, 128, True, 542, 647, 0.0),
    (torch.float16, 1, 512, 1, 1, 1024, 128, True, 542, 647, 0.0),
    (torch.bfloat16, 1, 128, 1, 1, 1024, 128, True, 64, 0, 0.0),
    # Softcap
    (torch.float16, 7, 5, 1, 777, 888, 192, False, -1, -1, 30.0),
    (torch.float16, 7, 5, 1, 1777, 1888, 256, True, -1, -1, 30.0),
    (torch.bfloat16, 1, 1, 1, 7777, 8192, 64, True, -1, -1, 30.0),
    (torch.bfloat16, 7, 5, 1, 711, 8192, 111, True, -1, -1, 30.0),
    (torch.float16, 1, 2, 2, 64, 192, 128, False, 32, 64, 30.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, True, 512, 0, 30.0),
    (torch.bfloat16, 1, 1, 1, 1, 1024, 128, True, 512, 0, 30.0),
    (torch.float16, 2, 1, 1, 512, 512, 128, False, 508, -256, 30.0),
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, True, -128, 864, 30.0),
    (torch.bfloat16, 2, 6, 2, 2, 1024, 128, True, 256, 0, 30.0),
]

varlen_cases = [(*case, False) for case in varlen_base_cases] + [
    # 旧 varlen 反向覆盖：每个 batch 使用不同 q/k 长度。
    (torch.bfloat16, 3, 1, 1, 512, 1024, 128, True, -1, -1, 0.0, True),
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, False, -1, -1, 0.0, True),
    (torch.float16, 7, 5, 1, 512, 512, 128, True, -1, -1, 0.0, True),
    (torch.float16, 7, 5, 1, 777, 888, 192, False, -1, -1, 0.0, True),
    (torch.float16, 7, 5, 1, 1777, 1888, 256, True, -1, -1, 0.0, True),
    (torch.bfloat16, 3, 1, 1, 7777, 8192, 64, True, -1, -1, 0.0, True),
    (torch.bfloat16, 7, 5, 1, 711, 8192, 111, True, -1, -1, 0.0, True),
    (torch.bfloat16, 3, 16, 16, 562, 562, 96, False, -1, -1, 0.0, True),
    (torch.bfloat16, 3, 1, 1, 512, 1024, 128, True, -1, -1, 30.0, True),
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, False, -1, -1, 30.0, True),
    (torch.float16, 7, 5, 1, 512, 512, 128, True, -1, -1, 30.0, True),
    (torch.float16, 7, 5, 1, 777, 888, 192, False, -1, -1, 30.0, True),
    (torch.float16, 7, 5, 1, 1777, 1888, 256, True, -1, -1, 30.0, True),
    (torch.bfloat16, 3, 1, 1, 7777, 8192, 64, True, -1, -1, 30.0, True),
    (torch.bfloat16, 7, 5, 1, 711, 8192, 111, True, -1, -1, 30.0, True),
    (torch.bfloat16, 3, 16, 16, 562, 562, 96, False, -1, -1, 30.0, True),
    (torch.bfloat16, 3, 4, 4, 1024, 1024, 128, True, 512, 0, 0.0, True),
    (torch.bfloat16, 3, 1, 1, 512, 1024, 128, True, 512, 0, 0.0, True),
    (torch.bfloat16, 3, 1, 1, 512, 1024, 128, False, 0, 256, 0.0, True),
    (torch.float16, 3, 2, 2, 512, 512, 128, False, 64, 128, 0.0, True),
    (torch.bfloat16, 3, 4, 4, 1024, 1024, 128, True, -128, 864, 0.0, True),
    (torch.bfloat16, 3, 4, 4, 1024, 1024, 128, True, 512, 0, 30.0, True),
    (torch.bfloat16, 3, 1, 1, 512, 1024, 128, True, 512, 0, 30.0, True),
    (torch.bfloat16, 3, 1, 1, 512, 1024, 128, False, 0, 256, 30.0, True),
    (torch.float16, 3, 2, 2, 512, 512, 128, False, 64, 128, 30.0, True),
    (torch.bfloat16, 3, 4, 4, 1024, 1024, 128, True, -128, 864, 30.0, True),
]


@pytest.mark.parametrize("data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, window_size_left, window_size_right, softcap, is_varied", varlen_cases)
def test_fa_varlen_ops(data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal, window_size_left, window_size_right, softcap, is_varied):
    name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
    if "Ascend910" not in name:
        pytest.skip("flash_attn_varlen_func only support Ascend910")
    if is_varied:
        seqlens_q, seqlens_k = make_varlen_seqlens(batch_size, q_seqlen, kv_seqlen)
    else:
        seqlens_q = [q_seqlen] * batch_size
        seqlens_k = [kv_seqlen] * batch_size
    cu_q = make_cu_seqlens(seqlens_q)
    cu_k = make_cu_seqlens(seqlens_k)
    total_q = int(cu_q[-1].item())
    total_k = int(cu_k[-1].item())
    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)
    query = make_packed_random_tensor(seqlens_q, max_seqlen_q, num_heads, head_size, data_type, device="npu", requires_grad=True)
    key = make_packed_random_tensor(seqlens_k, max_seqlen_k, kv_heads, head_size, data_type, device="npu", requires_grad=True)
    value = make_packed_random_tensor(seqlens_k, max_seqlen_k, kv_heads, head_size, data_type, device="npu", requires_grad=True)
    actual_seq_len = cu_q.npu()
    actual_kv_len = cu_k.npu()

    scale = 1.0 / (head_size ** 0.5)

    output_npu, softmax_lse = flash_attn_varlen_func(
        query,
        key,
        value,
        actual_seq_len,
        actual_kv_len,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=scale,
        causal=is_causal,
        window_size=(window_size_left, window_size_right),
        softcap=softcap,
        return_attn_probs=True,
    )
    query_ref = query.detach().cpu().requires_grad_(True)
    key_ref = key.detach().cpu().requires_grad_(True)
    value_ref = value.detach().cpu().requires_grad_(True)
    query_padded = pad_packed_tensor(query_ref, seqlens_q, max_seqlen_q)
    key_padded = pad_packed_tensor(key_ref, seqlens_k, max_seqlen_k)
    value_padded = pad_packed_tensor(value_ref, seqlens_k, max_seqlen_k)
    q_valid, k_valid, atten_mask = make_padded_varlen_mask(
        seqlens_q,
        seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        is_causal,
        window_size_left,
        window_size_right,
    )
    golden_out_ref, golden_lse_ref, golden_out_pt, golden_lse_pt = ref_flash_attention_pair(
        query_padded, key_padded, value_padded, scale, atten_mask, data_type, softcap
    )
    fully_masked = atten_mask.all(dim=-1)
    golden_out_ref[fully_masked] = 0
    golden_out_pt[fully_masked] = 0
    golden_lse_ref = golden_lse_ref.masked_fill(fully_masked[:, None, :], torch.inf)
    golden_lse_pt = golden_lse_pt.masked_fill(fully_masked[:, None, :], torch.inf)
    golden_out_ref = golden_out_ref[q_valid]
    golden_out_pt = golden_out_pt[q_valid]
    golden_lseL_ref = golden_lse_ref.permute(0, 2, 1)[q_valid].transpose(0, 1)
    golden_lseL_pt = golden_lse_pt.permute(0, 2, 1)[q_valid].transpose(0, 1)
    assert_fa_close(output_npu, golden_out_ref, golden_out_pt, softcap=softcap, name="out")
    assert_fa_close(softmax_lse, golden_lseL_ref, golden_lseL_pt, softcap=softcap, name="softmax_lse")
    dout = make_random_tensor(output_npu.shape, output_npu.dtype, low=-0.5, high=0.5, device="npu")
    dq_ag, dk_ag, dv_ag = torch.autograd.grad(output_npu, (query, key, value), dout)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        golden_out_ref,
        (query_ref, key_ref, value_ref),
        dout.detach().cpu(),
        retain_graph=True,
    )
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(
        golden_out_pt,
        (query_ref, key_ref, value_ref),
        dout.detach().cpu(),
    )
    assert_fa_close(dq_ag, dq_ref, dq_pt, softcap=softcap, name="dQ")
    assert_fa_close(dk_ag, dk_ref, dk_pt, softcap=softcap, name="dK")
    assert_fa_close(dv_ag, dv_ref, dv_pt, softcap=softcap, name="dV")

@pytest.mark.parametrize("data_type", [torch.float16])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("kv_heads", [2])
@pytest.mark.parametrize("head_size", [35,64,101,128,151,192,201,256])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("window_size_left", [-1])
@pytest.mark.parametrize("window_size_right", [-1])
@pytest.mark.parametrize("softcap", [0.0])
@pytest.mark.parametrize("batch_size, q_seqlen, kv_seqlen", [
    (1, 256, 128),
    (1, 136, 128),
    (2, 256, 256),
    (4, 128, 256),
    (2, 128, 128),
    (1, 1024, 128),
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

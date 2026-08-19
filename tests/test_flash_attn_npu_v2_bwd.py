# Copyright (c) 2026, Minghua Shen.
"""
FlashAttention v2 反向 pytest。

- 正向：flash_attn_func / flash_attn_varlen_func 得到 out / softmax_lse，不与标杆比较
- 标杆：小算子 fa_small_op_golden（golden_*_bwd_from_fwd，传入 FA out/lse，仅算反传）
- 被测：torch.autograd.grad（FlashAttnFunc / FlashAttnVarlenFunc）

用法:
  pytest tests/test_flash_attn_npu_v2_bwd.py -v
  pytest tests/test_flash_attn_npu_v2_bwd.py -k swa -v
"""

import gc
import os
import random
import sys

import pytest
import torch
import torch_npu

if "Ascend950" in (torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""):
    pytest.skip("flash_attn_npu (v2) not supported on Ascend950", allow_module_level=True)

from flash_attn_npu import flash_attn_func, flash_attn_varlen_func

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)


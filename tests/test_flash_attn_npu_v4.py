# Copyright (c) 2026, Minghua Shen.

import torch
import torch_npu
import pytest
from npu_precision_utils import compare_rule
from tests.common.attention_ref import ref_flash_attention, ref_masked_attention
if "Ascend950" in torch_npu.npu.get_device_name():
    from flash_attn_npu_4 import flash_attn_varlen_func
else:
    from flash_attn_npu_4 import flash_attn_varlen_func


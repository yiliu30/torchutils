import os
import gc
import functools
import habana_frameworks.torch.internal.bridge_config as bc

os.environ["PT_HPU_LAZY_MODE"] = "1"
os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "1"

import neural_compressor as inc
from neural_compressor.torch.algorithms.fp8_quant._core.quantized_func_wrappers import (
    init_quantized_func_wrapper_factory,
    clear_quantized_func_wrapper_factory,
)

init_quantized_func_wrapper_factory()

# os.environ['HABANA_PROFILE'] = '1'
# os.environ['GRAPH_VISUALIZATION'] = '1'
import torch.distributed as dist
import torch
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.internal.bridge_config as bc

# from vllm_hpu_extension.profiler import (HabanaMemoryProfiler, format_bytes)


from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu


import os

os.environ["PT_HPU_LAZY_MODE"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import habana_frameworks.torch as ht
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
import time
import argparse
import torch.nn.functional as F

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
# if is_hpu_gaudi2:
FP8_MAX = torch.finfo(torch.float8_e4m3fnuz).max

def dynamic_quant(data, single_scale = False, dim=-1):
    if single_scale:
        scale = ((torch.abs(data)).max() + 1e-8) / FP8_MAX
    else:
        scale = ((torch.abs(data)).max(dim=dim).values + 1e-8) / FP8_MAX
        scale = scale.unsqueeze(dim)
    data_fp8 = torch.ops.hpu.cast_to_fp8_v2(
        data, 1.0 / scale, False, False, torch.float8_e4m3fn)[0]
    return data_fp8, scale.float()

def log_tensor(name, tensor):
    print(f"{name}: shape {tensor.shape}, dtype {tensor.dtype}, device {tensor.device}")
    

def test():
    with torch.no_grad(), torch.autocast(device_type="hpu", dtype=torch.bfloat16):
        num_experts = 32
        num_tokens, in_features, out_features = 2048, 4096, 1024
        input = torch.randn(num_tokens, in_features).to("hpu").to(torch.bfloat16)
        weight = torch.randn(in_features, out_features).to("hpu").to(torch.bfloat16)
        ref_res = torch.matmul(input, weight)
        input_quant, input_scale = dynamic_quant(input, single_scale=False, dim=-1)
        weight_quant, weight_scale = dynamic_quant(weight, single_scale=False, dim=0)
        log_tensor("input", input)
        log_tensor("weight", weight)
        log_tensor("input_quant", input_quant)
        log_tensor("weight_quant", weight_quant)
        log_tensor("input_scale", input_scale)
        log_tensor("weight_scale", weight_scale)
        res = torch.ops.hpu.fp8_gemm_v2(
            input_quant,
            False,
            weight_quant,
            False,
            None,
            torch.bfloat16,
            input_scale,
            weight_scale,
        )
        print(f"res shape: {res.shape}, range {res.min()} - {res.max()}")
        diff = ref_res - res
        print(f"diff range {diff.min()} - {diff.max()}, mse {torch.mean(diff * diff)}")

if __name__ == "__main__":
    test()

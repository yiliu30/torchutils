import os

# TORCH_CUDA_ARCH_LIST=10.0
# os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
import argparse
import time
from pathlib import Path
import copy

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.cpp_extension import load
from triton.testing import do_bench

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

CURRENT_DIR = Path(__file__).parent


# CURRENT_DIR = Path(__file__).parent
# Only load cuda-related source files
sources = (
    [str(p) for p in CURRENT_DIR.glob("attention*.cu")]
    + [str(p) for p in CURRENT_DIR.glob("attention*.cpp")]
    + [str(p) for p in CURRENT_DIR.glob("attention*.cc")]
    + [str(p) for p in CURRENT_DIR.glob("attention*.c")]
)


module = load(
    "my_ext",
    # sources=list(CURRENT_DIR.glob("attention*")),
    sources=sources,
    # !!!!!!! Gen PTX and SASS with source line info for better debugging !!!!!!!
    extra_cuda_cflags=[
        "-lineinfo",
        "--ptxas-options=-v",
        "-arch=compute_80",
        "-code=sm_80,compute_80",
        "--generate-line-info",
    ],
    verbose=True,
)

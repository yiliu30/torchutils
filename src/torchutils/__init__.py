#   -------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------
"""Python Package Template"""
from __future__ import annotations

__version__ = "0.0.2"


# from torchutils.bench import bench_module, bench_more, inspect_tensor, see_memory_usage


from torchutils.freeze import freeze_seed


def log_info():
    import logging

    logging.basicConfig(level=logging.INFO)
    logging.info("Set logging level to INFO successfully")

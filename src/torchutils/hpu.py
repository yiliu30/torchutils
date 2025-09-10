import logging
from logging import logger

def init_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

def rank_debug(msg, level="info", target_rank=None):
    import torch
    fn = getattr(logger, level)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    if target_rank is not None:
        if rank == target_rank:
            fn(f"[Rank {rank}] {msg}")
    else:
        fn(f"[Rank {rank}] {msg}")


def show_mem_info(logger=None, msg="", loglevel="info"):
    import torch
    if logger is None:
        logger = init_logger(__name__)
    hpu_mem_mb = get_used_hpu_mem_MB()
    show_fn = getattr(logger, loglevel)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    if rank == 0:
        show_fn(f"[Rank {rank}] {msg}")
        cpu_mem_mb = get_used_cpu_mem_MB()
        show_fn(f"[Rank {rank}] Used HPU: {hpu_mem_mb // 1000} GB {hpu_mem_mb % 1000:.2f} MB; CPU: {cpu_mem_mb // 1000} GB {cpu_mem_mb % 1000:.2f} MB")
    

def get_used_hpu_mem_MB():
    """Get HPU used memory: MiB."""
    import torch
    import numpy as np
    import habana_frameworks.torch as htorch
    from habana_frameworks.torch.hpu import memory_stats
    htorch.core.mark_step()
    torch.hpu.synchronize()
    mem_stats = memory_stats()
    used_hpu_mem = np.round(mem_stats["InUse"] / 1024**2, 3)
    return used_hpu_mem


def get_used_cpu_mem_MB():
    """Get the amount of CPU memory used by the current process in MiB (Mebibytes)."""
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    used_cpu_mem = round(mem_info.rss / 1024**2, 3)
    return used_cpu_mem
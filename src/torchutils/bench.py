seed = 0


import os

DEBUG = os.environ.get("DEBUG", "0") == "1"

from triton.testing import do_bench
import torch
import time

from torchutils.freeze import freeze_seed
from torchutils.config import bench_config
from collections import defaultdict
import functools

freeze_seed()


execution_records = defaultdict(list)

def dump_elapsed_time(customized_msg="", record=False, record_only=False):
    """Get the elapsed time for decorated functions.

    Args:
        customized_msg (string, optional): The parameter passed to decorator. Defaults to None.
    """
    

    def f(func):
        @functools.wraps(func)
        def fi(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            dur = round((end - start) * 1000, 2)
            if record:
                execution_records[customized_msg if customized_msg else func.__qualname__].append(dur)
            if not record_only:
                print(
                    "%s elapsed time: %s ms"
                    % (
                        customized_msg if customized_msg else func.__qualname__,
                        dur,
                    )
                )
            return res

        return fi

    return f

def bench_module(func, warmup=25, rep=200):
    torch.cuda.synchronize()
    for i in range(warmup):
        func()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(rep):
        func()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / rep * 1000


@torch.no_grad()
def bench_more(func, warmup=25, rep=200, kernel=True, profile=True, msg="", export_trace=False):
    import torch

    module_bench_time = bench_module(func, warmup, rep)
    kernel_bench_time = do_bench(func, warmup, rep) if kernel else None
    if profile:
        print(f"----{msg}----")
        from torch.profiler import profile, record_function, ProfilerActivity

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(activities=activities, with_stack=True) as prof:
            for i in range(rep):
                func()
                prof.step()
        if export_trace or bench_config.EXPORT_TRACE:
            prof.export_chrome_trace(f"{msg}.json")
            print(f"Exported trace to {msg}.json")
        print("----" * 10, "CPU time", "----" * 10)
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
        print("----" * 10, "CUDA time", "----" * 10)
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    return module_bench_time, kernel_bench_time


def inspect_tensor(x, msg="", force=False):
    print(
        f"{msg}\n: shape={x.shape}, dtype={x.dtype}, device={x.device}, layout={x.layout}, strides={x.stride()}, is_contiguous={x.is_contiguous()}"
    )
    if DEBUG or force:
        print(x)


def see_memory_usage(message, force=True):
    # Modified from DeepSpeed
    import gc
    import warnings

    import torch.distributed as dist

    if not force:
        return
    # if dist.is_initialized() and not dist.get_rank() == 0:
    #     return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # Print message except when distributed but not rank 0
    print(message)
    print(
        f"AllocatedMem {round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024),2 )} GB \
        MaxAllocatedMem {round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),2)} GB \
        ReservedMem {round(torch.cuda.memory_reserved() / (1024 * 1024 * 1024),2)} GB \
        MaxReservedMem {round(torch.cuda.max_memory_reserved() / (1024 * 1024 * 1024))} GB "
    )

    # get the peak memory to report correct data, so reset the counter for the next call
    torch.cuda.reset_peak_memory_stats()

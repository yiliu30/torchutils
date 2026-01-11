import os
import sys
import time
from datetime import timedelta
import argparse
import torch
import torch.distributed as dist


def env_summary():
    keys = [
        "NCCL_DEBUG",
        "NCCL_IB_DISABLE",
        "NCCL_SHARP_DISABLE",
        "NCCL_BLOCKING_WAIT",
        "NCCL_ASYNC_ERROR_HANDLING",
        "NCCL_NET",
        "NCCL_SOCKET_IFNAME",
        "NCCL_PLUGIN_PATH",
    ]
    return {k: os.environ.get(k) for k in keys}


def parse_args():
    p = argparse.ArgumentParser(description="NCCL multi-op sanity with timeouts")
    p.add_argument("--tensor-size", type=int, default=1024, help="Tensor length for ops")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="Tensor dtype")
    p.add_argument("--timeout-secs", type=int, default=60, help="Process group/op timeout seconds")
    p.add_argument("--ops", type=str, nargs="*", default=["all_reduce", "broadcast", "all_gather", "reduce_scatter", "barrier"], help="Ops to run in order")
    return p.parse_args()


def make_tensor(device, size, dtype, fill):
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    t = torch.ones(size, device=device, dtype=dtypes[dtype])
    if isinstance(fill, (int, float)):
        t.mul_(fill)
    return t


def main():
    args = parse_args()

    # Initialize process group with timeout to avoid silent hangs
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=args.timeout_secs),
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))

    if not torch.cuda.is_available():
        if rank == 0:
            print("[FAIL] CUDA not available; cannot run NCCL sanity.")
        dist.destroy_process_group()
        sys.exit(2)

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    # Prepare tensors
    x = make_tensor(device, args.tensor_size, args.dtype, fill=rank)

    results = {"rank": rank, "world_size": world_size, "env": env_summary(), "ops": []}

    # Run selected ops
    try:
        for op in args.ops:
            t0 = time.time()
            if op == "all_reduce":
                dist.all_reduce(x, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize()
                elapsed = time.time() - t0
                # expected sum: sum(range(world_size))
                expected = sum(range(world_size))
                ok = float(x[0].item()) == float(expected)
                results["ops"].append({"op": op, "elapsed_sec": elapsed, "ok": ok, "value": float(x[0].item()), "expected": float(expected)})
            elif op == "broadcast":
                # broadcast first element from rank 0, then verify
                b = make_tensor(device, 1, args.dtype, fill=42 if rank == 0 else 0)
                dist.broadcast(b, src=0)
                torch.cuda.synchronize()
                elapsed = time.time() - t0
                ok = float(b.item()) == 42.0
                results["ops"].append({"op": op, "elapsed_sec": elapsed, "ok": ok, "value": float(b.item())})
            elif op == "all_gather":
                g = make_tensor(device, 1, args.dtype, fill=rank)
                gather_list = [make_tensor(device, 1, args.dtype, fill=0) for _ in range(world_size)]
                dist.all_gather(gather_list, g)
                torch.cuda.synchronize()
                elapsed = time.time() - t0
                vals = [float(t.item()) for t in gather_list]
                ok = vals == list(map(float, range(world_size)))
                results["ops"].append({"op": op, "elapsed_sec": elapsed, "ok": ok, "values": vals})
            elif op == "reduce_scatter":
                # build input of size world_size with rank values, expect each rank gets sum of its position
                inp = make_tensor(device, world_size, args.dtype, fill=rank)
                out = make_tensor(device, 1, args.dtype, fill=0)
                dist.reduce_scatter(out, list(inp.chunk(world_size)), op=dist.ReduceOp.SUM)
                torch.cuda.synchronize()
                elapsed = time.time() - t0
                expected = sum(range(world_size))
                ok = float(out.item()) == float(expected)
                results["ops"].append({"op": op, "elapsed_sec": elapsed, "ok": ok, "value": float(out.item()), "expected": float(expected)})
            elif op == "barrier":
                dist.barrier()
                torch.cuda.synchronize()
                elapsed = time.time() - t0
                results["ops"].append({"op": op, "elapsed_sec": elapsed, "ok": True})
            else:
                results["ops"].append({"op": op, "error": "unknown op"})
    except Exception as e:
        results["error"] = str(e)
        # Print per-rank failure for easier triage
        print(f"[RANK {rank}] ERROR: {e}")
        dist.destroy_process_group()
        sys.exit(1)

    # Print per-rank structured line for debugging
    print({"rank": rank, "summary": results})

    # Only rank 0 prints concise OK line
    if rank == 0:
        ok_ops = all(item.get("ok", True) for item in results["ops"] if "ok" in item)
        first = next((item for item in results["ops"] if item.get("op") == "all_reduce"), None)
        first_val = None if first is None else first.get("value")
        print(
            f"Sanity OK: world_size={world_size}, allreduce sum={first_val}, "
            f"ops_ok={ok_ops}, timeout={args.timeout_secs}s"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

import torch
from habana_frameworks.torch import core as htcore
import torch.distributed

htcore.hpu_initialize()
a = torch.tensor([1, 2, 3])
device = torch.device("hpu")
a.to(device)
print(a)
print(f"torch version: {torch.__version__}")


import torch
import habana_frameworks.torch.dynamo.compile_backend

# input_cpu = torch.randint(size=[3,4], low=-10, high=10, dtype=torch.int32)
# input_hpu = input_cpu.to("hpu")

# def fn(input, other):
#     return torch.div(input, other)

# output_cpu = fn(input_cpu, 0.6)
# output_hpu = fn(input_hpu, 0.6)

# print(output_cpu)
# print(output_hpu.cpu())

# init torch distributed group
import torch.distributed as dist
import habana_frameworks.torch.distributed.hccl


# torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
def run(args):
    import time

    dist.init_process_group(backend="hccl", world_size=args.wd)
    with torch.device("hpu"):
        for i in range(1000000):
            M, N, K = 4096, 4096, 4096
            M, N, K = M * 8, N * 8, K * 8
            a = torch.randn(M, N)
            b = torch.randn(N, K)
            out = a @ b
            if i % 10 == 0:
                print(f"iteration {i}============")
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
                print(f"Rank {rank} iteration {i}")
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            htcore.mark_step()
            time.sleep(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", "--local_rank", type=int)
    parser.add_argument("-w", "--wd", type=int, default=1, help="Number of processes for distributed training")
    args = parser.parse_args()
    run(args)

"""
 python -m torch.distributed.run  --nproc-per-node 8  keep_run.py 
"""

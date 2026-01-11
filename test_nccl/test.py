#  NCCL_HOME="/home/yiliu7/workspace/nccl_2.29.2-1+cuda13.1_x86_64"    torchrun --standalone --nproc-per-nod
import os
from datetime import timedelta

# Optional safe-local mode: disable IB/SHARP, force socket, and use stable NCCL paths
if os.environ.get("SAFE_LOCAL"):
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_SHARP_DISABLE", "1")
    os.environ.setdefault("NCCL_NET", "Socket")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_BLOCKING_WAIT", "0")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    # Stable algorithm/protocol for tp>2 on B200 NVLink systems
    # os.environ.setdefault("NCCL_ALGO", "Ring")
    # os.environ.setdefault("NCCL_PROTO", "Simple")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ.pop("NCCL_PLUGIN_PATH", None)

# Test PyTorch NCCL
import torch
import torch.distributed as dist
dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(seconds=int(os.environ.get("PG_TIMEOUT", "60"))))
local_rank = dist.get_rank() % torch.cuda.device_count()
torch.cuda.set_device(local_rank)
data = torch.FloatTensor([1,] * 128).to("cuda")
print(f"Running on rank {dist.get_rank()} / {dist.get_world_size()} using device {torch.cuda.current_device()}")
dist.all_reduce(data, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
value = data.mean().item()
world_size = dist.get_world_size()
assert value == world_size, f"Expected {world_size}, got {value}"

print("PyTorch NCCL is successful!")

# Test PyTorch GLOO
gloo_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
cpu_data = torch.FloatTensor([1,] * 128)
dist.all_reduce(cpu_data, op=dist.ReduceOp.SUM, group=gloo_group)
value = cpu_data.mean().item()
assert value == world_size, f"Expected {world_size}, got {value}"

print("PyTorch GLOO is successful!")

# Test vLLM NCCL, with cuda graph
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

pynccl = PyNcclCommunicator(group=gloo_group, device=local_rank)
pynccl.disabled = False
from vllm.utils.torch_utils import current_stream
s = current_stream()
# s = torch.cuda.Stream()
with torch.cuda.stream(s):
    data.fill_(1)
    pynccl.all_reduce(data, stream=s, op=dist.ReduceOp.SUM)
# Synchronize stream before reading result - all_reduce is async
s.synchronize()
print(f"After vLLM all_reduce, data : {data}")
# breakpoint()
value = data.mean().item()
assert value == world_size, f"Expected {world_size}, got {value}"

print("vLLM NCCL is successful!")

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(cuda_graph=g, stream=s):
    pynccl.all_reduce(data, stream=torch.cuda.current_stream(), op=dist.ReduceOp.SUM)

data.fill_(1)
g.replay()
torch.cuda.current_stream().synchronize()
value = data.mean().item()
assert value == world_size, f"Expected {world_size}, got {value}"

print("vLLM NCCL with cuda graph is successful!")

dist.destroy_process_group(gloo_group)
dist.destroy_process_group()

#  NCCL_NVLS_ENABLE=0  torchrun --nproc-per-node=4 test_nccl/test.py 

#         GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7     NIC0    NIC1    CPU Affinity    NUMA Affinity   GPU NUMA ID
# GPU0     X      NV18    NV18    NV18    NV18    NV18    NV18    NV18     SYS    SYS     0-47,96-143     0               N/A
# GPU1    NV18     X      NV18    NV18    NV18    NV18    NV18    NV18     SYS    SYS     0-47,96-143     0               N/A
# GPU2    NV18    NV18     X      NV18    NV18    NV18    NV18    NV18     SYS    SYS     0-47,96-143     0               N/A
# GPU3    NV18    NV18    NV18     X      NV18    NV18    NV18    NV18     SYS    SYS     0-47,96-143     0               N/A
# GPU4    NV18    NV18    NV18    NV18     X      NV18    NV18    NV18     PIX    PIX     48-95,144-191   1               N/A
# GPU5    NV18    NV18    NV18    NV18    NV18     X      NV18    NV18     PIX    PIX     48-95,144-191   1               N/A
# GPU6    NV18    NV18    NV18    NV18    NV18    NV18     X      NV18     NODE    NODE    48-95,144-191   1               N/A
# GPU7    NV18    NV18    NV18    NV18    NV18    NV18    NV18     X       NODE    NODE    48-95,144-191   1               N/A
# NIC0    SYS     SYS     SYS     SYS     PIX     PIX     NODE    NODE      X     PIX
# NIC1    SYS     SYS     SYS     SYS     PIX     PIX     NODE    NODE     PIX     X 

# Legend:

#   X    = Self
#   SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
#   NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
#   PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
#   PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
#   PIX  = Connection traversing at most a single PCIe bridge
#   NV#  = Connection traversing a bonded set of # NVLinks

# NIC Legend:

#   NIC0: mlx5_0
#   NIC1: mlx5_1

# yiliu7@ip-10-0-146-1:~/workspace$ 
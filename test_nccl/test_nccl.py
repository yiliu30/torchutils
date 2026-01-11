# Simple runner that delegates to tools/nccl_sanity.py
# Usage (example):
#   torchrun --standalone --nproc_per_node=2 tools/test_nccl.py --tensor-size 2048 --timeout-secs 60

from nccl_sanity import main

if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail

# Local NCCL sanity test runner
# - Prints NCCL libs, GPU/driver, and PyTorch NCCL availability
# - Runs a small torchrun all-reduce using nccl_sanity.py with safe env defaults

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")"/.. && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SHARP_DISABLE="${NCCL_SHARP_DISABLE:-1}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-0}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_NET="${NCCL_NET:-Socket}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
unset NCCL_PLUGIN_PATH || true

# 1) Environment probes
echo "=== NCCL libraries (ldconfig) ==="
ldconfig -p | grep -i nccl || echo "nccl not in ldconfig"

echo "\n=== GPUs / Driver ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
else
  echo "nvidia-smi not found"
fi

# 2) PyTorch/NCCL checks
echo "\n=== PyTorch NCCL availability ==="
$PYTHON_BIN - <<'PY'
import torch, torch.distributed as dist
print('cuda_available', torch.cuda.is_available())
print('gpu_count', torch.cuda.device_count())
print('nccl_available', getattr(dist, 'is_nccl_available', lambda: False)())
print('torch_version', torch.__version__)
PY

# 3) Run torchrun all-reduce sanity
if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun not found; install PyTorch distributed utils to proceed" >&2
  exit 2
fi

# Prefer tools/nccl_sanity.py, fallback to repo root
if [[ -f "$ROOT_DIR/tools/nccl_sanity.py" ]]; then
  SANITY_SCRIPT="$ROOT_DIR/tools/nccl_sanity.py"
elif [[ -f "$ROOT_DIR/nccl_sanity.py" ]]; then
  SANITY_SCRIPT="$ROOT_DIR/nccl_sanity.py"
else
  echo "Sanity script not found in tools/ or repo root." >&2
  exit 3
fi

echo "\n=== Running torchrun NCCL sanity ==="
echo "Env overrides: NCCL_DEBUG=$NCCL_DEBUG, NCCL_NET=$NCCL_NET, NCCL_IB_DISABLE=$NCCL_IB_DISABLE, NCCL_SHARP_DISABLE=$NCCL_SHARP_DISABLE, NCCL_BLOCKING_WAIT=$NCCL_BLOCKING_WAIT, NCCL_ASYNC_ERROR_HANDLING=$NCCL_ASYNC_ERROR_HANDLING, NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "$SANITY_SCRIPT"

echo "\nSanity finished. If rank 0 printed 'Sanity OK', NCCL runtime is good locally."

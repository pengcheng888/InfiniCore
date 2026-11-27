#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$REPO_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$REPO_DIR/python/infinicore/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/compat/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python3 -u "$SCRIPT_DIR/bench_nvidia_rmsnorm.py" "$@"

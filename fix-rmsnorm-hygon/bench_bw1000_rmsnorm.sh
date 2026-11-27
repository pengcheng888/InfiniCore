#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$REPO_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$HOME/.infini/lib:$REPO_DIR/python/infinicore/lib:/opt/dtk/lib:/usr/local/lib/python3.10/dist-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python3 -u "$SCRIPT_DIR/bench_bw1000_rmsnorm.py" "$@"

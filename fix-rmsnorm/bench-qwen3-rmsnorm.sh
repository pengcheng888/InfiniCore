#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0}"
CASE="${CASE:-all}"

export MUSA_VISIBLE_DEVICES

cd "$REPO_DIR"
set +u
source scripts/set_env_linux.sh
set -u

python - "$CASE" <<'PY'
import sys
import time

import torch
import infinicore
from infinicore.nn.functional import rms_norm as infini_rms_norm


def bench_case(name, shape):
    dtype = torch.bfloat16
    device = "musa"
    eps = 1e-6

    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device=device)
    w = torch.randn((shape[-1],), dtype=dtype, device=device)
    out = torch.empty_like(x)

    x_ic = infinicore.from_torch(x)
    w_ic = infinicore.from_torch(w)
    out_ic = infinicore.from_torch(out)

    infini_rms_norm(x_ic, [shape[-1]], w_ic, eps, out=out_ic)
    torch.musa.synchronize()
    ref = torch.nn.functional.rms_norm(x, [shape[-1]], w, eps)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    def time_fn(fn, iterations=100):
        for _ in range(20):
            fn()
        torch.musa.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.musa.synchronize()
        return (time.perf_counter() - start) / iterations * 1000

    t_ic = time_fn(lambda: infini_rms_norm(x_ic, [shape[-1]], w_ic, eps, out=out_ic))
    t_torch = time_fn(lambda: torch.nn.functional.rms_norm(x, [shape[-1]], w, eps))

    print(f"[{name}]")
    print(f"shape: {shape}")
    print(f"infinicore_rms_norm: {t_ic:.6f} ms")
    print(f"torch_rms_norm:      {t_torch:.6f} ms")
    print(f"ratio:               {t_ic / t_torch:.3f}x")
    print(f"throughput_ic:       {x.numel() / t_ic / 1e6:.3f} Gelem/s")
    print(f"throughput_torch:    {x.numel() / t_torch / 1e6:.3f} Gelem/s")
    print()


case = sys.argv[1]
if case in ("q", "all"):
    bench_case("q_norm", (4096, 32, 128))
if case in ("k", "all"):
    bench_case("k_norm", (4096, 8, 128))
PY

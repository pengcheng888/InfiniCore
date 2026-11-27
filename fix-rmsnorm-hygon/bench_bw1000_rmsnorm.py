#!/usr/bin/env python3
"""RMSNorm benchmark for the Hygon BW1000 dynamic-block fix."""

from __future__ import annotations

import argparse
import csv
import time

import torch
import infinicore
from infinicore.nn.functional import rms_norm as infini_rms_norm

CASES = [
    ("q_norm", (4096, 32, 128)),
    ("k_norm", (4096, 8, 128)),
    ("dim256", (2048, 32, 256)),
    ("dim512", (1024, 32, 512)),
    ("dim1024", (512, 32, 1024)),
    ("dim2048", (256, 32, 2048)),
    ("dim4096", (128, 32, 4096)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def bench_case(name: str, shape: tuple[int, ...], warmup: int, iterations: int) -> dict:
    dtype = torch.bfloat16
    device = "cuda"
    epsilon = 1e-6

    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn((shape[-1],), dtype=dtype, device=device)
    out = torch.empty_like(x)

    x_ic = infinicore.from_torch(x)
    w_ic = infinicore.from_torch(weight)
    out_ic = infinicore.from_torch(out)

    infini_rms_norm(x_ic, [shape[-1]], w_ic, epsilon, out=out_ic)
    torch.cuda.synchronize()
    reference = torch.nn.functional.rms_norm(x, [shape[-1]], weight, epsilon)
    torch.testing.assert_close(out, reference, rtol=1e-2, atol=1e-2)

    def time_function(function) -> float:
        for _ in range(warmup):
            function()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            function()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / iterations * 1000

    infini_ms = time_function(
        lambda: infini_rms_norm(x_ic, [shape[-1]], w_ic, epsilon, out=out_ic)
    )
    torch_ms = time_function(
        lambda: torch.nn.functional.rms_norm(x, [shape[-1]], weight, epsilon)
    )

    row = {
        "case": name,
        "shape": str(shape),
        "infinicore_ms": round(infini_ms, 6),
        "torch_ms": round(torch_ms, 6),
        "infinicore_gelem_s": round(x.numel() / infini_ms / 1e6, 3),
        "torch_gelem_s": round(x.numel() / torch_ms / 1e6, 3),
        "infinicore_over_torch": round(infini_ms / torch_ms, 3),
    }
    print(
        f"{name:8s} shape={str(shape):22s} "
        f"infinicore={infini_ms:.6f} ms torch={torch_ms:.6f} ms "
        f"throughput={row['infinicore_gelem_s']:.3f} Gelem/s"
    )
    return row


def main() -> None:
    args = parse_args()
    props = torch.cuda.get_device_properties(0)
    print(f"torch={torch.__version__} hip={torch.version.hip}")
    print(f"infinicore={infinicore.__file__}")
    print(f"gpu={props.name} gcn={props.gcnArchName} CU={props.multi_processor_count}")
    print()

    rows = [
        bench_case(name, shape, args.warmup, args.iterations)
        for name, shape in CASES
    ]

    if args.output:
        fieldnames = list(rows[0])
        with open(args.output, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"csv={args.output}")


if __name__ == "__main__":
    main()

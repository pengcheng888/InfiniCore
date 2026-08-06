import argparse
import traceback

import torch


def _parse_shape(shape):
    parts = shape.lower().replace("x", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be M,N,K")
    try:
        m, n, k = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape values must be integers") from exc
    return m, n, k


def run(shapes):
    from aiter.tuned_gemm import tgemm

    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"hip={getattr(torch.version, 'hip', None)}")
    print(f"tgemm_module={getattr(tgemm, '__file__', type(tgemm).__name__)}")

    if not torch.cuda.is_available():
        raise RuntimeError("torch cuda device is not available")

    for m, n, k in shapes:
        torch.manual_seed(1000 + m + n + k)
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        y = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

        out = tgemm.mm(x, y, otype=x.dtype).float()
        ref = torch.mm(x, y.t(), out_dtype=torch.float32)
        torch.cuda.synchronize()

        max_abs = (out.float() - ref.float()).abs().max().item()
        ok = torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
        print(
            f"shape m={m} n={n} k={k}: "
            f"out_dtype={out.dtype} max_abs={max_abs:.6g} allclose={ok}"
        )
        if not ok:
            raise AssertionError(f"tgemm.mm mismatch for m={m}, n={n}, k={k}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        help="Test shape as M,N,K. Can be passed multiple times.",
    )
    args = parser.parse_args()

    shapes = args.shape or [
        (4, 8, 16),
        (32, 256, 4096),
    ]

    try:
        run(shapes)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

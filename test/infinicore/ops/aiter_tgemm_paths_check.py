import argparse
import traceback

import torch


def _parse_shape(text):
    parts = text.lower().replace("x", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be M,N,K")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape values must be integers") from exc


def _dtype(name):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unknown dtype: {name}")


def _make_inputs(shape, dtype, seed):
    m, n, k = shape
    torch.manual_seed(seed + m * 17 + n * 31 + k)
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    weight = torch.randn((n, k), device="cuda", dtype=dtype)
    return x, weight


def _check_output(name, out, x, weight, atol, rtol):
    if out is None:
        raise RuntimeError(f"{name} returned None")
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"{name} returned {type(out).__name__}, expected torch.Tensor")

    torch.cuda.synchronize()
    ref = torch.mm(x, weight.t(), out_dtype=torch.float32)
    compare_ref = ref.to(out.dtype) if out.dtype != torch.float32 else ref
    max_abs = (out.float() - compare_ref.float()).abs().max().item()
    ok = torch.allclose(out.float(), compare_ref.float(), atol=atol, rtol=rtol)
    print(
        f"{name}: PASS_CALL out_shape={tuple(out.shape)} out_dtype={out.dtype} "
        f"max_abs={max_abs:.6g} allclose={ok}"
    )
    if not ok:
        raise AssertionError(f"{name} output mismatch")


def _run_one(name, call, shape, dtype, seed, atol, rtol):
    print(f"\n===== {name} =====")
    print(f"shape={shape} dtype={dtype}")
    try:
        x, weight = _make_inputs(shape, dtype, seed)
        out = call(x, weight)
        _check_output(name, out, x, weight, atol, rtol)
        return True
    except Exception:
        traceback.print_exc()
        return False


def _register_skinny_custom_ops():
    import aiter

    if hasattr(aiter, "wvSpltK") and hasattr(aiter, "LLMM1"):
        print("skinny custom ops already exist in aiter namespace")
        return

    from aiter.ops.custom import LLMM1, wvSpltK

    aiter.wvSpltK = wvSpltK
    aiter.LLMM1 = LLMM1
    print("registered aiter.ops.custom wvSpltK/LLMM1 into aiter namespace")


def main():
    parser = argparse.ArgumentParser(
        description="Directly check aiter.tuned_gemm apply_hipb_mm/apply_rocb_mm/apply_skinny paths."
    )
    parser.add_argument("--hipb-shape", type=_parse_shape, default=(32, 256, 4096))
    parser.add_argument("--rocb-shape", type=_parse_shape, default=(32, 256, 4096))
    parser.add_argument("--skinny-shape", type=_parse_shape, default=(4, 256, 4096))
    parser.add_argument("--hipb-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--rocb-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--skinny-dtype", choices=["fp16"], default="fp16")
    parser.add_argument("--skinny-solidx", type=int, default=0)
    parser.add_argument(
        "--no-register-skinny-custom",
        dest="register_skinny_custom",
        action="store_false",
        help="Do not inject aiter.ops.custom wvSpltK/LLMM1 into the aiter namespace before apply_skinny.",
    )
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("torch cuda device is not available")

    from aiter.tuned_gemm import tgemm

    print(f"torch={torch.__version__}")
    print(f"hip={getattr(torch.version, 'hip', None)}")
    print(f"cuda_device={torch.cuda.get_device_name()}")
    print(f"tgemm={type(tgemm).__name__}")

    results = {}

    results["apply_hipb_mm"] = _run_one(
        "apply_hipb_mm",
        lambda x, weight: tgemm.apply_hipb_mm(
            x, weight, -1, bias=None, otype=x.dtype
        ),
        args.hipb_shape,
        _dtype(args.hipb_dtype),
        args.seed,
        args.atol,
        args.rtol,
    )

    results["apply_rocb_mm"] = _run_one(
        "apply_rocb_mm",
        lambda x, weight: tgemm.apply_rocb_mm(x, weight, -1, bias=None),
        args.rocb_shape,
        _dtype(args.rocb_dtype),
        args.seed,
        args.atol,
        args.rtol,
    )

    results["apply_skinny"] = _run_one(
        "apply_skinny",
        lambda x, weight: (
            _register_skinny_custom_ops() if args.register_skinny_custom else None,
            tgemm.apply_skinny(x, weight, args.skinny_solidx, bias=None),
        )[1],
        args.skinny_shape,
        _dtype(args.skinny_dtype),
        args.seed,
        args.atol,
        args.rtol,
    )

    print("\n===== SUMMARY =====")
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")

    if args.strict and not all(results.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import time

import infinicore
import torch
from infinicore.lib import _infinicore

try:
    from aiter.tuned_gemm import tgemm
except ImportError:
    tgemm = None


DEFAULT_TOKENS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"

def _parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _wrap(tensor, keepalive):
    wrapped = infinicore.from_torch(tensor)
    keepalive.append(wrapped)
    return wrapped._underlying


def _sync():
    infinicore.sync_stream()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    _sync()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - start) * 1000.0 / float(iters)


def _make_case(tokens, hidden, out_features, seed):
    torch.manual_seed(seed + tokens * 17 + hidden + out_features)
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((out_features, hidden), device="cuda", dtype=torch.bfloat16)
    naive_out = torch.empty((tokens, out_features), device="cuda", dtype=torch.float32)
    kernel_out = torch.empty_like(naive_out)
    blas_out = torch.empty_like(naive_out)
    return x, weight, naive_out, kernel_out, blas_out


def _run_case(tokens, args):
    x, weight, naive_out, kernel_out, blas_out = _make_case(tokens, args.hidden, args.out_features, args.seed)
    keepalive = []
    x_core = _wrap(x, keepalive)
    weight_core = _wrap(weight, keepalive)
    naive_core = _wrap(naive_out, keepalive)
    kernel_core = _wrap(kernel_out, keepalive)
    blas_core = _wrap(blas_out, keepalive)

    def naive():
        _infinicore.deepseek_v4_linear_bf16_fp32_naive_(naive_core, x_core, weight_core)

    def kernel():
        _infinicore.deepseek_v4_linear_bf16_fp32_kernel_(kernel_core, x_core, weight_core)

    def blas():
        _infinicore.deepseek_v4_linear_bf16_fp32_blas_(blas_core, x_core, weight_core)

    def tgemm_mm():
        return tgemm.mm(x, weight, otype=x.dtype).float()

    kernel_max_abs = float("nan")
    kernel_allclose = "skip"
    blas_max_abs = float("nan")
    blas_allclose = "skip"
    tgemm_max_abs = float("nan")
    tgemm_allclose = "skip"
    if args.check:
        naive()
        kernel()
        blas()
        tgemm_out = tgemm_mm() if tgemm is not None else None
        _sync()
        kernel_max_abs = (naive_out - kernel_out).abs().max().item()
        kernel_allclose = str(torch.allclose(naive_out, kernel_out, atol=args.atol, rtol=args.rtol))
        blas_max_abs = (naive_out - blas_out).abs().max().item()
        blas_allclose = str(torch.allclose(naive_out, blas_out, atol=args.atol, rtol=args.rtol))
        if tgemm_out is not None:
            tgemm_max_abs = (naive_out - tgemm_out).abs().max().item()
            tgemm_allclose = str(torch.allclose(naive_out, tgemm_out, atol=args.atol, rtol=args.rtol))

    naive_ms = _bench(naive, args.warmup, args.iters)
    kernel_ms = _bench(kernel, args.warmup, args.iters)
    blas_ms = _bench(blas, args.warmup, args.iters)
    tgemm_ms = _bench(tgemm_mm, args.warmup, args.iters) if tgemm is not None else float("nan")
    kernel_speedup = naive_ms / kernel_ms if kernel_ms > 0 else float("inf")
    blas_speedup = naive_ms / blas_ms if blas_ms > 0 else float("inf")
    tgemm_speedup = naive_ms / tgemm_ms if tgemm_ms > 0 else float("inf")
    return {
        "tokens": tokens,
        "naive_ms": naive_ms,
        "kernel_ms": kernel_ms,
        "blas_ms": blas_ms,
        "tgemm_ms": tgemm_ms,
        "kernel_speedup": kernel_speedup,
        "blas_speedup": blas_speedup,
        "tgemm_speedup": tgemm_speedup,
        "kernel_max_abs": kernel_max_abs,
        "kernel_allclose": kernel_allclose,
        "blas_max_abs": blas_max_abs,
        "blas_allclose": blas_allclose,
        "tgemm_max_abs": tgemm_max_abs,
        "tgemm_allclose": tgemm_allclose,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek-V4 bf16->fp32 linear naive/kernel operators.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=256)
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--check", action="store_true", default=True)
    parser.add_argument("--no-check", dest="check", action="store_false")
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()

    tokens_list = _parse_int_list(args.tokens)
    print("DeepSeek-V4 linear bf16->fp32 性能对比")
    print(f"hidden={args.hidden} out_features={args.out_features} iters={args.iters} warmup={args.warmup} check={args.check}")
    print(f"tgemm={'available' if tgemm is not None else 'unavailable'}")
    print(
        f"{'tokens':>8} | {'naive(ms)':>11} | {'kernel(ms)':>11} | "
        f"{'kernel spd':>10} | {'blas(ms)':>10} | {'blas spd':>8} | "
        f"{'tgemm(ms)':>11} | {'tgemm spd':>9} | {'ker_abs':>10} | "
        f"{'ker_ok':>6} | {'blas_abs':>10} | {'blas_ok':>7} | {'tgm_abs':>10} | {'tgm_ok':>6}"
    )
    print("-" * 151)
    for tokens in tokens_list:
        result = _run_case(tokens, args)
        kernel_max_abs = result["kernel_max_abs"]
        blas_max_abs = result["blas_max_abs"]
        tgemm_max_abs = result["tgemm_max_abs"]
        kernel_max_abs_text = "nan" if kernel_max_abs != kernel_max_abs else f"{kernel_max_abs:.4e}"
        blas_max_abs_text = "nan" if blas_max_abs != blas_max_abs else f"{blas_max_abs:.4e}"
        tgemm_max_abs_text = "nan" if tgemm_max_abs != tgemm_max_abs else f"{tgemm_max_abs:.4e}"
        blas_ms = result["blas_ms"]
        blas_ms_text = "nan" if blas_ms != blas_ms else f"{blas_ms:.4f}"
        blas_speedup = result["blas_speedup"]
        blas_speedup_text = "nan" if blas_speedup != blas_speedup else f"{blas_speedup:.2f}"
        tgemm_ms = result["tgemm_ms"]
        tgemm_ms_text = "nan" if tgemm_ms != tgemm_ms else f"{tgemm_ms:.4f}"
        tgemm_speedup = result["tgemm_speedup"]
        tgemm_speedup_text = "nan" if tgemm_speedup != tgemm_speedup else f"{tgemm_speedup:.2f}"
        print(
            f"{result['tokens']:8d} | "
            f"{result['naive_ms']:11.4f} | "
            f"{result['kernel_ms']:11.4f} | "
            f"{result['kernel_speedup']:10.2f} | "
            f"{blas_ms_text:>10} | "
            f"{blas_speedup_text:>8} | "
            f"{tgemm_ms_text:>11} | "
            f"{tgemm_speedup_text:>9} | "
            f"{kernel_max_abs_text:>10} | "
            f"{result['kernel_allclose']:>6} | "
            f"{blas_max_abs_text:>10} | "
            f"{result['blas_allclose']:>7} | "
            f"{tgemm_max_abs_text:>10} | "
            f"{result['tgemm_allclose']:>6}"
        )


if __name__ == "__main__":
    main()

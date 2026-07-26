import argparse
import time

import infinicore
import torch
from infinicore.lib import _infinicore


DEFAULT_TOKENS = "1,4,16,64,256,1024,4096"


def _parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _wrap(tensor, keepalive):
    wrapped = infinicore.from_torch(tensor)
    keepalive.append(wrapped)
    return wrapped._underlying


def _sync():
    infinicore.sync_stream()


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
    return x, weight, naive_out, kernel_out


def _run_case(tokens, args):
    x, weight, naive_out, kernel_out = _make_case(tokens, args.hidden, args.out_features, args.seed)
    keepalive = []
    x_core = _wrap(x, keepalive)
    weight_core = _wrap(weight, keepalive)
    naive_core = _wrap(naive_out, keepalive)
    kernel_core = _wrap(kernel_out, keepalive)

    def naive():
        _infinicore.deepseek_v4_linear_bf16_fp32_naive_(naive_core, x_core, weight_core)

    def kernel():
        _infinicore.deepseek_v4_linear_bf16_fp32_kernel_(kernel_core, x_core, weight_core)

    max_abs = float("nan")
    allclose = "skip"
    if args.check:
        naive()
        kernel()
        _sync()
        max_abs = (naive_out - kernel_out).abs().max().item()
        allclose = str(torch.allclose(naive_out, kernel_out, atol=args.atol, rtol=args.rtol))

    naive_ms = _bench(naive, args.warmup, args.iters)
    kernel_ms = _bench(kernel, args.warmup, args.iters)
    speedup = naive_ms / kernel_ms if kernel_ms > 0 else float("inf")
    return {
        "tokens": tokens,
        "naive_ms": naive_ms,
        "kernel_ms": kernel_ms,
        "speedup": speedup,
        "max_abs": max_abs,
        "allclose": allclose,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek-V4 bf16->fp32 linear naive/kernel operators.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=256)
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--check", action="store_true", default=True)
    parser.add_argument("--no-check", dest="check", action="store_false")
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()

    tokens_list = _parse_int_list(args.tokens)
    print("DeepSeek-V4 linear bf16->fp32 性能对比")
    print(f"hidden={args.hidden} out_features={args.out_features} iters={args.iters} warmup={args.warmup} check={args.check}")
    print(f"{'tokens':>8} | {'naive(ms)':>11} | {'kernel(ms)':>11} | {'speedup':>8} | {'max_abs':>10} | {'allclose':>8}")
    print("-" * 73)
    for tokens in tokens_list:
        result = _run_case(tokens, args)
        max_abs = result["max_abs"]
        max_abs_text = "nan" if max_abs != max_abs else f"{max_abs:.4e}"
        print(
            f"{result['tokens']:8d} | "
            f"{result['naive_ms']:11.4f} | "
            f"{result['kernel_ms']:11.4f} | "
            f"{result['speedup']:8.2f} | "
            f"{max_abs_text:>10} | "
            f"{result['allclose']:>8}"
        )


if __name__ == "__main__":
    main()

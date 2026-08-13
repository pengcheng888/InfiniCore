import argparse
import csv
import statistics
import time

import infinicore
import torch
from infinicore.lib import _infinicore


DEFAULT_TOKENS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"


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

    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _sync()
        end = time.perf_counter()
        samples.append((end - start) * 1000.0)

    total_ms = sum(samples)
    return {
        "total_ms": total_ms,
        "avg_ms": total_ms / float(iters),
        "median_ms": statistics.median(samples),
    }


def _max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def _check_pair(name, naive_fn, kernel_fn, naive_outs, kernel_outs, atol=2e-2, rtol=2e-2):
    naive_fn()
    kernel_fn()
    _sync()
    for idx, (lhs, rhs) in enumerate(zip(naive_outs, kernel_outs)):
        if not torch.allclose(lhs.float(), rhs.float(), atol=atol, rtol=rtol):
            raise AssertionError(f"{name}[{idx}] mismatch: max_abs={_max_abs(lhs, rhs)}")


def _make_case(tokens, hc, hidden, seed):
    torch.manual_seed(seed + tokens * 17 + hidden)
    device = "cuda"
    dtype = torch.bfloat16
    k = hc * hidden

    x = torch.randn(tokens, hc, hidden, device=device, dtype=dtype)
    fn = torch.randn(hc, k, device=device, dtype=torch.float32) * 0.02
    scale = torch.randn(1, device=device, dtype=torch.float32) * 0.1
    base = torch.randn(hc, device=device, dtype=torch.float32) * 0.1

    y_naive = torch.empty(tokens, hidden, device=device, dtype=dtype)
    y_kernel = torch.empty_like(y_naive)

    keepalive = []
    u = {
        "x": _wrap(x, keepalive),
        "fn": _wrap(fn, keepalive),
        "scale": _wrap(scale, keepalive),
        "base": _wrap(base, keepalive),
        "y_naive": _wrap(y_naive, keepalive),
        "y_kernel": _wrap(y_kernel, keepalive),
    }
    tensors = {
        "y_naive": y_naive,
        "y_kernel": y_kernel,
    }
    return u, tensors, keepalive


def _run_case(tokens, hc, hidden, args):
    u, tensors, keepalive = _make_case(tokens, hc, hidden, args.seed)

    def naive():
        _infinicore.deepseek_v4_hc_head_naive_(
            u["y_naive"],
            u["x"],
            u["fn"],
            u["scale"],
            u["base"],
            args.rms_eps,
            args.hc_eps,
        )

    def kernel():
        _infinicore.deepseek_v4_hc_head_kernel_(
            u["y_kernel"],
            u["x"],
            u["fn"],
            u["scale"],
            u["base"],
            args.rms_eps,
            args.hc_eps,
        )

    if args.check:
        _check_pair("hc_head_kernel", naive, kernel, (tensors["y_naive"],), (tensors["y_kernel"],))

    rows = []
    for backend, fn in (
        ("naive", naive),
        ("kernel", kernel),
    ):
        result = _bench(fn, args.warmup, args.iters)
        rows.append(
            {
                "tokens": tokens,
                "hc": hc,
                "hidden": hidden,
                "backend": backend,
                "iters": args.iters,
                "total_ms": result["total_ms"],
                "avg_ms": result["avg_ms"],
                "median_ms": result["median_ms"],
            }
        )
    _sync()
    del keepalive
    torch.cuda.empty_cache()
    return rows


def _print_rows(rows):
    indexed = {(row["tokens"], row["backend"]): row for row in rows}
    tokens_list = sorted({row["tokens"] for row in rows})
    backends = []
    for row in rows:
        if row["backend"] not in backends:
            backends.append(row["backend"])

    header = f"{'tokens':>8}  {'hc':>2}  {'hidden':>6}  {'iters':>5}"
    for backend in backends:
        header += (
            f"  {backend + ' total(ms)':>20}"
            f"  {backend + ' avg(ms)':>18}"
            f"  {backend + ' median(ms)':>21}"
        )
        if backend != "naive":
            header += f"  {backend + ' speedup':>18}"
    print(header)
    print("-" * len(header))

    for tokens in tokens_list:
        naive = indexed.get((tokens, "naive"))
        if naive is None:
            continue
        line = f"{tokens:8d}  {naive['hc']:2d}  {naive['hidden']:6d}  {naive['iters']:5d}"
        for backend in backends:
            row = indexed.get((tokens, backend))
            line += f"  {row['total_ms']:20.3f}  {row['avg_ms']:18.6f}  {row['median_ms']:21.6f}"
            if backend != "naive":
                speedup = naive["avg_ms"] / row["avg_ms"] if row["avg_ms"] > 0 else float("inf")
                line += f"  {speedup:17.2f}x"
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--hc", type=int, default=4)
    parser.add_argument("--tokens", type=str, default=DEFAULT_TOKENS)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--rms-eps", type=float, default=1e-6)
    parser.add_argument("--hc-eps", type=float, default=1e-6)
    parser.add_argument("--check", dest="check", action="store_true", default=True)
    parser.add_argument("--no-check", dest="check", action="store_false")
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/Hygon device is required for HC head perf test")

    all_rows = []
    for tokens in _parse_int_list(args.tokens):
        all_rows.extend(_run_case(tokens, args.hc, args.hidden, args))

    _print_rows(all_rows)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["tokens", "hc", "hidden", "backend", "iters", "total_ms", "avg_ms", "median_ms"],
            )
            writer.writeheader()
            writer.writerows(all_rows)


if __name__ == "__main__":
    main()

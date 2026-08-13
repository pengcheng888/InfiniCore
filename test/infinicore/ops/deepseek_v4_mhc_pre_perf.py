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
    mix_hc = (2 + hc) * hc
    k = hc * hidden

    residual = torch.randn(tokens, hc, hidden, device=device, dtype=dtype)
    fn = torch.randn(mix_hc, k, device=device, dtype=torch.float32) * 0.02
    hc_scale = torch.randn(3, device=device, dtype=torch.float32) * 0.1
    hc_base = torch.randn(mix_hc, device=device, dtype=torch.float32) * 0.1

    y_naive = torch.empty(tokens, hidden, device=device, dtype=dtype)
    post_naive = torch.empty(tokens, hc, device=device, dtype=torch.float32)
    comb_naive = torch.empty(tokens, hc, hc, device=device, dtype=torch.float32)
    y_kernel = torch.empty_like(y_naive)
    post_kernel = torch.empty_like(post_naive)
    comb_kernel = torch.empty_like(comb_naive)
    keepalive = []
    u = {
        "residual": _wrap(residual, keepalive),
        "fn": _wrap(fn, keepalive),
        "hc_scale": _wrap(hc_scale, keepalive),
        "hc_base": _wrap(hc_base, keepalive),
        "y_naive": _wrap(y_naive, keepalive),
        "post_naive": _wrap(post_naive, keepalive),
        "comb_naive": _wrap(comb_naive, keepalive),
        "y_kernel": _wrap(y_kernel, keepalive),
        "post_kernel": _wrap(post_kernel, keepalive),
        "comb_kernel": _wrap(comb_kernel, keepalive),
    }
    tensors = {
        "y_naive": y_naive,
        "post_naive": post_naive,
        "comb_naive": comb_naive,
        "y_kernel": y_kernel,
        "post_kernel": post_kernel,
        "comb_kernel": comb_kernel,
    }
    return u, tensors, keepalive


def _run_case(tokens, hc, hidden, args):
    u, tensors, keepalive = _make_case(tokens, hc, hidden, args.seed)

    def naive():
        _infinicore.deepseek_v4_mhc_pre_naive_(
            u["y_naive"],
            u["post_naive"],
            u["comb_naive"],
            u["residual"],
            u["fn"],
            u["hc_scale"],
            u["hc_base"],
            args.rms_eps,
            args.hc_pre_eps,
            args.hc_sinkhorn_eps,
            args.sinkhorn_repeat,
        )

    def kernel():
        _infinicore.deepseek_v4_mhc_pre_kernel_(
            u["y_kernel"],
            u["post_kernel"],
            u["comb_kernel"],
            u["residual"],
            u["fn"],
            u["hc_scale"],
            u["hc_base"],
            args.rms_eps,
            args.hc_pre_eps,
            args.hc_sinkhorn_eps,
            args.sinkhorn_repeat,
        )

    if args.check:
        ref = (tensors["y_naive"], tensors["post_naive"], tensors["comb_naive"])
        _check_pair("mhc_pre_kernel", naive, kernel, ref, (tensors["y_kernel"], tensors["post_kernel"], tensors["comb_kernel"]))

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
    parser.add_argument("--hc-pre-eps", type=float, default=1e-6)
    parser.add_argument("--hc-sinkhorn-eps", type=float, default=1e-6)
    parser.add_argument("--sinkhorn-repeat", "--sinkhorn-iters", dest="sinkhorn_repeat", type=int, default=5)
    parser.add_argument("--check", dest="check", action="store_true", default=True)
    parser.add_argument("--no-check", dest="check", action="store_false")
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/Hygon device is required for MHC pre perf test")

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

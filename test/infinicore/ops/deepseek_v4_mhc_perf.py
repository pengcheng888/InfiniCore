import argparse
import csv
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


def _bench(name, fn, warmup, iters):
    for _ in range(warmup):
        fn()
    _sync()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    end = time.perf_counter()

    total_ms = (end - start) * 1000.0
    return {
        "op": name,
        "total_ms": total_ms,
        "avg_ms": total_ms / float(iters),
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

    x = torch.randn(tokens, hc, hidden, device=device, dtype=dtype)
    fn_pre = torch.randn(mix_hc, k, device=device, dtype=torch.float32) * 0.02
    scale_pre = torch.randn(3, device=device, dtype=torch.float32) * 0.1
    base_pre = torch.randn(mix_hc, device=device, dtype=torch.float32) * 0.1

    y_pre_naive = torch.empty(tokens, hidden, device=device, dtype=dtype)
    post_pre_naive = torch.empty(tokens, hc, device=device, dtype=torch.float32)
    comb_pre_naive = torch.empty(tokens, hc, hc, device=device, dtype=torch.float32)
    y_pre_kernel = torch.empty_like(y_pre_naive)
    post_pre_kernel = torch.empty_like(post_pre_naive)
    comb_pre_kernel = torch.empty_like(comb_pre_naive)

    x_post = torch.randn(tokens, hidden, device=device, dtype=dtype)
    post = torch.rand(tokens, hc, device=device, dtype=torch.float32) * 2.0
    comb = torch.rand(tokens, hc, hc, device=device, dtype=torch.float32)
    comb = (comb / (comb.sum(dim=1, keepdim=True) + 1e-6)).contiguous()
    y_post_naive = torch.empty(tokens, hc, hidden, device=device, dtype=dtype)
    y_post_kernel = torch.empty_like(y_post_naive)

    fn_head = torch.randn(hc, k, device=device, dtype=torch.float32) * 0.02
    scale_head = torch.randn(1, device=device, dtype=torch.float32) * 0.1
    base_head = torch.randn(hc, device=device, dtype=torch.float32) * 0.1
    y_head_naive = torch.empty(tokens, hidden, device=device, dtype=dtype)
    y_head_kernel = torch.empty_like(y_head_naive)

    keepalive = []
    u = {
        "x": _wrap(x, keepalive),
        "fn_pre": _wrap(fn_pre, keepalive),
        "scale_pre": _wrap(scale_pre, keepalive),
        "base_pre": _wrap(base_pre, keepalive),
        "y_pre_naive": _wrap(y_pre_naive, keepalive),
        "post_pre_naive": _wrap(post_pre_naive, keepalive),
        "comb_pre_naive": _wrap(comb_pre_naive, keepalive),
        "y_pre_kernel": _wrap(y_pre_kernel, keepalive),
        "post_pre_kernel": _wrap(post_pre_kernel, keepalive),
        "comb_pre_kernel": _wrap(comb_pre_kernel, keepalive),
        "x_post": _wrap(x_post, keepalive),
        "post": _wrap(post, keepalive),
        "comb": _wrap(comb, keepalive),
        "y_post_naive": _wrap(y_post_naive, keepalive),
        "y_post_kernel": _wrap(y_post_kernel, keepalive),
        "fn_head": _wrap(fn_head, keepalive),
        "scale_head": _wrap(scale_head, keepalive),
        "base_head": _wrap(base_head, keepalive),
        "y_head_naive": _wrap(y_head_naive, keepalive),
        "y_head_kernel": _wrap(y_head_kernel, keepalive),
    }

    tensors = {
        "y_pre_naive": y_pre_naive,
        "post_pre_naive": post_pre_naive,
        "comb_pre_naive": comb_pre_naive,
        "y_pre_kernel": y_pre_kernel,
        "post_pre_kernel": post_pre_kernel,
        "comb_pre_kernel": comb_pre_kernel,
        "y_post_naive": y_post_naive,
        "y_post_kernel": y_post_kernel,
        "y_head_naive": y_head_naive,
        "y_head_kernel": y_head_kernel,
    }
    return u, tensors, keepalive


def _run_case(tokens, hc, hidden, args):
    rms_eps = args.rms_eps
    hc_eps = args.hc_eps
    sinkhorn_iters = args.sinkhorn_iters
    u, tensors, keepalive = _make_case(tokens, hc, hidden, args.seed)

    def pre_naive():
        _infinicore.deepseek_v4_mhc_pre_naive_(
            u["y_pre_naive"],
            u["post_pre_naive"],
            u["comb_pre_naive"],
            u["x"],
            u["fn_pre"],
            u["scale_pre"],
            u["base_pre"],
            rms_eps,
            hc_eps,
            sinkhorn_iters,
        )

    def pre_kernel():
        _infinicore.deepseek_v4_mhc_pre_kernel_(
            u["y_pre_kernel"],
            u["post_pre_kernel"],
            u["comb_pre_kernel"],
            u["x"],
            u["fn_pre"],
            u["scale_pre"],
            u["base_pre"],
            rms_eps,
            hc_eps,
            sinkhorn_iters,
        )

    def post_naive():
        _infinicore.deepseek_v4_mhc_post_naive_(
            u["y_post_naive"],
            u["x_post"],
            u["x"],
            u["post"],
            u["comb"],
        )

    def post_kernel():
        _infinicore.deepseek_v4_mhc_post_kernel_(
            u["y_post_kernel"],
            u["x_post"],
            u["x"],
            u["post"],
            u["comb"],
        )

    def head_naive():
        _infinicore.deepseek_v4_mhc_head_naive_(
            u["y_head_naive"],
            u["x"],
            u["fn_head"],
            u["scale_head"],
            u["base_head"],
            rms_eps,
            hc_eps,
        )

    def head_kernel():
        _infinicore.deepseek_v4_mhc_head_kernel_(
            u["y_head_kernel"],
            u["x"],
            u["fn_head"],
            u["scale_head"],
            u["base_head"],
            rms_eps,
            hc_eps,
        )

    if args.check:
        _check_pair(
            "mhc_pre",
            pre_naive,
            pre_kernel,
            (tensors["y_pre_naive"], tensors["post_pre_naive"], tensors["comb_pre_naive"]),
            (tensors["y_pre_kernel"], tensors["post_pre_kernel"], tensors["comb_pre_kernel"]),
        )
        _check_pair(
            "hc_post",
            post_naive,
            post_kernel,
            (tensors["y_post_naive"],),
            (tensors["y_post_kernel"],),
        )
        _check_pair(
            "hc_head",
            head_naive,
            head_kernel,
            (tensors["y_head_naive"],),
            (tensors["y_head_kernel"],),
        )

    rows = []
    for op_name, backend, fn in (
        ("mhc_pre", "naive", pre_naive),
        ("mhc_pre", "kernel", pre_kernel),
        ("hc_post", "naive", post_naive),
        ("hc_post", "kernel", post_kernel),
        ("hc_head", "naive", head_naive),
        ("hc_head", "kernel", head_kernel),
    ):
        result = _bench(f"{op_name}_{backend}", fn, args.warmup, args.iters)
        rows.append(
            {
                "tokens": tokens,
                "hc": hc,
                "hidden": hidden,
                "op": op_name,
                "backend": backend,
                "iters": args.iters,
                "total_ms": result["total_ms"],
                "avg_ms": result["avg_ms"],
            }
        )
    _sync()
    del keepalive
    torch.cuda.empty_cache()
    return rows


def _print_by_op(rows):
    indexed = {}
    ops = []
    for row in rows:
        indexed[(row["tokens"], row["op"], row["backend"])] = row
        if row["op"] not in ops:
            ops.append(row["op"])

    tokens_list = sorted({row["tokens"] for row in rows})
    for op in ops:
        print("")
        print(f"op: {op}")
        header = (
            f"{'tokens':>8}  {'hc':>2}  {'hidden':>6}  {'iters':>5}  "
            f"{'naive total(ms)':>16}  {'naive avg(ms)':>14}  "
            f"{'kernel total(ms)':>17}  {'kernel avg(ms)':>15}  {'speedup':>8}"
        )
        print(header)
        print("-" * len(header))
        for tokens in tokens_list:
            naive = indexed.get((tokens, op, "naive"))
            kernel = indexed.get((tokens, op, "kernel"))
            if naive is None or kernel is None:
                continue
            speedup = naive["avg_ms"] / kernel["avg_ms"] if kernel["avg_ms"] > 0 else float("inf")
            print(
                f"{tokens:8d}  {naive['hc']:2d}  {naive['hidden']:6d}  {naive['iters']:5d}  "
                f"{naive['total_ms']:16.3f}  {naive['avg_ms']:14.6f}  "
                f"{kernel['total_ms']:17.3f}  {kernel['avg_ms']:15.6f}  {speedup:7.2f}x"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--hc", type=int, default=4)
    parser.add_argument("--tokens", type=str, default=DEFAULT_TOKENS)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--rms-eps", type=float, default=1e-6)
    parser.add_argument("--hc-eps", type=float, default=1e-6)
    parser.add_argument("--sinkhorn-iters", type=int, default=5)
    parser.add_argument("--check", dest="check", action="store_true", default=True)
    parser.add_argument("--no-check", dest="check", action="store_false")
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/Hygon device is required for MHC perf test")

    all_rows = []
    for tokens in _parse_int_list(args.tokens):
        all_rows.extend(_run_case(tokens, args.hc, args.hidden, args))

    _print_by_op(all_rows)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["tokens", "hc", "hidden", "op", "backend", "iters", "total_ms", "avg_ms"],
            )
            writer.writeheader()
            writer.writerows(all_rows)


if __name__ == "__main__":
    main()

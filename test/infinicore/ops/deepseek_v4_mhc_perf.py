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


def _bench(name, fn, warmup, iters):
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
        "op": name,
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

    x = torch.randn(tokens, hc, hidden, device=device, dtype=dtype)
    fn_pre = torch.randn(mix_hc, k, device=device, dtype=torch.float32) * 0.02
    hc_scale_pre = torch.randn(3, device=device, dtype=torch.float32) * 0.1
    hc_base_pre = torch.randn(mix_hc, device=device, dtype=torch.float32) * 0.1

    y_pre_naive = torch.empty(tokens, hidden, device=device, dtype=dtype)
    post_pre_naive = torch.empty(tokens, hc, device=device, dtype=torch.float32)
    comb_pre_naive = torch.empty(tokens, hc, hc, device=device, dtype=torch.float32)
    y_pre_kernel = torch.empty_like(y_pre_naive)
    post_pre_kernel = torch.empty_like(post_pre_naive)
    comb_pre_kernel = torch.empty_like(comb_pre_naive)
    y_pre_kernel_v2 = torch.empty_like(y_pre_naive)
    post_pre_kernel_v2 = torch.empty_like(post_pre_naive)
    comb_pre_kernel_v2 = torch.empty_like(comb_pre_naive)

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
        "hc_scale_pre": _wrap(hc_scale_pre, keepalive),
        "hc_base_pre": _wrap(hc_base_pre, keepalive),
        "y_pre_naive": _wrap(y_pre_naive, keepalive),
        "post_pre_naive": _wrap(post_pre_naive, keepalive),
        "comb_pre_naive": _wrap(comb_pre_naive, keepalive),
        "y_pre_kernel": _wrap(y_pre_kernel, keepalive),
        "post_pre_kernel": _wrap(post_pre_kernel, keepalive),
        "comb_pre_kernel": _wrap(comb_pre_kernel, keepalive),
        "y_pre_kernel_v2": _wrap(y_pre_kernel_v2, keepalive),
        "post_pre_kernel_v2": _wrap(post_pre_kernel_v2, keepalive),
        "comb_pre_kernel_v2": _wrap(comb_pre_kernel_v2, keepalive),
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
        "y_pre_kernel_v2": y_pre_kernel_v2,
        "post_pre_kernel_v2": post_pre_kernel_v2,
        "comb_pre_kernel_v2": comb_pre_kernel_v2,
        "y_post_naive": y_post_naive,
        "y_post_kernel": y_post_kernel,
        "y_head_naive": y_head_naive,
        "y_head_kernel": y_head_kernel,
    }
    return u, tensors, keepalive


def _run_case(tokens, hc, hidden, args):
    rms_eps = args.rms_eps
    hc_eps = args.hc_eps
    hc_pre_eps = args.hc_pre_eps
    hc_sinkhorn_eps = args.hc_sinkhorn_eps
    sinkhorn_repeat = args.sinkhorn_repeat
    u, tensors, keepalive = _make_case(tokens, hc, hidden, args.seed)

    def pre_naive():
        _infinicore.deepseek_v4_mhc_pre_naive_(
            u["y_pre_naive"],
            u["post_pre_naive"],
            u["comb_pre_naive"],
            u["x"],
            u["fn_pre"],
            u["hc_scale_pre"],
            u["hc_base_pre"],
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            sinkhorn_repeat,
        )

    def pre_kernel():
        _infinicore.deepseek_v4_mhc_pre_kernel_(
            u["y_pre_kernel"],
            u["post_pre_kernel"],
            u["comb_pre_kernel"],
            u["x"],
            u["fn_pre"],
            u["hc_scale_pre"],
            u["hc_base_pre"],
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            sinkhorn_repeat,
        )

    def pre_kernel_v2():
        _infinicore.deepseek_v4_mhc_pre_kernel_v2_(
            u["y_pre_kernel_v2"],
            u["post_pre_kernel_v2"],
            u["comb_pre_kernel_v2"],
            u["x"],
            u["fn_pre"],
            u["hc_scale_pre"],
            u["hc_base_pre"],
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            sinkhorn_repeat,
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
        _infinicore.deepseek_v4_hc_head_naive_(
            u["y_head_naive"],
            u["x"],
            u["fn_head"],
            u["scale_head"],
            u["base_head"],
            rms_eps,
            hc_eps,
        )

    def head_kernel():
        _infinicore.deepseek_v4_hc_head_kernel_(
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
            "mhc_pre_v2",
            pre_naive,
            pre_kernel_v2,
            (tensors["y_pre_naive"], tensors["post_pre_naive"], tensors["comb_pre_naive"]),
            (tensors["y_pre_kernel_v2"], tensors["post_pre_kernel_v2"], tensors["comb_pre_kernel_v2"]),
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
        ("mhc_pre", "kernel_v2", pre_kernel_v2),
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
                "median_ms": result["median_ms"],
            }
        )
    _sync()
    del keepalive
    torch.cuda.empty_cache()
    return rows


def _print_by_op(rows):
    indexed = {}
    ops = []
    backends_by_op = {}
    for row in rows:
        indexed[(row["tokens"], row["op"], row["backend"])] = row
        if row["op"] not in ops:
            ops.append(row["op"])
        backends_by_op.setdefault(row["op"], [])
        if row["backend"] not in backends_by_op[row["op"]]:
            backends_by_op[row["op"]].append(row["backend"])

    tokens_list = sorted({row["tokens"] for row in rows})
    for op in ops:
        print("")
        print(f"op: {op}")
        backends = backends_by_op[op]
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
            naive = indexed.get((tokens, op, "naive"))
            if naive is None:
                continue
            line = f"{tokens:8d}  {naive['hc']:2d}  {naive['hidden']:6d}  {naive['iters']:5d}"
            for backend in backends:
                row = indexed.get((tokens, op, backend))
                if row is None:
                    line += f"  {'-':>20}  {'-':>18}  {'-':>21}"
                    if backend != "naive":
                        line += f"  {'-':>18}"
                    continue
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
    parser.add_argument("--hc-pre-eps", type=float, default=1e-6)
    parser.add_argument("--hc-sinkhorn-eps", type=float, default=1e-6)
    parser.add_argument("--sinkhorn-repeat", "--sinkhorn-iters", dest="sinkhorn_repeat", type=int, default=5)
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
                fieldnames=["tokens", "hc", "hidden", "op", "backend", "iters", "total_ms", "avg_ms", "median_ms"],
            )
            writer.writeheader()
            writer.writerows(all_rows)


if __name__ == "__main__":
    main()

import argparse
import csv
import statistics
import time

import infinicore
import torch
from infinicore.lib import _infinicore


DEFAULT_TOKENS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"

DEFAULT_TOKENS = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128"


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


def _check_outputs(name, ref_outs, candidate_outs, atol=2e-2, rtol=2e-2):
    for idx, (lhs, rhs) in enumerate(zip(ref_outs, candidate_outs)):
        if not torch.allclose(lhs.float(), rhs.float(), atol=atol, rtol=rtol):
            raise AssertionError(
                f"{name}[{idx}] mismatch: max_abs={_max_abs(lhs, rhs)}"
            )


def _make_case(tokens, hc, hidden, seed):
    torch.manual_seed(seed + tokens * 17 + hidden)
    device = "cuda"
    dtype = torch.bfloat16
    mix_hc = (2 + hc) * hc
    k = hc * hidden

    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    residual = torch.randn(tokens, hc, hidden, device=device, dtype=dtype)
    post_layer_mix = torch.rand(tokens, hc, device=device, dtype=torch.float32) * 2.0
    comb_res_mix = torch.rand(tokens, hc, hc, device=device, dtype=torch.float32)
    comb_res_mix = (
        comb_res_mix / (comb_res_mix.sum(dim=1, keepdim=True) + 1e-6)
    ).contiguous()
    fn = torch.randn(mix_hc, k, device=device, dtype=torch.float32) * 0.02
    hc_scale = torch.randn(3, device=device, dtype=torch.float32) * 0.1
    hc_base = torch.randn(mix_hc, device=device, dtype=torch.float32) * 0.1
    norm_weight = torch.randn(hidden, device=device, dtype=dtype) * 0.1

    residual_sep = torch.empty(tokens, hc, hidden, device=device, dtype=dtype)
    post_sep = torch.empty(tokens, hc, device=device, dtype=torch.float32)
    comb_sep = torch.empty(tokens, hc, hc, device=device, dtype=torch.float32)
    layer_input_sep = torch.empty(tokens, hidden, device=device, dtype=dtype)

    residual_fused = torch.empty_like(residual_sep)
    post_fused = torch.empty_like(post_sep)
    comb_fused = torch.empty_like(comb_sep)
    layer_input_fused = torch.empty_like(layer_input_sep)

    residual_public = torch.empty_like(residual_sep)
    post_public = torch.empty_like(post_sep)
    comb_public = torch.empty_like(comb_sep)
    layer_input_public = torch.empty_like(layer_input_sep)

    residual_naive = torch.empty_like(residual_sep)
    post_naive = torch.empty_like(post_sep)
    comb_naive = torch.empty_like(comb_sep)
    layer_input_naive = torch.empty_like(layer_input_sep)

    keepalive = []
    u = {
        "x": _wrap(x, keepalive),
        "residual": _wrap(residual, keepalive),
        "post_layer_mix": _wrap(post_layer_mix, keepalive),
        "comb_res_mix": _wrap(comb_res_mix, keepalive),
        "fn": _wrap(fn, keepalive),
        "hc_scale": _wrap(hc_scale, keepalive),
        "hc_base": _wrap(hc_base, keepalive),
        "norm_weight": _wrap(norm_weight, keepalive),
        "residual_sep": _wrap(residual_sep, keepalive),
        "post_sep": _wrap(post_sep, keepalive),
        "comb_sep": _wrap(comb_sep, keepalive),
        "layer_input_sep": _wrap(layer_input_sep, keepalive),
        "residual_fused": _wrap(residual_fused, keepalive),
        "post_fused": _wrap(post_fused, keepalive),
        "comb_fused": _wrap(comb_fused, keepalive),
        "layer_input_fused": _wrap(layer_input_fused, keepalive),
        "residual_public": _wrap(residual_public, keepalive),
        "post_public": _wrap(post_public, keepalive),
        "comb_public": _wrap(comb_public, keepalive),
        "layer_input_public": _wrap(layer_input_public, keepalive),
        "residual_naive": _wrap(residual_naive, keepalive),
        "post_naive": _wrap(post_naive, keepalive),
        "comb_naive": _wrap(comb_naive, keepalive),
        "layer_input_naive": _wrap(layer_input_naive, keepalive),
    }
    tensors = {
        "residual_sep": residual_sep,
        "post_sep": post_sep,
        "comb_sep": comb_sep,
        "layer_input_sep": layer_input_sep,
        "residual_fused": residual_fused,
        "post_fused": post_fused,
        "comb_fused": comb_fused,
        "layer_input_fused": layer_input_fused,
        "residual_public": residual_public,
        "post_public": post_public,
        "comb_public": comb_public,
        "layer_input_public": layer_input_public,
        "residual_naive": residual_naive,
        "post_naive": post_naive,
        "comb_naive": comb_naive,
        "layer_input_naive": layer_input_naive,
    }
    return u, tensors, keepalive


def _run_case(tokens, hc, hidden, args):
    rms_eps = args.rms_eps
    hc_pre_eps = args.hc_pre_eps
    hc_sinkhorn_eps = args.hc_sinkhorn_eps
    norm_eps = args.norm_eps
    hc_post_mult_value = args.hc_post_mult_value
    sinkhorn_repeat = args.sinkhorn_repeat
    u, tensors, keepalive = _make_case(tokens, hc, hidden, args.seed)

    def separate_kernel():
        _infinicore.deepseek_v4_mhc_post_kernel_(
            u["residual_sep"],
            u["x"],
            u["residual"],
            u["post_layer_mix"],
            u["comb_res_mix"],
        )
        _infinicore.deepseek_v4_mhc_pre_kernel_(
            u["layer_input_sep"],
            u["post_sep"],
            u["comb_sep"],
            u["residual_sep"],
            u["fn"],
            u["hc_scale"],
            u["hc_base"],
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            sinkhorn_repeat,
        )
        _infinicore.deepseek_v4_rms_norm_(
            u["layer_input_sep"],
            u["layer_input_sep"],
            u["norm_weight"],
            norm_eps,
        )

    def fused_kernel():
        _infinicore.deepseek_v4_mhc_fused_post_pre_kernel_(
            u["residual_fused"],
            u["post_fused"],
            u["comb_fused"],
            u["layer_input_fused"],
            u["x"],
            u["residual"],
            u["post_layer_mix"],
            u["comb_res_mix"],
            u["fn"],
            u["hc_scale"],
            u["hc_base"],
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            u["norm_weight"],
            norm_eps,
        )

    def fused_public():
        _infinicore.deepseek_v4_mhc_fused_post_pre_(
            u["residual_public"],
            u["post_public"],
            u["comb_public"],
            u["layer_input_public"],
            u["x"],
            u["residual"],
            u["post_layer_mix"],
            u["comb_res_mix"],
            u["fn"],
            u["hc_scale"],
            u["hc_base"],
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            u["norm_weight"],
            norm_eps,
        )

    def fused_naive():
        _infinicore.deepseek_v4_mhc_fused_post_pre_naive_(
            u["residual_naive"],
            u["post_naive"],
            u["comb_naive"],
            u["layer_input_naive"],
            u["x"],
            u["residual"],
            u["post_layer_mix"],
            u["comb_res_mix"],
            u["fn"],
            u["hc_scale"],
            u["hc_base"],
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            u["norm_weight"],
            norm_eps,
        )

    if args.check:
        separate_kernel()
        fused_kernel()
        fused_public()
        _sync()
        ref = (
            tensors["residual_sep"],
            tensors["layer_input_sep"],
            tensors["post_sep"],
            tensors["comb_sep"],
        )
        _check_outputs(
            "fused_kernel",
            ref,
            (
                tensors["residual_fused"],
                tensors["layer_input_fused"],
                tensors["post_fused"],
                tensors["comb_fused"],
            ),
        )
        _check_outputs(
            "fused_public",
            ref,
            (
                tensors["residual_public"],
                tensors["layer_input_public"],
                tensors["post_public"],
                tensors["comb_public"],
            ),
        )
        if args.check_naive:
            fused_naive()
            _sync()
            _check_outputs(
                "fused_naive",
                ref,
                (
                    tensors["residual_naive"],
                    tensors["layer_input_naive"],
                    tensors["post_naive"],
                    tensors["comb_naive"],
                ),
            )

    rows = []
    for backend, fn in (
        ("separate_kernel", separate_kernel),
        ("fused_kernel", fused_kernel),
        ("fused_public", fused_public),
    ):
        result = _bench(backend, fn, args.warmup, args.iters)
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
    if args.bench_naive:
        result = _bench("fused_naive", fused_naive, args.warmup, args.iters)
        rows.append(
            {
                "tokens": tokens,
                "hc": hc,
                "hidden": hidden,
                "backend": "fused_naive",
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
            f"  {backend + ' total(ms)':>26}"
            f"  {backend + ' avg(ms)':>24}"
            f"  {backend + ' median(ms)':>27}"
        )
        if backend != "separate_kernel":
            header += f"  {backend + ' speedup':>24}"
    print(header)
    print("-" * len(header))

    for tokens in tokens_list:
        base = indexed.get((tokens, "separate_kernel"))
        if base is None:
            continue
        line = f"{tokens:8d}  {base['hc']:2d}  {base['hidden']:6d}  {base['iters']:5d}"
        for backend in backends:
            row = indexed.get((tokens, backend))
            if row is None:
                line += f"  {'-':>26}  {'-':>24}  {'-':>27}"
                if backend != "separate_kernel":
                    line += f"  {'-':>24}"
                continue
            line += f"  {row['total_ms']:26.3f}  {row['avg_ms']:24.6f}  {row['median_ms']:27.6f}"
            if backend != "separate_kernel":
                speedup = (
                    base["avg_ms"] / row["avg_ms"]
                    if row["avg_ms"] > 0
                    else float("inf")
                )
                line += f"  {speedup:23.2f}x"
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
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--rms-eps", type=float, default=1e-6)
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--hc-pre-eps", type=float, default=1e-6)
    parser.add_argument("--hc-sinkhorn-eps", type=float, default=1e-6)
    parser.add_argument("--hc-post-mult-value", type=float, default=2.0)
    parser.add_argument(
        "--sinkhorn-repeat",
        "--sinkhorn-iters",
        dest="sinkhorn_repeat",
        type=int,
        default=20,
    )
    parser.add_argument("--check", dest="check", action="store_true", default=True)
    parser.add_argument("--no-check", dest="check", action="store_false")
    parser.add_argument("--check-naive", action="store_true")
    parser.add_argument("--bench-naive", action="store_true")
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA/Hygon device is required for MHC fused_post_pre perf test"
        )

    all_rows = []
    for tokens in _parse_int_list(args.tokens):
        all_rows.extend(_run_case(tokens, args.hc, args.hidden, args))

    _print_rows(all_rows)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "tokens",
                    "hc",
                    "hidden",
                    "backend",
                    "iters",
                    "total_ms",
                    "avg_ms",
                    "median_ms",
                ],
            )
            writer.writeheader()
            writer.writerows(all_rows)


if __name__ == "__main__":
    main()

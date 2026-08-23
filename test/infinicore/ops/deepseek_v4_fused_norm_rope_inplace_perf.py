import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from infinicore.lib import _infinicore


DEFAULT_TOKENS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"
DEFAULT_HEAD_DIM = 512
DEFAULT_ROPE_DIM = 64
DEFAULT_MAX_POS = 1048576
DEFAULT_EPS = 1e-6
DEFAULT_POS_DTYPE = "int64"


def _parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _torch_pos_dtype(name):
    if name == "int32":
        return torch.int32
    if name == "int64":
        return torch.int64
    raise ValueError(f"unsupported position dtype: {name}")


def _as_core(tensor, keepalive):
    base = infinicore.from_torch(tensor)
    keepalive.append(base)
    return base._underlying


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
        samples.append((time.perf_counter() - start) * 1000.0)

    total_ms = sum(samples)
    return {
        "total_ms": total_ms,
        "avg_ms": total_ms / float(iters),
        "median_ms": statistics.median(samples),
    }


def _make_freqs(max_pos, dim=DEFAULT_ROPE_DIM, device="cuda"):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    angles = torch.outer(t, inv_freq)
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).flatten(-2).contiguous()


def _make_case(tokens, freqs, pos_dtype, seed):
    torch.manual_seed(seed + tokens * 17)
    kv = torch.randn((tokens, DEFAULT_HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    norm_weight = torch.randn((DEFAULT_HEAD_DIM,), device="cuda", dtype=torch.bfloat16)
    naive_input = kv.clone()
    kernel_input = kv.clone()
    positions = ((torch.arange(tokens, device="cuda", dtype=pos_dtype) * 5) % freqs.shape[0]).contiguous()
    keepalive = []
    tensors = {
        "kv": kv,
        "naive_input": naive_input,
        "kernel_input": kernel_input,
        "norm_weight": norm_weight,
        "freqs": freqs,
        "positions": positions,
    }
    core = {
        "naive_input": _as_core(naive_input, keepalive),
        "kernel_input": _as_core(kernel_input, keepalive),
        "norm_weight": _as_core(norm_weight, keepalive),
        "freqs": _as_core(freqs, keepalive),
        "positions": _as_core(positions, keepalive),
    }
    return tensors, core, keepalive


def _run_naive(core, eps):
    _infinicore.deepseek_v4_fused_norm_rope_inplace_naive_(
        core["naive_input"],
        core["norm_weight"],
        eps,
        core["freqs"],
        core["positions"],
    )


def _run_kernel(core, eps):
    _infinicore.deepseek_v4_fused_norm_rope_inplace_kernel_(
        core["kernel_input"],
        core["norm_weight"],
        eps,
        core["freqs"],
        core["positions"],
    )


def _check_result(naive_input, kernel_input, atol, rtol):
    _sync()
    max_abs = (naive_input.float() - kernel_input.float()).abs().max().item()
    allclose = torch.allclose(naive_input, kernel_input, atol=atol, rtol=rtol)
    if not allclose:
        raise AssertionError(f"kernel mismatch: max_abs={max_abs:.6e}")
    return max_abs, allclose


def _run_case(tokens, freqs, pos_dtype, args):
    tensors, core, keepalive = _make_case(tokens, freqs, pos_dtype, args.seed)

    def naive_fn():
        _run_naive(core, args.eps)

    def kernel_fn():
        _run_kernel(core, args.eps)

    max_abs = float("nan")
    allclose = "skip"
    if args.check:
        naive_fn()
        kernel_fn()
        max_abs, allclose = _check_result(tensors["naive_input"], tensors["kernel_input"], args.atol, args.rtol)
        tensors["naive_input"].copy_(tensors["kv"])
        tensors["kernel_input"].copy_(tensors["kv"])
        _sync()

    naive = _bench(naive_fn, args.warmup, args.iters)
    kernel = _bench(kernel_fn, args.warmup, args.iters)
    del keepalive
    return {
        "tokens": tokens,
        "rows": tokens,
        "stride0": tensors["kernel_input"].stride(0),
        "naive": naive,
        "kernel": kernel,
        "speedup_avg": naive["avg_ms"] / kernel["avg_ms"] if kernel["avg_ms"] > 0 else float("inf"),
        "speedup_median": naive["median_ms"] / kernel["median_ms"] if kernel["median_ms"] > 0 else float("inf"),
        "max_abs": max_abs,
        "allclose": allclose,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek-V4 fused_norm_rope_inplace over DEFAULT_TOKENS.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--max-pos", type=int, default=DEFAULT_MAX_POS)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--pos-dtype", choices=("int32", "int64"), default=DEFAULT_POS_DTYPE)
    parser.add_argument("--check", action="store_true", default=True)
    parser.add_argument("--no-check", dest="check", action="store_false")
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    tokens_list = _parse_int_list(args.tokens)
    pos_dtype = _torch_pos_dtype(args.pos_dtype)
    freqs = _make_freqs(max_pos=args.max_pos, device="cuda")
    print("DeepSeek-V4 fused_norm_rope_inplace 性能测试")
    print(
        f"tokens={args.tokens} head_dim={DEFAULT_HEAD_DIM} rope_dim={DEFAULT_ROPE_DIM} "
        f"max_pos={args.max_pos} eps={args.eps} pos_dtype={args.pos_dtype} "
        f"iters={args.iters} warmup={args.warmup} check={args.check}"
    )
    print(
        f"{'tokens':>8} | {'rows':>8} | {'stride0':>8} | "
        f"{'naive avg':>10} | {'naive med':>10} | "
        f"{'kernel avg':>10} | {'kernel med':>10} | "
        f"{'spd avg':>8} | {'spd med':>8} | {'max_abs':>10} | {'allclose':>8}"
    )
    print("-" * 132)
    for tokens in tokens_list:
        result = _run_case(tokens, freqs, pos_dtype, args)
        naive = result["naive"]
        kernel = result["kernel"]
        print(
            f"{result['tokens']:8d} | "
            f"{result['rows']:8d} | "
            f"{result['stride0']:8d} | "
            f"{naive['avg_ms']:10.4f} | "
            f"{naive['median_ms']:10.4f} | "
            f"{kernel['avg_ms']:10.4f} | "
            f"{kernel['median_ms']:10.4f} | "
            f"{result['speedup_avg']:8.2f} | "
            f"{result['speedup_median']:8.2f} | "
            f"{result['max_abs']:10.4e} | "
            f"{str(result['allclose']):>8}"
        )


if __name__ == "__main__":
    main()

import argparse
import statistics
import time

import infinicore
import torch
from infinicore.lib import _infinicore


DEFAULT_TOKENS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"
DEFAULT_HEAD_DIM = 512
DEFAULT_GROUPS_C4 = 2
DEFAULT_GROUPS_C128 = 1


def _parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _as_core(tensor, keepalive=None):
    wrapped = infinicore.from_torch(tensor)
    if keepalive is not None:
        keepalive.append(wrapped)
    return wrapped._underlying


def _sync():
    infinicore.sync_stream()
    torch.cuda.synchronize()


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
        "avg_ms": total_ms / float(iters),
        "median_ms": statistics.median(samples),
    }


def _make_c4_case(tokens, head_dim, groups, seed):
    torch.manual_seed(seed + tokens * 17 + 4)
    kv_score = torch.randn((tokens, 4 * head_dim), device="cuda", dtype=torch.bfloat16)
    ape = torch.randn((8, head_dim), device="cuda", dtype=torch.bfloat16)
    base_state = torch.randn((4 * groups, 4 * head_dim), device="cuda", dtype=torch.float32)
    write_loc = (torch.arange(tokens, device="cuda", dtype=torch.int32) // 4) % groups
    extra_loc = torch.clamp(write_loc - 1, min=-1).reshape(tokens, 1).contiguous()
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32)
    return {
        "kv_score": kv_score,
        "ape": ape,
        "base_state": base_state,
        "write_loc": write_loc.contiguous(),
        "extra_loc": extra_loc,
        "positions": positions,
    }


def _make_c128_case(tokens, head_dim, groups, seed):
    torch.manual_seed(seed + tokens * 17 + 128)
    kv_score = torch.randn((tokens, 2 * head_dim), device="cuda", dtype=torch.bfloat16)
    ape = torch.randn((128, head_dim), device="cuda", dtype=torch.bfloat16)
    base_state = torch.randn((128 * groups, 2 * head_dim), device="cuda", dtype=torch.float32)
    write_loc = (torch.arange(tokens, device="cuda", dtype=torch.int32) // 128) % groups
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32)
    return {
        "kv_score": kv_score,
        "ape": ape,
        "base_state": base_state,
        "write_loc": write_loc.contiguous(),
        "positions": positions,
    }


def _bench_c4(tokens, args):
    case = _make_c4_case(tokens, args.head_dim, args.c4_groups, args.seed)
    state_naive = case["base_state"].clone()
    state_kernel = case["base_state"].clone()
    state_sglang = case["base_state"].clone()
    keepalive = []
    core = {
        "kv_score": _as_core(case["kv_score"], keepalive),
        "ape": _as_core(case["ape"], keepalive),
        "state_naive": _as_core(state_naive, keepalive),
        "state_kernel": _as_core(state_kernel, keepalive),
        "state_sglang": _as_core(state_sglang, keepalive),
        "write_loc": _as_core(case["write_loc"], keepalive),
        "extra_loc": _as_core(case["extra_loc"], keepalive),
        "positions": _as_core(case["positions"], keepalive),
    }

    def naive_fn():
        _infinicore.deepseek_v4_c4_compress_stateful_naive(
            core["kv_score"],
            core["ape"],
            core["state_naive"],
            core["write_loc"],
            core["extra_loc"],
            core["positions"],
        )

    def kernel_fn():
        _infinicore.deepseek_v4_c4_compress_stateful_kernel(
            core["kv_score"],
            core["ape"],
            core["state_kernel"],
            core["write_loc"],
            core["extra_loc"],
            core["positions"],
        )

    def sglang_fn():
        _infinicore.deepseek_v4_c4_compress_sglang_stateful_kernel(
            core["kv_score"],
            core["ape"],
            core["state_sglang"],
            core["write_loc"],
            core["extra_loc"],
            core["positions"],
        )

    naive = _bench(naive_fn, args.warmup, args.iters)
    kernel = _bench(kernel_fn, args.warmup, args.iters)
    sglang = _bench(sglang_fn, args.warmup, args.iters)
    del keepalive
    return naive, kernel, sglang


def _bench_c128(tokens, args):
    case = _make_c128_case(tokens, args.head_dim, args.c128_groups, args.seed)
    state_naive = case["base_state"].clone()
    state_kernel = case["base_state"].clone()
    state_sglang = case["base_state"].clone()
    keepalive = []
    core = {
        "kv_score": _as_core(case["kv_score"], keepalive),
        "ape": _as_core(case["ape"], keepalive),
        "state_naive": _as_core(state_naive, keepalive),
        "state_kernel": _as_core(state_kernel, keepalive),
        "state_sglang": _as_core(state_sglang, keepalive),
        "write_loc": _as_core(case["write_loc"], keepalive),
        "positions": _as_core(case["positions"], keepalive),
    }

    def naive_fn():
        _infinicore.deepseek_v4_c128_compress_stateful_naive(
            core["kv_score"],
            core["ape"],
            core["state_naive"],
            core["write_loc"],
            core["positions"],
        )

    def kernel_fn():
        _infinicore.deepseek_v4_c128_compress_stateful_kernel(
            core["kv_score"],
            core["ape"],
            core["state_kernel"],
            core["write_loc"],
            core["positions"],
        )

    def sglang_fn():
        _infinicore.deepseek_v4_c128_compress_sglang_stateful_kernel(
            core["kv_score"],
            core["ape"],
            core["state_sglang"],
            core["write_loc"],
            core["positions"],
        )

    iters = max(1, args.iters // max(1, args.c128_iter_divisor))
    naive = _bench(naive_fn, args.warmup, iters)
    kernel = _bench(kernel_fn, args.warmup, iters)
    sglang = _bench(sglang_fn, args.warmup, iters)
    del keepalive
    return naive, kernel, sglang


def _print_header(title, args):
    print(title)
    print(
        f"tokens={args.tokens} head_dim={args.head_dim} iters={args.iters} "
        f"warmup={args.warmup} c4_groups={args.c4_groups} c128_groups={args.c128_groups}"
    )
    print(
        f"{'tokens':>8} | {'naive avg':>10} | {'naive med':>10} | "
        f"{'kernel avg':>10} | {'kernel med':>10} | "
        f"{'sglang avg':>10} | {'sglang med':>10} | "
        f"{'sg/kn avg':>9} | {'sg/kn med':>9} | {'sg/nv avg':>9}"
    )
    print("-" * 122)


def _print_result(tokens, naive, kernel, sglang):
    kernel_speedup_avg = kernel["avg_ms"] / sglang["avg_ms"] if sglang["avg_ms"] > 0 else float("inf")
    kernel_speedup_median = kernel["median_ms"] / sglang["median_ms"] if sglang["median_ms"] > 0 else float("inf")
    naive_speedup_avg = naive["avg_ms"] / sglang["avg_ms"] if sglang["avg_ms"] > 0 else float("inf")
    print(
        f"{tokens:8d} | "
        f"{naive['avg_ms']:10.4f} | "
        f"{naive['median_ms']:10.4f} | "
        f"{kernel['avg_ms']:10.4f} | "
        f"{kernel['median_ms']:10.4f} | "
        f"{sglang['avg_ms']:10.4f} | "
        f"{sglang['median_ms']:10.4f} | "
        f"{kernel_speedup_avg:9.2f} | "
        f"{kernel_speedup_median:9.2f} | "
        f"{naive_speedup_avg:9.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek-V4 compress_sglang_stateful kernels.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--c4-groups", type=int, default=DEFAULT_GROUPS_C4)
    parser.add_argument("--c128-groups", type=int, default=DEFAULT_GROUPS_C128)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--c128-iter-divisor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    tokens_list = _parse_int_list(args.tokens)
    _print_header("DeepSeek-V4 C4 compress_sglang_stateful 性能测试", args)
    for tokens in tokens_list:
        naive, kernel, sglang = _bench_c4(tokens, args)
        _print_result(tokens, naive, kernel, sglang)

    print()
    _print_header("DeepSeek-V4 C128 compress_sglang_stateful 性能测试", args)
    for tokens in tokens_list:
        naive, kernel, sglang = _bench_c128(tokens, args)
        _print_result(tokens, naive, kernel, sglang)


if __name__ == "__main__":
    main()

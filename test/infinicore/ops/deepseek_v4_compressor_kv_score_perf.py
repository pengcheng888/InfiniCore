import argparse
import statistics
import time

import infinicore
import torch
from infinicore.lib import _infinicore


DSV4_HIDDEN = 4096
DSV4_HEAD_DIM = 512
DEFAULT_TOKENS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"


def _parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _wrap(tensor, keepalive):
    wrapped = infinicore.from_torch(tensor)
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
    return statistics.mean(samples), statistics.median(samples)


def _run_case(name, tokens, proj_size, args):
    torch.manual_seed(args.seed + tokens * 17 + proj_size)
    x = torch.randn((tokens, DSV4_HIDDEN), device="cuda", dtype=torch.bfloat16)
    wkv = torch.randn((proj_size, DSV4_HIDDEN), device="cuda", dtype=torch.bfloat16)
    wgate = torch.randn((proj_size, DSV4_HIDDEN), device="cuda", dtype=torch.bfloat16)
    wkv_gate = torch.cat([wkv, wgate], dim=0).contiguous()

    out_shape = (tokens, proj_size * 2)
    unpacked = torch.empty(out_shape, device="cuda", dtype=torch.bfloat16)
    packed = torch.empty_like(unpacked)

    keepalive = []
    x_core = _wrap(x, keepalive)
    wkv_core = _wrap(wkv, keepalive)
    wgate_core = _wrap(wgate, keepalive)
    wkv_gate_core = _wrap(wkv_gate, keepalive)
    unpacked_core = _wrap(unpacked, keepalive)
    packed_core = _wrap(packed, keepalive)

    def unpacked_fn():
        _infinicore.deepseek_v4_compressor_kv_score_unpacked_(unpacked_core, x_core, wkv_core, wgate_core)

    def packed_fn():
        _infinicore.deepseek_v4_compressor_kv_score_packed_(packed_core, x_core, wkv_gate_core)

    max_abs = float("nan")
    ok = "skip"
    if args.check:
        unpacked_fn()
        packed_fn()
        _sync()
        max_abs = (packed.float() - unpacked.float()).abs().max().item()
        ok = str(torch.allclose(packed, unpacked, atol=args.atol, rtol=args.rtol))

    unpacked_avg, unpacked_median = _bench(unpacked_fn, args.warmup, args.iters)
    packed_avg, packed_median = _bench(packed_fn, args.warmup, args.iters)
    avg_speedup = unpacked_avg / packed_avg if packed_avg > 0 else float("inf")
    median_speedup = unpacked_median / packed_median if packed_median > 0 else float("inf")

    return {
        "case": name,
        "tokens": tokens,
        "proj": proj_size,
        "out": proj_size * 2,
        "unpacked_avg": unpacked_avg,
        "unpacked_median": unpacked_median,
        "packed_avg": packed_avg,
        "packed_median": packed_median,
        "avg_speedup": avg_speedup,
        "median_speedup": median_speedup,
        "max_abs": max_abs,
        "ok": ok,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek-V4 compressor kv-score packed/unpacked InfiniCore paths.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--cases", default="c4,c128", help="Comma-separated list from: c4,c128")
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--check", action="store_true", default=True)
    parser.add_argument("--no-check", dest="check", action="store_false")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()

    case_to_proj = {
        "c4": 2 * DSV4_HEAD_DIM,
        "c128": DSV4_HEAD_DIM,
    }
    tokens_list = _parse_int_list(args.tokens)
    cases = [item.strip().lower() for item in args.cases.split(",") if item.strip()]

    print("DeepSeek-V4 compressor kv-score packed/unpacked performance")
    print(f"hidden={DSV4_HIDDEN} head_dim={DSV4_HEAD_DIM} iters={args.iters} warmup={args.warmup} check={args.check}")
    print(
        f"{'case':>5} | {'tokens':>6} | {'proj':>5} | {'out':>5} | "
        f"{'unpack avg':>10} | {'pack avg':>9} | {'avg spd':>7} | "
        f"{'unpack med':>10} | {'pack med':>9} | {'med spd':>7} | {'max_abs':>10} | {'ok':>5}"
    )
    print("-" * 124)
    for case in cases:
        if case not in case_to_proj:
            raise ValueError(f"unsupported case: {case}")
        for tokens in tokens_list:
            result = _run_case(case, tokens, case_to_proj[case], args)
            max_abs = result["max_abs"]
            max_abs_text = "nan" if max_abs != max_abs else f"{max_abs:.4e}"
            print(
                f"{result['case']:>5} | {result['tokens']:6d} | {result['proj']:5d} | {result['out']:5d} | "
                f"{result['unpacked_avg']:10.4f} | {result['packed_avg']:9.4f} | {result['avg_speedup']:7.2f} | "
                f"{result['unpacked_median']:10.4f} | {result['packed_median']:9.4f} | {result['median_speedup']:7.2f} | "
                f"{max_abs_text:>10} | {result['ok']:>5}"
            )


if __name__ == "__main__":
    main()

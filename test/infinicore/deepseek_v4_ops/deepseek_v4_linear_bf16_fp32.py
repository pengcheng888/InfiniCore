import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
from infinicore.lib import _infinicore


DEFAULT_TOKENS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"


def _parse_tokens(text):
    return [int(item) for item in text.split(",") if item.strip()]


def _as_core(tensor):
    return infinicore.from_torch(tensor)


def _make_inputs(tokens, hidden, out_features, seed):
    torch.manual_seed(seed + tokens * 17 + hidden + out_features)
    device = torch.device("cuda")
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16).contiguous()
    weight = torch.randn(out_features, hidden, device=device, dtype=torch.bfloat16).contiguous()
    return x, weight


def _max_diff(lhs, rhs):
    lhs_f = lhs.float()
    rhs_f = rhs.float()
    abs_diff = (lhs_f - rhs_f).abs()
    max_abs = abs_diff.max().item() if abs_diff.numel() > 0 else 0.0
    denom = rhs_f.abs().clamp_min(1e-6)
    max_rel = (abs_diff / denom).max().item() if abs_diff.numel() > 0 else 0.0
    return max_abs, max_rel


def _bench(fn, warmup, iters):
    warmup_value = None
    for _ in range(warmup):
        warmup_value = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    return {
        "avg_ms": total_ms / iters,
        "total_ms": total_ms,
        "warmup_value": warmup_value,
    }


def _run_case(tokens, args):
    x, weight = _make_inputs(tokens, args.hidden, args.out_features, args.seed)
    aten_out = torch.empty(tokens, args.out_features, device=x.device, dtype=torch.float32)
    kernel_out = torch.empty_like(aten_out)

    core_x = _as_core(x)
    core_weight = _as_core(weight)
    core_aten_out = _as_core(aten_out)
    core_kernel_out = _as_core(kernel_out)

    def run_aten():
        _infinicore.deepseek_v4_linear_bf16_fp32_aten_(
            core_aten_out._underlying,
            core_x._underlying,
            core_weight._underlying,
        )
        return aten_out

    def run_kernel():
        _infinicore.deepseek_v4_linear_bf16_fp32_(
            core_kernel_out._underlying,
            core_x._underlying,
            core_weight._underlying,
        )
        return kernel_out

    aten_perf = _bench(run_aten, args.warmup, args.iters)
    kernel_perf = _bench(run_kernel, args.warmup, args.iters)

    ref = aten_perf["warmup_value"]
    got = kernel_perf["warmup_value"]
    max_abs, max_rel = _max_diff(got, ref)
    allclose = torch.allclose(got, ref, atol=args.atol, rtol=args.rtol)

    return {
        "tokens": tokens,
        "aten_avg": aten_perf["avg_ms"],
        "kernel_avg": kernel_perf["avg_ms"],
        "kernel_speedup": aten_perf["avg_ms"] / kernel_perf["avg_ms"] if kernel_perf["avg_ms"] > 0 else float("inf"),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "allclose": bool(allclose),
    }


def _fmt_diff(value):
    return "nan" if value != value else f"{value:.6g}"


def _print_header(args):
    print(f"hidden={args.hidden} out_features={args.out_features}")
    print(
        f"{'tokens':>8} | {'aten avg':>10} | {'kernel avg':>10} | {'kernel spd':>10} | "
        f"{'max_abs':>10} | {'max_rel':>10} | {'allclose':>8}"
    )
    print("-" * 81)


def _print_row(result):
    print(
        f"{result['tokens']:8d} | "
        f"{result['aten_avg']:10.4f} | "
        f"{result['kernel_avg']:10.4f} | "
        f"{result['kernel_speedup']:10.2f} | "
        f"{_fmt_diff(result['max_abs']):>10} | "
        f"{_fmt_diff(result['max_rel']):>10} | "
        f"{str(result['allclose']):>8}"
    )


def main():
    parser = argparse.ArgumentParser(description="Check and benchmark DeepSeek-V4 bf16->fp32 linear.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    args = parser.parse_args()

    ok = True
    _print_header(args)
    for tokens in _parse_tokens(args.tokens):
        result = _run_case(tokens, args)
        _print_row(result)
        if result["allclose"] is False:
            ok = False
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

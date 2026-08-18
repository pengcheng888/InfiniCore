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


def _make_inputs(tokens, experts, seed):
    torch.manual_seed(seed + tokens * 17 + experts)
    device = torch.device("cuda")
    router_logits = torch.randn(tokens, experts, dtype=torch.float32, device=device).contiguous()
    bias = (torch.randn(experts, dtype=torch.float32, device=device) * 0.2).contiguous()
    return router_logits, bias


def _make_outputs(tokens, topk):
    device = torch.device("cuda")
    weights = torch.empty(tokens, topk, dtype=torch.float32, device=device)
    indices = torch.empty(tokens, topk, dtype=torch.int32, device=device)
    return weights, indices


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
    router_logits, bias = _make_inputs(tokens, args.experts, args.seed)
    aten_weights, aten_indices = _make_outputs(tokens, args.topk)
    kernel_weights, kernel_indices = _make_outputs(tokens, args.topk)

    core_router_logits = _as_core(router_logits)
    core_bias = _as_core(bias)
    core_aten_weights = _as_core(aten_weights)
    core_aten_indices = _as_core(aten_indices)
    core_kernel_weights = _as_core(kernel_weights)
    core_kernel_indices = _as_core(kernel_indices)

    def run_aten():
        _infinicore.deepseek_v4_topk_aten_(
            core_aten_weights._underlying,
            core_aten_indices._underlying,
            core_router_logits._underlying,
            core_bias._underlying,
            args.renormalize,
        )
        return aten_weights, aten_indices

    def run_kernel():
        _infinicore.deepseek_v4_topk_(
            core_kernel_weights._underlying,
            core_kernel_indices._underlying,
            core_router_logits._underlying,
            core_bias._underlying,
            args.renormalize,
        )
        return kernel_weights, kernel_indices

    aten_perf = _bench(run_aten, args.warmup, args.iters)
    kernel_perf = _bench(run_kernel, args.warmup, args.iters)
    ref_weights, ref_indices = aten_perf["warmup_value"]
    got_weights, got_indices = kernel_perf["warmup_value"]
    max_abs, max_rel = _max_diff(got_weights, ref_weights)
    allclose = torch.equal(got_indices, ref_indices) and torch.allclose(
        got_weights, ref_weights, atol=args.atol, rtol=args.rtol
    )

    return {
        "tokens": tokens,
        "aten_avg": aten_perf["avg_ms"],
        "kernel_avg": kernel_perf["avg_ms"],
        "speedup": aten_perf["avg_ms"] / kernel_perf["avg_ms"] if kernel_perf["avg_ms"] > 0 else float("inf"),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "allclose": bool(allclose),
    }


def _print_header(args):
    print(f"experts={args.experts} topk={args.topk} renormalize={args.renormalize}")
    print(
        f"{'tokens':>8} | {'aten avg':>10} | {'kernel avg':>10} | "
        f"{'speedup':>8} | {'max_abs':>10} | {'max_rel':>10} | {'allclose':>8}"
    )
    print("-" * 86)


def _print_row(result):
    print(
        f"{result['tokens']:8d} | "
        f"{result['aten_avg']:10.4f} | "
        f"{result['kernel_avg']:10.4f} | "
        f"{result['speedup']:8.2f} | "
        f"{result['max_abs']:10.6g} | "
        f"{result['max_rel']:10.6g} | "
        f"{str(result['allclose']):>8}"
    )


def main():
    parser = argparse.ArgumentParser(description="Check and benchmark DeepSeek-V4 biased topk.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--renormalize", action="store_true", default=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()

    if args.experts != 256 or args.topk != 6:
        raise RuntimeError("DeepSeek V4 biased_topk kernel expects experts=256 and topk=6.")

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

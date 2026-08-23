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


def _make_inputs(tokens, hc, hidden, seed):
    torch.manual_seed(seed + tokens * 17 + hidden + hc * 101)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    k = hc * hidden

    x = torch.randn((tokens, hc, hidden), device=device, dtype=dtype).contiguous()
    fn = (torch.randn((hc, k), device=device, dtype=torch.float32) * 0.02).contiguous()
    scale = (torch.randn((1,), device=device, dtype=torch.float32) * 0.1).contiguous()
    base = (torch.randn((hc,), device=device, dtype=torch.float32) * 0.1).contiguous()
    return x, fn, scale, base


def _aten_ref(core_out, core_x, core_fn, core_scale, core_base, args, out):
    _infinicore.deepseek_v4_hc_head_aten_(
        core_out._underlying,
        core_x._underlying,
        core_fn._underlying,
        core_scale._underlying,
        core_base._underlying,
        args.rms_eps,
        args.hc_eps,
    )
    return out


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


def _run_case(tokens, hc, hidden, args):
    x, fn, scale, base = _make_inputs(tokens, hc, hidden, args.seed)
    core_x = _as_core(x)
    core_fn = _as_core(fn)
    core_scale = _as_core(scale)
    core_base = _as_core(base)

    ref_out = torch.empty((tokens, hidden), device=torch.device("cuda"), dtype=torch.bfloat16)
    op_out = torch.empty_like(ref_out)
    core_ref_out = _as_core(ref_out)
    core_op_out = _as_core(op_out)

    def run_aten():
        return _aten_ref(core_ref_out, core_x, core_fn, core_scale, core_base, args, ref_out)

    def run_op():
        _infinicore.deepseek_v4_hc_head_(
            core_op_out._underlying,
            core_x._underlying,
            core_fn._underlying,
            core_scale._underlying,
            core_base._underlying,
            args.rms_eps,
            args.hc_eps,
        )
        return op_out

    aten_perf = _bench(run_aten, args.warmup, args.iters)
    op_perf = _bench(run_op, args.warmup, args.iters)

    ref = aten_perf["warmup_value"]
    got = op_perf["warmup_value"]
    max_abs, max_rel = _max_diff(got, ref)
    allclose = torch.allclose(got.float(), ref.float(), atol=args.atol, rtol=args.rtol)
    if not allclose:
        print(
            f"[FAIL] tokens={tokens} hc={hc} hidden={hidden} "
            f"max_abs={max_abs:.6g} max_rel={max_rel:.6g}"
        )

    return {
        "tokens": tokens,
        "aten_avg": aten_perf["avg_ms"],
        "op_avg": op_perf["avg_ms"],
        "speedup": aten_perf["avg_ms"] / op_perf["avg_ms"] if op_perf["avg_ms"] > 0 else float("inf"),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "allclose": allclose,
    }


def _print_header(hc, hidden):
    print(f"\nhc={hc} hidden={hidden}")
    print(
        f"{'tokens':>8} | {'aten avg':>10} | {'op avg':>10} | "
        f"{'speedup':>8} | {'max_abs':>10} | {'max_rel':>10} | {'allclose':>8}"
    )
    print("-" * 82)


def _print_row(result):
    print(
        f"{result['tokens']:8d} | "
        f"{result['aten_avg']:10.4f} | "
        f"{result['op_avg']:10.4f} | "
        f"{result['speedup']:8.2f} | "
        f"{result['max_abs']:10.6g} | "
        f"{result['max_rel']:10.6g} | "
        f"{str(result['allclose']):>8}"
    )


def main():
    parser = argparse.ArgumentParser(description="Check and benchmark DeepSeek-V4 HC head.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--hc", type=int, default=4)
    parser.add_argument("--rms-eps", type=float, default=1e-6)
    parser.add_argument("--hc-eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    args = parser.parse_args()

    ok = True
    _print_header(args.hc, args.hidden)
    for tokens in _parse_tokens(args.tokens):
        result = _run_case(tokens, args.hc, args.hidden, args)
        _print_row(result)
        if result["allclose"] is False:
            ok = False

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

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


def _torch_dtype(name):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def _dtype_names(name):
    if name == "all":
        return ["bf16", "fp16"]
    return [name]


def _as_core(tensor):
    return infinicore.from_torch(tensor)


def _make_input(tokens, hidden, dtype, device):
    return torch.randn((tokens, hidden * 2), device=device, dtype=dtype).contiguous()


def _sglang_dispatcher_ref(core_out, core_x, out):
    _infinicore.deepseek_v4_silu_and_mul_dispatcher_(core_out._underlying, core_x._underlying)
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


def _run_case(tokens, hidden, dtype_name, args):
    dtype = _torch_dtype(dtype_name)
    device = torch.device("cuda")
    torch.manual_seed(args.seed + tokens * 17 + hidden)
    x = _make_input(tokens, hidden, dtype, device)
    core_x = _as_core(x)
    ref_out = torch.empty((tokens, hidden), device=device, dtype=dtype)
    core_ref_out = _as_core(ref_out)
    out = torch.empty((tokens, hidden), device=device, dtype=dtype)
    core_out = _as_core(out)

    def run_kernel():
        _infinicore.deepseek_v4_silu_and_mul_(core_out._underlying, core_x._underlying)
        return out

    dispatcher_perf = _bench(
        lambda: _sglang_dispatcher_ref(core_ref_out, core_x, ref_out),
        args.warmup,
        args.iters,
    )
    kernel_perf = _bench(run_kernel, args.warmup, args.iters)

    ref = dispatcher_perf["warmup_value"]
    got = kernel_perf["warmup_value"]
    max_abs, max_rel = _max_diff(got, ref)
    allclose = torch.allclose(got.float(), ref.float(), atol=args.atol, rtol=args.rtol)
    if not allclose:
        print(
            f"[FAIL] dtype={dtype_name} tokens={tokens} hidden={hidden} "
            f"max_abs={max_abs:.6g} max_rel={max_rel:.6g}"
        )

    return {
        "tokens": tokens,
        "dispatcher_avg": dispatcher_perf["avg_ms"],
        "kernel_avg": kernel_perf["avg_ms"],
        "speedup": dispatcher_perf["avg_ms"] / kernel_perf["avg_ms"] if kernel_perf["avg_ms"] > 0 else float("inf"),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "allclose": allclose,
    }


def _print_header(dtype_name, hidden):
    print(f"\ndtype={dtype_name} hidden={hidden} input_hidden={hidden * 2}")
    print(
        f"{'tokens':>8} | {'disp avg':>10} | {'kernel avg':>10} | "
        f"{'speedup':>8} | {'max_abs':>10} | {'max_rel':>10} | {'allclose':>8}"
    )
    print("-" * 86)


def _print_row(result):
    print(
        f"{result['tokens']:8d} | "
        f"{result['dispatcher_avg']:10.4f} | "
        f"{result['kernel_avg']:10.4f} | "
        f"{result['speedup']:8.2f} | "
        f"{result['max_abs']:10.6g} | "
        f"{result['max_rel']:10.6g} | "
        f"{str(result['allclose']):>8}"
    )


def main():
    parser = argparse.ArgumentParser(description="Check and benchmark DeepSeek-V4 silu_and_mul.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "all"], default="all")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    args = parser.parse_args()

    ok = True
    for dtype_name in _dtype_names(args.dtype):
        _print_header(dtype_name, args.hidden)
        for tokens in _parse_tokens(args.tokens):
            result = _run_case(tokens, args.hidden, dtype_name, args)
            _print_row(result)
            if result["allclose"] is False:
                ok = False

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

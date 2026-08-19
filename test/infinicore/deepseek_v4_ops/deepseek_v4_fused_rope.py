import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
from infinicore.lib import _infinicore


DEFAULT_TOKENS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"
DEFAULT_ROPE_DIM = 64


def _parse_tokens(text):
    return [int(item) for item in text.split(",") if item.strip()]


def _torch_int_dtype(name):
    if name == "int32":
        return torch.int32
    if name == "int64":
        return torch.int64
    raise ValueError(f"unsupported dtype: {name}")


def _inverse_values(name):
    if name == "all":
        return [False, True]
    return [name == "true"]


def _as_core(tensor):
    return infinicore.from_torch(tensor)


def _make_freqs(max_pos, dim=DEFAULT_ROPE_DIM, device="cuda"):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    angles = torch.outer(t, inv_freq)
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).flatten(-2).contiguous()


def _make_inputs(tokens, heads, max_pos, pos_dtype, seed, device):
    torch.manual_seed(seed + tokens * 17 + heads)
    query = torch.randn((tokens, heads, DEFAULT_ROPE_DIM), device=device, dtype=torch.bfloat16).contiguous()
    key = torch.randn((tokens, 1, DEFAULT_ROPE_DIM), device=device, dtype=torch.bfloat16).contiguous()
    freqs = _make_freqs(max_pos=max_pos, device=device)
    positions = ((torch.arange(tokens, device=device, dtype=pos_dtype) * 3) % max_pos).contiguous()
    return query, key, freqs, positions


def _aten_ref(core_query, core_key, core_freqs, core_positions, inverse, query, key):
    _infinicore.deepseek_v4_fused_rope_aten_(
        core_query._underlying,
        core_key._underlying,
        core_freqs._underlying,
        core_positions._underlying,
        inverse,
    )
    return query, key


def _tuple_max_diff(got, ref):
    max_abs = 0.0
    max_rel = 0.0
    for lhs, rhs in zip(got, ref):
        lhs_f = lhs.float()
        rhs_f = rhs.float()
        abs_diff = (lhs_f - rhs_f).abs()
        if abs_diff.numel() == 0:
            continue
        max_abs = max(max_abs, abs_diff.max().item())
        max_rel = max(max_rel, (abs_diff / rhs_f.abs().clamp_min(1e-6)).max().item())
    return max_abs, max_rel


def _tuple_allclose(got, ref, atol, rtol):
    return all(torch.allclose(lhs.float(), rhs.float(), atol=atol, rtol=rtol) for lhs, rhs in zip(got, ref))


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


def _run_case(tokens, heads, inverse, pos_dtype, args):
    device = torch.device("cuda")
    query, key, freqs, positions = _make_inputs(tokens, heads, args.max_pos, pos_dtype, args.seed, device)
    ref_query = query.clone()
    ref_key = key.clone()
    out_query = query.clone()
    out_key = key.clone()

    core_freqs = _as_core(freqs)
    core_positions = _as_core(positions)
    core_ref_query = _as_core(ref_query)
    core_ref_key = _as_core(ref_key)
    core_out_query = _as_core(out_query)
    core_out_key = _as_core(out_key)

    def run_aten():
        return _aten_ref(core_ref_query, core_ref_key, core_freqs, core_positions, inverse, ref_query, ref_key)

    def run_kernel():
        _infinicore.deepseek_v4_fused_rope_(
            core_out_query._underlying,
            core_out_key._underlying,
            core_freqs._underlying,
            core_positions._underlying,
            inverse,
        )
        return out_query, out_key

    aten_perf = _bench(run_aten, args.warmup, args.iters)
    kernel_perf = _bench(run_kernel, args.warmup, args.iters)

    check_ref_query = query.clone()
    check_ref_key = key.clone()
    check_out_query = query.clone()
    check_out_key = key.clone()
    _infinicore.deepseek_v4_fused_rope_aten_(
        _as_core(check_ref_query)._underlying,
        _as_core(check_ref_key)._underlying,
        core_freqs._underlying,
        core_positions._underlying,
        inverse,
    )
    _infinicore.deepseek_v4_fused_rope_(
        _as_core(check_out_query)._underlying,
        _as_core(check_out_key)._underlying,
        core_freqs._underlying,
        core_positions._underlying,
        inverse,
    )
    torch.cuda.synchronize()

    ref = (check_ref_query, check_ref_key)
    got = (check_out_query, check_out_key)
    max_abs, max_rel = _tuple_max_diff(got, ref)
    allclose = _tuple_allclose(got, ref, args.atol, args.rtol)
    if not allclose:
        print(
            f"[FAIL] tokens={tokens} heads={heads} inverse={inverse} "
            f"max_abs={max_abs:.6g} max_rel={max_rel:.6g}"
        )

    return {
        "tokens": tokens,
        "aten_avg": aten_perf["avg_ms"],
        "kernel_avg": kernel_perf["avg_ms"],
        "speedup": aten_perf["avg_ms"] / kernel_perf["avg_ms"] if kernel_perf["avg_ms"] > 0 else float("inf"),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "allclose": allclose,
    }


def _print_header(heads, inverse, pos_dtype_name):
    print(f"\nheads={heads} rope_dim={DEFAULT_ROPE_DIM} inverse={inverse} pos_dtype={pos_dtype_name}")
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
    parser = argparse.ArgumentParser(description="Check and benchmark DeepSeek-V4 fused_rope.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--max-pos", type=int, default=1048576)
    parser.add_argument("--pos-dtype", choices=["int32", "int64"], default="int64")
    parser.add_argument("--inverse", choices=["false", "true", "all"], default="all")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    args = parser.parse_args()

    ok = True
    pos_dtype = _torch_int_dtype(args.pos_dtype)
    for inverse in _inverse_values(args.inverse):
        _print_header(args.heads, inverse, args.pos_dtype)
        for tokens in _parse_tokens(args.tokens):
            result = _run_case(tokens, args.heads, inverse, pos_dtype, args)
            _print_row(result)
            if result["allclose"] is False:
                ok = False

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

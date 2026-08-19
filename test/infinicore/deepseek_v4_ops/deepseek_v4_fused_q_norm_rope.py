import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
from infinicore.lib import _infinicore


DEFAULT_TOKENS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"
DEFAULT_NUM_ATTENTION_HEADS = 64
DEFAULT_TP = 8
DEFAULT_HEAD_DIM = 512
DEFAULT_ROPE_DIM = 64
DEFAULT_MAX_POS = 1048576


def _parse_tokens(text):
    return [int(item) for item in text.split(",") if item.strip()]


def _torch_int_dtype(name):
    if name == "int32":
        return torch.int32
    if name == "int64":
        return torch.int64
    raise ValueError(f"unsupported dtype: {name}")


def _resolve_heads(args):
    if args.heads is not None:
        return args.heads
    if args.num_attention_heads % args.tp != 0:
        raise ValueError(f"num_attention_heads={args.num_attention_heads} must be divisible by tp={args.tp}")
    return args.num_attention_heads // args.tp


def _as_core(tensor):
    return infinicore.from_torch(tensor).as_strided(list(tensor.shape), list(tensor.stride()))


def _make_freqs(max_pos, dim=DEFAULT_ROPE_DIM, device="cuda"):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    angles = torch.outer(t, inv_freq)
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).flatten(-2).contiguous()


def _make_inputs(tokens, heads, freqs, pos_dtype, strided_batch, seed, device):
    torch.manual_seed(seed + tokens * 17 + heads)
    if strided_batch:
        q_base = torch.randn((tokens, heads + 1, DEFAULT_HEAD_DIM), device=device, dtype=torch.bfloat16)
        ref_base = torch.empty((tokens, heads + 1, DEFAULT_HEAD_DIM), device=device, dtype=torch.bfloat16)
        out_base = torch.empty_like(ref_base)
        q = q_base[:, :heads, :]
        ref_out = ref_base[:, :heads, :]
        out = out_base[:, :heads, :]
    else:
        q = torch.randn((tokens, heads, DEFAULT_HEAD_DIM), device=device, dtype=torch.bfloat16)
        ref_out = torch.empty_like(q)
        out = torch.empty_like(q)
    positions = ((torch.arange(tokens, device=device, dtype=pos_dtype) * 3) % freqs.shape[0]).contiguous()
    return q, ref_out, out, positions


def _aten_ref(core_out, core_q, eps, core_freqs, core_positions, out):
    _infinicore.deepseek_v4_fused_q_norm_rope_aten_(
        core_out._underlying,
        core_q._underlying,
        eps,
        core_freqs._underlying,
        core_positions._underlying,
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


def _run_case(tokens, heads, freqs, pos_dtype, args):
    device = torch.device("cuda")
    q, ref_out, out, positions = _make_inputs(tokens, heads, freqs, pos_dtype, args.strided_batch, args.seed, device)
    core_q = _as_core(q)
    core_ref_out = _as_core(ref_out)
    core_out = _as_core(out)
    core_freqs = _as_core(freqs)
    core_positions = _as_core(positions)

    def run_aten():
        return _aten_ref(core_ref_out, core_q, args.eps, core_freqs, core_positions, ref_out)

    def run_kernel():
        _infinicore.deepseek_v4_fused_q_norm_rope_(
            core_out._underlying,
            core_q._underlying,
            args.eps,
            core_freqs._underlying,
            core_positions._underlying,
        )
        return out

    aten_perf = _bench(run_aten, args.warmup, args.iters)
    kernel_perf = _bench(run_kernel, args.warmup, args.iters)

    ref = aten_perf["warmup_value"]
    got = kernel_perf["warmup_value"]
    max_abs, max_rel = _max_diff(got, ref)
    allclose = torch.allclose(got.float(), ref.float(), atol=args.atol, rtol=args.rtol)
    if not allclose:
        print(
            f"[FAIL] tokens={tokens} heads={heads} stride0={q.stride(0)} "
            f"max_abs={max_abs:.6g} max_rel={max_rel:.6g}"
        )

    return {
        "tokens": tokens,
        "rows": tokens * heads,
        "stride0": q.stride(0),
        "aten_avg": aten_perf["avg_ms"],
        "kernel_avg": kernel_perf["avg_ms"],
        "speedup": aten_perf["avg_ms"] / kernel_perf["avg_ms"] if kernel_perf["avg_ms"] > 0 else float("inf"),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "allclose": allclose,
    }


def _print_header(heads, pos_dtype_name, strided_batch):
    print(
        f"\nheads={heads} head_dim={DEFAULT_HEAD_DIM} rope_dim={DEFAULT_ROPE_DIM} "
        f"pos_dtype={pos_dtype_name} strided_batch={strided_batch}"
    )
    print(
        f"{'tokens':>8} | {'rows':>8} | {'stride0':>8} | {'aten avg':>10} | "
        f"{'kernel avg':>10} | {'speedup':>8} | {'max_abs':>10} | {'max_rel':>10} | {'allclose':>8}"
    )
    print("-" * 110)


def _print_row(result):
    print(
        f"{result['tokens']:8d} | "
        f"{result['rows']:8d} | "
        f"{result['stride0']:8d} | "
        f"{result['aten_avg']:10.4f} | "
        f"{result['kernel_avg']:10.4f} | "
        f"{result['speedup']:8.2f} | "
        f"{result['max_abs']:10.6g} | "
        f"{result['max_rel']:10.6g} | "
        f"{str(result['allclose']):>8}"
    )


def main():
    parser = argparse.ArgumentParser(description="Check and benchmark DeepSeek-V4 fused_q_norm_rope.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--num-attention-heads", type=int, default=DEFAULT_NUM_ATTENTION_HEADS)
    parser.add_argument("--tp", type=int, default=DEFAULT_TP)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--max-pos", type=int, default=DEFAULT_MAX_POS)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--pos-dtype", choices=["int32", "int64"], default="int64")
    parser.add_argument("--strided-batch", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    args = parser.parse_args()

    heads = _resolve_heads(args)
    pos_dtype = _torch_int_dtype(args.pos_dtype)
    freqs = _make_freqs(args.max_pos, device="cuda")

    ok = True
    _print_header(heads, args.pos_dtype, args.strided_batch)
    for tokens in _parse_tokens(args.tokens):
        result = _run_case(tokens, heads, freqs, pos_dtype, args)
        _print_row(result)
        if result["allclose"] is False:
            ok = False

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

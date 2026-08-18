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
    mix_hc = (2 + hc) * hc
    k = hc * hidden

    residual = torch.randn((tokens, hc, hidden), device=device, dtype=dtype).contiguous()
    fn = (torch.randn((mix_hc, k), device=device, dtype=torch.float32) * 0.02).contiguous()
    hc_scale = (torch.randn((3,), device=device, dtype=torch.float32) * 0.1).contiguous()
    hc_base = (torch.randn((mix_hc,), device=device, dtype=torch.float32) * 0.1).contiguous()
    return residual, fn, hc_scale, hc_base


def _make_outputs(tokens, hc, hidden):
    device = torch.device("cuda")
    y = torch.empty((tokens, hidden), device=device, dtype=torch.bfloat16)
    post = torch.empty((tokens, hc), device=device, dtype=torch.float32)
    comb = torch.empty((tokens, hc, hc), device=device, dtype=torch.float32)
    return y, post, comb


def _aten_ref(core_y, core_post, core_comb, core_residual, core_fn, core_hc_scale, core_hc_base, args, outputs):
    _infinicore.deepseek_v4_mhc_pre_aten_(
        core_y._underlying,
        core_post._underlying,
        core_comb._underlying,
        core_residual._underlying,
        core_fn._underlying,
        core_hc_scale._underlying,
        core_hc_base._underlying,
        args.rms_eps,
        args.hc_pre_eps,
        args.hc_sinkhorn_eps,
        args.sinkhorn_repeat,
    )
    return outputs


def _tuple_max_diff(lhs_tuple, rhs_tuple):
    max_abs = 0.0
    max_rel = 0.0
    for lhs, rhs in zip(lhs_tuple, rhs_tuple):
        lhs_f = lhs.float()
        rhs_f = rhs.float()
        abs_diff = (lhs_f - rhs_f).abs()
        item_abs = abs_diff.max().item() if abs_diff.numel() > 0 else 0.0
        denom = rhs_f.abs().clamp_min(1e-6)
        item_rel = (abs_diff / denom).max().item() if abs_diff.numel() > 0 else 0.0
        max_abs = max(max_abs, item_abs)
        max_rel = max(max_rel, item_rel)
    return max_abs, max_rel


def _tuple_allclose(lhs_tuple, rhs_tuple, atol, rtol):
    return all(torch.allclose(lhs.float(), rhs.float(), atol=atol, rtol=rtol) for lhs, rhs in zip(lhs_tuple, rhs_tuple))


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
    residual, fn, hc_scale, hc_base = _make_inputs(tokens, hc, hidden, args.seed)
    ref_outputs = _make_outputs(tokens, hc, hidden)
    op_outputs = _make_outputs(tokens, hc, hidden)

    core_residual = _as_core(residual)
    core_fn = _as_core(fn)
    core_hc_scale = _as_core(hc_scale)
    core_hc_base = _as_core(hc_base)
    core_ref = tuple(_as_core(tensor) for tensor in ref_outputs)
    core_op = tuple(_as_core(tensor) for tensor in op_outputs)

    def run_aten():
        return _aten_ref(
            core_ref[0],
            core_ref[1],
            core_ref[2],
            core_residual,
            core_fn,
            core_hc_scale,
            core_hc_base,
            args,
            ref_outputs,
        )

    def run_op():
        _infinicore.deepseek_v4_mhc_pre_(
            core_op[0]._underlying,
            core_op[1]._underlying,
            core_op[2]._underlying,
            core_residual._underlying,
            core_fn._underlying,
            core_hc_scale._underlying,
            core_hc_base._underlying,
            args.rms_eps,
            args.hc_pre_eps,
            args.hc_sinkhorn_eps,
            args.sinkhorn_repeat,
        )
        return op_outputs

    aten_perf = _bench(run_aten, args.warmup, args.iters)
    op_perf = _bench(run_op, args.warmup, args.iters)

    ref = aten_perf["warmup_value"]
    got = op_perf["warmup_value"]
    max_abs, max_rel = _tuple_max_diff(got, ref)
    allclose = _tuple_allclose(got, ref, args.atol, args.rtol)
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
    parser = argparse.ArgumentParser(description="Check and benchmark DeepSeek-V4 MHC pre.")
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--hc", type=int, default=4)
    parser.add_argument("--rms-eps", type=float, default=1e-6)
    parser.add_argument("--hc-pre-eps", type=float, default=1e-6)
    parser.add_argument("--hc-sinkhorn-eps", type=float, default=1e-6)
    parser.add_argument("--sinkhorn-repeat", "--sinkhorn-iters", dest="sinkhorn_repeat", type=int, default=5)
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

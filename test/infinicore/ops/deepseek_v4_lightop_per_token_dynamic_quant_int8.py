#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch


DEFAULT_INFINICORE_REPO = "/workspace_codex/InfiniCore"
DEFAULT_TOKENS = "1,4,16,17,32,64,128"


def _prepend(path: str | None) -> None:
    if path and Path(path).exists() and path not in sys.path:
        sys.path.insert(0, path)


def _add_paths(args) -> None:
    for root in (args.infinicore_repo, os.environ.get("INFINICORE_REPO")):
        if root:
            _prepend(str(Path(root) / "python"))
            _prepend(root)


def _as_core(tensor: torch.Tensor):
    import infinicore

    return infinicore.from_torch(tensor)._underlying


def _sync() -> None:
    import infinicore

    infinicore.sync_stream()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _torch_reference(x: torch.Tensor, smooth_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xs = x.float() * smooth_scale.float().view(1, -1)
    absmax = xs.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-10)
    scale = absmax / 127.0
    q = torch.round(xs / scale).clamp(-128, 127).to(torch.int8)
    return q.contiguous(), scale.float().contiguous()


def _call_infinicore(q: torch.Tensor, x: torch.Tensor, scale: torch.Tensor, smooth_scale: torch.Tensor) -> None:
    from infinicore.lib import _infinicore

    _infinicore.deepseek_v4_lightop_per_token_dynamic_quant_int8_(
        _as_core(q),
        _as_core(x),
        _as_core(scale),
        _as_core(smooth_scale),
    )


def _call_infinicore_core(q, x, scale, smooth_scale) -> None:
    from infinicore.lib import _infinicore

    _infinicore.deepseek_v4_lightop_per_token_dynamic_quant_int8_(q, x, scale, smooth_scale)


def _call_lightop(q: torch.Tensor, x: torch.Tensor, scale: torch.Tensor, smooth_scale: torch.Tensor) -> None:
    import lightop.op as lightop_op

    lightop_op.per_token_dynamic_quant_int8(q, x, scale, smooth_scale)


def _assert_close(
    name: str,
    got_q: torch.Tensor,
    got_scale: torch.Tensor,
    ref_q: torch.Tensor,
    ref_scale: torch.Tensor,
    q_atol: int,
) -> None:
    torch.testing.assert_close(got_q.float(), ref_q.float(), atol=q_atol, rtol=0)
    torch.testing.assert_close(got_scale.float(), ref_scale.float(), atol=1.0e-6, rtol=1.0e-6)
    q_max_diff = (got_q.float() - ref_q.float()).abs().max().item()
    scale_max_diff = (got_scale.float() - ref_scale.float()).abs().max().item()
    print(f"{name:<24s} q_max_diff={q_max_diff:.0f} scale_max_diff={scale_max_diff:.8f}")


def _run_case(tokens: int, hidden: int, dtype: torch.dtype, args) -> None:
    torch.manual_seed(args.seed + tokens)
    x = torch.randn((tokens, hidden), device="cuda", dtype=dtype).contiguous()
    smooth_scale = (torch.rand((hidden,), device="cuda", dtype=torch.float32) * 1.5 + 0.5).contiguous()

    q = torch.empty((tokens, hidden), device="cuda", dtype=torch.int8)
    scale = torch.empty((tokens, 1), device="cuda", dtype=torch.float32)
    _call_infinicore(q, x, scale, smooth_scale)
    _sync()

    lightop_q = torch.empty_like(q)
    lightop_scale = torch.empty_like(scale)
    _call_lightop(lightop_q, x, lightop_scale, smooth_scale)
    torch.cuda.synchronize()

    torch_q, torch_scale = _torch_reference(x, smooth_scale)
    _assert_close(f"tokens={tokens} lightop", q, scale, lightop_q, lightop_scale, q_atol=0)
    _assert_close(f"tokens={tokens} torch", q, scale, torch_q, torch_scale, q_atol=1)

    if not args.skip_graph:
        import infinicore

        graph_q = torch.empty_like(q)
        graph_scale = torch.empty_like(scale)
        infinicore.start_graph_recording()
        _call_infinicore(graph_q, x, graph_scale, smooth_scale)
        graph = infinicore.stop_graph_recording()
        graph_q.zero_()
        graph_scale.zero_()
        _sync()
        graph.run()
        _sync()
        _assert_close(f"tokens={tokens} graph", graph_q, graph_scale, lightop_q, lightop_scale, q_atol=0)


def _bench_one(label: str, fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - start) * 1000.0)
    times.sort()
    median = times[len(times) // 2]
    avg = sum(times) / len(times)
    print(f"{label:<24s} avg={avg:.6f} ms median={median:.6f} ms")
    return median


def _run_perf(tokens: int, hidden: int, dtype: torch.dtype, args) -> None:
    torch.manual_seed(args.seed + tokens + 1000)
    x = torch.randn((tokens, hidden), device="cuda", dtype=dtype).contiguous()
    smooth_scale = (torch.rand((hidden,), device="cuda", dtype=torch.float32) * 1.5 + 0.5).contiguous()
    q = torch.empty((tokens, hidden), device="cuda", dtype=torch.int8)
    scale = torch.empty((tokens, 1), device="cuda", dtype=torch.float32)
    lightop_q = torch.empty_like(q)
    lightop_scale = torch.empty_like(scale)
    q_core = _as_core(q)
    x_core = _as_core(x)
    scale_core = _as_core(scale)
    smooth_scale_core = _as_core(smooth_scale)

    print(f"\nperf tokens={tokens} hidden={hidden}")
    t_core = _bench_one(
        "infinicore",
        lambda: _call_infinicore_core(q_core, x_core, scale_core, smooth_scale_core),
        args.iters,
        args.warmup,
    )
    t_lightop = _bench_one(
        "lightop.python",
        lambda: _call_lightop(lightop_q, x, lightop_scale, smooth_scale),
        args.iters,
        args.warmup,
    )
    print(f"{'speedup':<24s} lightop_python/infinicore={t_lightop / t_core:.3f}x")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--infinicore-repo", default=DEFAULT_INFINICORE_REPO)
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--perf", action="store_true")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--skip-graph", action="store_true")
    args = parser.parse_args()
    _add_paths(args)

    import infinicore  # noqa: F401

    if not torch.cuda.is_available():
        raise RuntimeError("deepseek_v4_lightop_per_token_dynamic_quant_int8 test requires an accelerator device")
    if args.hidden % 64 != 0:
        raise ValueError("hidden must be divisible by 64")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    tokens_list = [int(x) for x in args.tokens.split(",") if x.strip()]
    for tokens in tokens_list:
        _run_case(tokens, args.hidden, dtype, args)
    if args.perf:
        for tokens in tokens_list:
            _run_perf(tokens, args.hidden, dtype, args)
    print("deepseek_v4_lightop_per_token_dynamic_quant_int8 ok")


if __name__ == "__main__":
    main()

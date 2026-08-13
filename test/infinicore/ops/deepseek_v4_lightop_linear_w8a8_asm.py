#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch


DEFAULT_INFINICORE_REPO = "/workspace_codex/InfiniCore"
DEFAULT_TOKENS = "17,32,64,128"
BLOCK = 128


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


def _reference(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    input_block_scale: torch.Tensor,
    weight_block_scale: torch.Tensor,
) -> torch.Tensor:
    m, k = q_input.shape
    n = weight.shape[0]
    k_blocks = k // BLOCK
    n_blocks = n // BLOCK
    out = torch.zeros((m, n), device=q_input.device, dtype=torch.float32)
    for kb in range(k_blocks):
        k0 = kb * BLOCK
        partial = q_input[:, k0 : k0 + BLOCK].float() @ weight[:, k0 : k0 + BLOCK].float().t()
        row_scale = input_block_scale[kb].view(m, 1)
        col_scale = weight_block_scale[:, kb].repeat_interleave(BLOCK)[:n].view(1, n)
        out += partial * row_scale * col_scale
    assert weight_block_scale.shape == (n_blocks, k_blocks)
    return out.to(torch.bfloat16).contiguous()


def _quant_int8_per_token(x: torch.Tensor, smooth_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xs = x.float() * smooth_scale.float().view(1, -1)
    absmax = xs.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-10)
    scale = absmax / 127.0
    q = torch.round(xs / scale).clamp(-128, 127).to(torch.int8)
    return q.contiguous(), scale.float().contiguous()


def _reference_per_channel(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    smooth_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_input, input_scale = _quant_int8_per_token(x, smooth_scale)
    out = torch.matmul(q_input.float(), weight.float().t())
    out = out * input_scale * weight_scale.view(1, -1)
    return out.to(torch.bfloat16).contiguous(), q_input, input_scale


def _call_infinicore(
    output: torch.Tensor,
    q_input: torch.Tensor,
    weight: torch.Tensor,
    input_block_scale: torch.Tensor,
    weight_block_scale: torch.Tensor,
) -> None:
    from infinicore.lib import _infinicore

    _infinicore.deepseek_v4_lightop_linear_w8a8_asm_(
        _as_core(output),
        _as_core(q_input),
        _as_core(weight),
        _as_core(input_block_scale),
        _as_core(weight_block_scale),
    )


def _call_infinicore_per_channel(
    output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    q_input: torch.Tensor,
    input_scale: torch.Tensor,
    input_block_scale: torch.Tensor,
    weight_block_scale: torch.Tensor,
    smooth_scale: torch.Tensor,
) -> None:
    from infinicore.lib import _infinicore

    _infinicore.deepseek_v4_lightop_linear_w8a8_asm_per_channel_(
        _as_core(output),
        _as_core(x),
        _as_core(weight),
        _as_core(weight_scale),
        _as_core(q_input),
        _as_core(input_scale),
        _as_core(input_block_scale),
        _as_core(weight_block_scale),
        _as_core(smooth_scale),
    )


def _make_case(tokens: int, n: int, k: int, seed: int):
    torch.manual_seed(seed + tokens)
    q_input = torch.randint(-64, 64, (tokens, k), device="cuda", dtype=torch.int8).contiguous()
    weight = torch.randint(-64, 64, (n, k), device="cuda", dtype=torch.int8).contiguous()
    input_block_scale = (torch.rand((k // BLOCK, tokens), device="cuda", dtype=torch.float32) * 0.01 + 0.001).contiguous()
    weight_block_scale = (torch.rand((n // BLOCK, k // BLOCK), device="cuda", dtype=torch.float32) * 0.01 + 0.001).contiguous()
    return q_input, weight, input_block_scale, weight_block_scale


def _make_per_channel_case(tokens: int, n: int, k: int, seed: int):
    torch.manual_seed(seed + tokens + 2000)
    x = torch.randn((tokens, k), device="cuda", dtype=torch.bfloat16).contiguous()
    weight = torch.randint(-64, 64, (n, k), device="cuda", dtype=torch.int8).contiguous()
    weight_scale = (torch.rand((n, 1), device="cuda", dtype=torch.float32) * 0.01 + 0.001).contiguous()
    smooth_scale = torch.ones((k,), device="cuda", dtype=torch.float32).contiguous()
    q_input = torch.empty((tokens, k), device="cuda", dtype=torch.int8)
    input_scale = torch.empty((tokens, 1), device="cuda", dtype=torch.float32)
    input_block_scale = torch.empty((k // BLOCK, tokens), device="cuda", dtype=torch.float32)
    weight_block_scale = torch.empty((n // BLOCK, k // BLOCK), device="cuda", dtype=torch.float32)
    return x, weight, weight_scale, q_input, input_scale, input_block_scale, weight_block_scale, smooth_scale


def _assert_close(name: str, got: torch.Tensor, expected: torch.Tensor, args) -> None:
    try:
        torch.testing.assert_close(got.float(), expected.float(), atol=args.atol, rtol=args.rtol)
    except AssertionError as exc:
        diff = (got.float() - expected.float()).abs()
        raise AssertionError(f"{name} mismatch max_abs={diff.max().item()} mean_abs={diff.mean().item()}") from exc
    diff = (got.float() - expected.float()).abs()
    print(f"{name:<24s} max_diff={diff.max().item():.6f} mean_diff={diff.mean().item():.6f}")


def _assert_aux(name: str, got_q: torch.Tensor, expected_q: torch.Tensor, got_scale: torch.Tensor, expected_scale: torch.Tensor) -> None:
    torch.testing.assert_close(got_q.float(), expected_q.float(), atol=1, rtol=0)
    torch.testing.assert_close(got_scale.float(), expected_scale.float(), atol=1.0e-5, rtol=1.0e-5)
    print(f"{name:<24s} q_max_diff={(got_q.float() - expected_q.float()).abs().max().item():.0f}")


def _run_case(tokens: int, n: int, k: int, args) -> None:
    q_input, weight, input_block_scale, weight_block_scale = _make_case(tokens, n, k, args.seed)
    expected = _reference(q_input, weight, input_block_scale, weight_block_scale)
    got = torch.empty_like(expected)

    _call_infinicore(got, q_input, weight, input_block_scale, weight_block_scale)
    _sync()
    _assert_close(f"tokens={tokens} eager", got, expected, args)

    if not args.skip_graph:
        import infinicore

        graph_got = torch.empty_like(expected)
        infinicore.start_graph_recording()
        _call_infinicore(graph_got, q_input, weight, input_block_scale, weight_block_scale)
        graph = infinicore.stop_graph_recording()
        graph_got.zero_()
        _sync()
        graph.run()
        _sync()
        _assert_close(f"tokens={tokens} graph", graph_got, expected, args)


def _run_per_channel_case(tokens: int, n: int, k: int, args) -> None:
    case = _make_per_channel_case(tokens, n, k, args.seed)
    x, weight, weight_scale, q_input, input_scale, input_block_scale, weight_block_scale, smooth_scale = case
    expected, expected_q, expected_scale = _reference_per_channel(x, weight, weight_scale, smooth_scale)
    got = torch.empty_like(expected)

    _call_infinicore_per_channel(
        got, x, weight, weight_scale, q_input, input_scale, input_block_scale, weight_block_scale, smooth_scale)
    _sync()
    _assert_close(f"pc tokens={tokens} eager", got, expected, args)
    _assert_aux(f"pc tokens={tokens} aux", q_input, expected_q, input_scale, expected_scale)

    if not args.skip_graph:
        import infinicore

        graph_got = torch.empty_like(expected)
        graph_q = torch.empty_like(q_input)
        graph_scale = torch.empty_like(input_scale)
        graph_input_block = torch.empty_like(input_block_scale)
        graph_weight_block = torch.empty_like(weight_block_scale)
        infinicore.start_graph_recording()
        _call_infinicore_per_channel(
            graph_got,
            x,
            weight,
            weight_scale,
            graph_q,
            graph_scale,
            graph_input_block,
            graph_weight_block,
            smooth_scale,
        )
        graph = infinicore.stop_graph_recording()
        graph_got.zero_()
        graph_q.zero_()
        graph_scale.zero_()
        _sync()
        graph.run()
        _sync()
        _assert_close(f"pc tokens={tokens} graph", graph_got, expected, args)
        _assert_aux(f"pc tokens={tokens} gaux", graph_q, expected_q, graph_scale, expected_scale)


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


def _run_perf(tokens: int, n: int, k: int, args) -> None:
    q_input, weight, input_block_scale, weight_block_scale = _make_case(tokens, n, k, args.seed + 1000)
    output = torch.empty((tokens, n), device="cuda", dtype=torch.bfloat16)
    print(f"\nperf tokens={tokens} N={n} K={k}")
    t_op = _bench_one(
        "infinicore.lightop_asm",
        lambda: _call_infinicore(output, q_input, weight, input_block_scale, weight_block_scale),
        args.iters,
        args.warmup,
    )
    t_ref = _bench_one(
        "torch.blockwise_ref",
        lambda: _reference(q_input, weight, input_block_scale, weight_block_scale),
        args.iters,
        args.warmup,
    )
    print(f"{'speedup':<24s} torch_ref/op={t_ref / t_op:.3f}x")


def _run_per_channel_perf(tokens: int, n: int, k: int, args) -> None:
    case = _make_per_channel_case(tokens, n, k, args.seed + 3000)
    x, weight, weight_scale, q_input, input_scale, input_block_scale, weight_block_scale, smooth_scale = case
    output = torch.empty((tokens, n), device="cuda", dtype=torch.bfloat16)
    print(f"\nperf per-channel tokens={tokens} N={n} K={k}")
    _bench_one(
        "infinicore.pc_asm",
        lambda: _call_infinicore_per_channel(
            output, x, weight, weight_scale, q_input, input_scale, input_block_scale, weight_block_scale, smooth_scale),
        args.iters,
        args.warmup,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--infinicore-repo", default=DEFAULT_INFINICORE_REPO)
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--n", type=int, default=1536)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--atol", type=float, default=0.25)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--perf", action="store_true")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--skip-graph", action="store_true")
    args = parser.parse_args()
    _add_paths(args)

    import infinicore  # noqa: F401

    if not torch.cuda.is_available():
        raise RuntimeError("deepseek_v4_lightop_linear_w8a8_asm test requires an accelerator device")
    if args.n % BLOCK != 0 or args.k % BLOCK != 0:
        raise ValueError("n and k must be divisible by 128")

    tokens_list = [int(x) for x in args.tokens.split(",") if x.strip()]
    for tokens in tokens_list:
        _run_case(tokens, args.n, args.k, args)
        _run_per_channel_case(tokens, args.n, args.k, args)
    if args.perf:
        for tokens in tokens_list:
            _run_perf(tokens, args.n, args.k, args)
            _run_per_channel_perf(tokens, args.n, args.k, args)
    print("deepseek_v4_lightop_linear_w8a8_asm ok")


if __name__ == "__main__":
    main()

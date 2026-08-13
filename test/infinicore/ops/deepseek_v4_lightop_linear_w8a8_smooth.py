#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


DEFAULT_INFINICORE_REPO = "/workspace_codex/InfiniCore"


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


def _quant_int8_per_token(x: torch.Tensor, smooth_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xf = x.float() * smooth_scale.float().view(1, -1)
    absmax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-10)
    scale = absmax / 127.0
    q = torch.round(xf * (127.0 / absmax)).clamp(-128, 127).to(torch.int8)
    return q.contiguous(), scale.float().contiguous()


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    smooth_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_x, x_scale = _quant_int8_per_token(x, smooth_scale)
    out = torch.matmul(q_x.float(), weight.float().transpose(0, 1))
    out = out * x_scale * weight_scale.view(1, -1)
    if bias is not None:
        out = out + bias.float().view(1, -1)
    return out.to(torch.bfloat16).contiguous(), q_x, x_scale


def _call_op(got, x, weight, weight_scale, smooth_scale, bias, q_input, input_scale) -> None:
    from infinicore.lib import _infinicore

    _infinicore.deepseek_v4_lightop_linear_w8a8_smooth_(
        _as_core(got),
        _as_core(x),
        _as_core(weight),
        _as_core(weight_scale),
        None if bias is None else _as_core(bias),
        _as_core(q_input),
        _as_core(input_scale),
        _as_core(smooth_scale),
        False,
        255,
    )


def _assert_case(name: str, got, expected, q_input, expected_q, input_scale, expected_scale, args) -> None:
    torch.testing.assert_close(q_input.float(), expected_q.float(), atol=1, rtol=0)
    torch.testing.assert_close(input_scale.float(), expected_scale.float(), atol=1e-5, rtol=1e-5)
    try:
        torch.testing.assert_close(got.float(), expected.float(), atol=args.atol, rtol=args.rtol)
    except AssertionError as exc:
        max_diff = (got.float() - expected.float()).abs().max().item()
        raise AssertionError(f"{name} mismatch, max_abs={max_diff}") from exc
    max_diff = (got.float() - expected.float()).abs().max().item()
    print(f"{name:<28s} max_diff={max_diff:.6f}")


def _run_case(tokens: int, hidden: int, out_features: int, with_bias: bool, use_random_smooth: bool, args) -> None:
    torch.manual_seed(args.seed + tokens + (13 if with_bias else 0))
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16).contiguous()
    weight = torch.randint(-64, 64, (out_features, hidden), device="cuda", dtype=torch.int8).contiguous()
    weight_scale = (torch.rand((out_features, 1), device="cuda", dtype=torch.float32) * 0.01 + 0.001).contiguous()
    if use_random_smooth:
        smooth_scale = (torch.rand((hidden,), device="cuda", dtype=torch.float32) * 1.5 + 0.5).contiguous()
    else:
        smooth_scale = torch.ones((hidden,), device="cuda", dtype=torch.float32).contiguous()
    bias = torch.randn((out_features,), device="cuda", dtype=torch.bfloat16).contiguous() if with_bias else None
    q_input = torch.empty_like(x, dtype=torch.int8)
    input_scale = torch.empty((tokens, 1), device="cuda", dtype=torch.float32)

    expected, expected_q, expected_scale = _reference(x, weight, weight_scale, smooth_scale, bias)
    got = torch.empty_like(expected)
    _call_op(got, x, weight, weight_scale, smooth_scale, bias, q_input, input_scale)
    _sync()
    _assert_case(
        f"tokens={tokens} bias={with_bias} smooth={use_random_smooth} eager",
        got,
        expected,
        q_input,
        expected_q,
        input_scale,
        expected_scale,
        args,
    )

    if not args.skip_graph:
        import infinicore

        graph_got = torch.empty_like(expected)
        graph_q = torch.empty_like(q_input)
        graph_scale = torch.empty_like(input_scale)
        infinicore.start_graph_recording()
        _call_op(graph_got, x, weight, weight_scale, smooth_scale, bias, graph_q, graph_scale)
        graph = infinicore.stop_graph_recording()
        graph_got.zero_()
        graph_q.zero_()
        graph_scale.zero_()
        _sync()
        graph.run()
        _sync()
        _assert_case(
            f"tokens={tokens} bias={with_bias} smooth={use_random_smooth} graph",
            graph_got,
            expected,
            graph_q,
            expected_q,
            graph_scale,
            expected_scale,
            args,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--infinicore-repo", default=DEFAULT_INFINICORE_REPO)
    parser.add_argument("--tokens", default="1,4,16")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=576)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--atol", type=float, default=0.75)
    parser.add_argument("--rtol", type=float, default=0.08)
    parser.add_argument("--skip-graph", action="store_true")
    args = parser.parse_args()
    _add_paths(args)

    import infinicore  # noqa: F401

    if not torch.cuda.is_available():
        raise RuntimeError("deepseek_v4_lightop_linear_w8a8_smooth test requires an accelerator device")
    if args.hidden % 64 != 0 or args.out_features % 16 != 0:
        raise ValueError("hidden must be divisible by 64 and out-features by 16")

    for tokens in [int(x) for x in args.tokens.split(",") if x.strip()]:
        _run_case(tokens, args.hidden, args.out_features, False, False, args)
        _run_case(tokens, args.hidden, args.out_features, True, False, args)
        _run_case(tokens, args.hidden, args.out_features, False, True, args)
    print("deepseek_v4_lightop_linear_w8a8_smooth ok")


if __name__ == "__main__":
    main()

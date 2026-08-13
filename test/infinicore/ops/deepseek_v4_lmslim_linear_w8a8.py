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


def _quant_reference(x: torch.Tensor, smooth_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xs = x.float() * smooth_scale.float().view(1, -1)
    absmax = xs.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-10)
    scale = absmax / 127.0
    q = torch.round(xs / scale).clamp(-128, 127).to(torch.int8)
    return q.contiguous(), scale.float().contiguous()


def _reference(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    smooth_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, x_scale = _quant_reference(x, smooth_scale)
    packed = q.cpu().to(torch.int32) @ weight_t.cpu().to(torch.int32)
    packed = packed.to(device=x.device, dtype=torch.float32)
    out = x_scale * weight_scale.view(1, -1).float() * packed
    if bias is not None:
        out = out + bias.view(1, -1).float()
    return out.to(x.dtype).contiguous(), q, x_scale


def _reference_from_quant(
    q: torch.Tensor,
    x_scale: torch.Tensor,
    weight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    packed = q.cpu().to(torch.int32) @ weight_t.cpu().to(torch.int32)
    packed = packed.to(device=x_scale.device, dtype=torch.float32)
    out = x_scale.float() * weight_scale.view(1, -1).float() * packed
    if bias is not None:
        out = out + bias.view(1, -1).float()
    return out.to(dtype).contiguous()


def _call(output, x, weight_t_core, weight_scale, bias, q_input, input_scale, smooth_scale) -> None:
    from infinicore.lib import _infinicore

    _infinicore.deepseek_v4_lmslim_linear_w8a8_(
        _as_core(output),
        _as_core(x),
        weight_t_core,
        _as_core(weight_scale),
        None if bias is None else _as_core(bias),
        _as_core(q_input),
        _as_core(input_scale),
        _as_core(smooth_scale),
    )


def _run_case(tokens: int, k: int, n: int, dtype: torch.dtype, use_bias: bool, args) -> None:
    torch.manual_seed(args.seed + tokens * 13 + n)
    x = (torch.randn((tokens, k), device="cuda", dtype=dtype) * 0.7).contiguous()
    weight = torch.randint(-64, 64, (n, k), device="cuda", dtype=torch.int8).contiguous()
    weight_t = weight.t()
    weight_t_core = _as_core(weight).permute([1, 0])
    weight_scale = (torch.rand((n, 1), device="cuda", dtype=torch.float32) * 0.02 + 0.001).contiguous()
    smooth_scale = torch.ones((k,), device="cuda", dtype=torch.float32).contiguous()
    bias = None
    if use_bias:
        bias = (torch.randn((n,), device="cuda", dtype=dtype) * 0.1).contiguous()
    output = torch.empty((tokens, n), device="cuda", dtype=dtype)
    q_input = torch.empty((tokens, k), device="cuda", dtype=torch.int8)
    input_scale = torch.empty((tokens, 1), device="cuda", dtype=torch.float32)

    _call(output, x, weight_t_core, weight_scale, bias, q_input, input_scale, smooth_scale)
    _sync()

    ref, ref_q, ref_scale = _reference(x, weight_t, weight_scale, bias, smooth_scale)
    torch.testing.assert_close(q_input.float(), ref_q.float(), atol=1, rtol=0)
    torch.testing.assert_close(input_scale, ref_scale, atol=1.0e-6, rtol=1.0e-6)
    ref = _reference_from_quant(q_input, input_scale, weight_t, weight_scale, bias, dtype)
    torch.testing.assert_close(output.float(), ref.float(), atol=args.atol, rtol=args.rtol)

    max_diff = (output.float() - ref.float()).abs().max().item()
    print(f"tokens={tokens:<5d} n={n:<5d} bias={int(use_bias)} max_diff={max_diff:.6f}")

    if not args.skip_perf:
        for _ in range(args.warmup):
            _call(output, x, weight_t_core, weight_scale, bias, q_input, input_scale, smooth_scale)
        _sync()
        begin = time.perf_counter()
        for _ in range(args.iters):
            _call(output, x, weight_t_core, weight_scale, bias, q_input, input_scale, smooth_scale)
        _sync()
        avg_ms = (time.perf_counter() - begin) * 1000.0 / args.iters
        print(f"  avg={avg_ms:.4f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infinicore-repo", default=DEFAULT_INFINICORE_REPO)
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--out", type=int, default=1536)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--atol", type=float, default=2.0e-2)
    parser.add_argument("--rtol", type=float, default=2.0e-2)
    parser.add_argument("--skip-perf", action="store_true")
    parser.add_argument("--hygon", action="store_true")
    args = parser.parse_args()
    _add_paths(args)

    if not torch.cuda.is_available() and torch.cuda.device_count() == 0:
        raise RuntimeError("deepseek_v4_lmslim_linear_w8a8 test requires an accelerator device")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tokens_list = [int(x) for x in args.tokens.split(",") if x]
    for tokens in tokens_list:
        _run_case(tokens, args.hidden, args.out, dtype, False, args)
    _run_case(tokens_list[-1], args.hidden, args.out, dtype, True, args)
    print("deepseek_v4_lmslim_linear_w8a8 ok")


if __name__ == "__main__":
    main()

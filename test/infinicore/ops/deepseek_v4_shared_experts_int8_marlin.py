#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import time
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


def _sync() -> None:
    try:
        import infinicore

        infinicore.sync_stream()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _marlin_repack(weight: torch.Tensor, gemm2: bool) -> torch.Tensor:
    if weight.dim() != 3:
        raise ValueError("weight must be [E,N,K]")
    expert_out = []
    for expert in range(weight.shape[0]):
        transposed = weight[expert].transpose(0, 1).contiguous()
        size_k, size_n = transposed.shape
        if not gemm2:
            tmp = transposed.reshape(size_k // 64, 64, size_n).transpose(1, 2).contiguous()
            expert_out.append(tmp.reshape(size_k // 64, size_n * 64))
        else:
            tmp = (
                transposed.reshape(size_k // 64, 64, size_n // 16, 16)
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(size_k // 64, size_n // 16, 1, 16, 4, 16)
                .permute(0, 1, 2, 4, 3, 5)
                .contiguous()
            )
            expert_out.append(tmp.reshape(size_k // 64, size_n * 64))
    return torch.stack(expert_out, dim=0).contiguous()


def _quant_int8_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xf = x.float()
    absmax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-10)
    scale = absmax / 127.0
    q = torch.round(xf * (127.0 / absmax)).clamp(-128, 127).to(torch.int8)
    return q.contiguous(), scale.float().contiguous()


def _reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    q_hidden, hidden_scale = _quant_int8_per_token(hidden_states)
    gate_up = torch.matmul(q_hidden.float(), w1[0].transpose(0, 1).float())
    gate_up = gate_up * hidden_scale * w1_scale[0, :, 0].view(1, -1)
    gate_up = gate_up.to(torch.bfloat16)
    inter = gate_up.shape[-1] // 2
    activated = torch.nn.functional.silu(gate_up[:, :inter].float()) * gate_up[:, inter:].float()
    activated = activated.to(torch.bfloat16)
    q_activated, activated_scale = _quant_int8_per_token(activated)
    out = torch.matmul(q_activated.float(), w2[0].transpose(0, 1).float())
    out = out * activated_scale * w2_scale[0, :, 0].view(1, -1)
    return out.to(torch.bfloat16).contiguous()


def _call_infinicore(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1_marlin: torch.Tensor,
    w2_marlin: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gemm1_mode: int,
    gemm2_mode: int,
    delta: int,
) -> torch.Tensor:
    import infinicore
    from infinicore.lib import _infinicore

    _infinicore.deepseek_v4_shared_experts_impl_int8_marlin_(
        infinicore.from_torch(output)._underlying,
        infinicore.from_torch(hidden_states)._underlying,
        infinicore.from_torch(w1_marlin)._underlying,
        infinicore.from_torch(w2_marlin)._underlying,
        infinicore.from_torch(w1_scale)._underlying,
        infinicore.from_torch(w2_scale)._underlying,
        gemm1_mode,
        gemm2_mode,
        delta,
    )
    return output


def _call_fused_reference(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1_marlin: torch.Tensor,
    w2_marlin: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    import infinicore

    tokens = hidden_states.shape[0]
    topk_weights = torch.zeros((tokens, 6), device=hidden_states.device, dtype=torch.float32)
    topk_weights[:, 0] = 1.0
    topk_ids = torch.zeros((tokens, 6), device=hidden_states.device, dtype=torch.int32)
    infinicore.deepseek_v4_fused_experts_impl_int8_marlin_(
        infinicore.from_torch(output),
        infinicore.from_torch(hidden_states),
        infinicore.from_torch(w1_marlin),
        infinicore.from_torch(w2_marlin),
        infinicore.from_torch(topk_weights),
        infinicore.from_torch(topk_ids),
        infinicore.from_torch(w1_scale),
        infinicore.from_torch(w2_scale),
        1,
        1.0,
        False,
        None,
    )
    return output


def _bench(fn, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync()
    return (time.perf_counter() - start) * 1000.0 / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.add_argument("--infinicore-repo", default=DEFAULT_INFINICORE_REPO)
    parser.add_argument("--tokens", default="1,4,16,64")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--gemm1-mode", type=int, default=-1)
    parser.add_argument("--gemm2-mode", type=int, default=-1)
    parser.add_argument("--delta", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=3.0)
    parser.add_argument("--rtol", type=float, default=0.08)
    args = parser.parse_args()
    _add_paths(args)

    import infinicore  # noqa: F401

    if args.hygon:
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("deepseek_v4_shared_experts_int8_marlin test requires an accelerator device")
    if args.hidden % 64 != 0 or args.intermediate % 64 != 0:
        raise ValueError("hidden/intermediate must be divisible by 64")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    w1 = torch.randint(-32, 32, (1, args.intermediate * 2, args.hidden), dtype=torch.int8, generator=generator).contiguous().to(device)
    w2 = torch.randint(-32, 32, (1, args.hidden, args.intermediate), dtype=torch.int8, generator=generator).contiguous().to(device)
    w1_scale = (torch.rand((1, args.intermediate * 2, 1), dtype=torch.float32, generator=generator) * 0.02 + 0.001).contiguous().to(device)
    w2_scale = (torch.rand((1, args.hidden, 1), dtype=torch.float32, generator=generator) * 0.02 + 0.001).contiguous().to(device)
    w1_marlin = _marlin_repack(w1.cpu(), gemm2=False).to(device)
    w2_marlin = _marlin_repack(w2.cpu(), gemm2=True).to(device)

    print("case                 max_diff     ref_ms    kernel_ms  speedup")
    for tokens in [int(x) for x in args.tokens.split(",") if x.strip()]:
        torch.manual_seed(args.seed + tokens)
        hidden_states = torch.randn((tokens, args.hidden), device=device, dtype=torch.bfloat16)
        expected = torch.empty_like(hidden_states)
        _call_fused_reference(expected, hidden_states, w1_marlin, w2_marlin, w1_scale, w2_scale)
        _sync()
        output = torch.empty_like(hidden_states)
        _call_infinicore(output, hidden_states, w1_marlin, w2_marlin, w1_scale, w2_scale, args.gemm1_mode, args.gemm2_mode, args.delta)
        _sync()
        max_diff = (output.float() - expected.float()).abs().max().item()
        torch.testing.assert_close(output.float(), expected.float(), atol=args.atol, rtol=args.rtol)

        ref_ms = _bench(lambda: _call_fused_reference(expected, hidden_states, w1_marlin, w2_marlin, w1_scale, w2_scale), args.repeats, args.warmup)
        kernel_ms = _bench(
            lambda: _call_infinicore(output, hidden_states, w1_marlin, w2_marlin, w1_scale, w2_scale, args.gemm1_mode, args.gemm2_mode, args.delta),
            args.repeats,
            args.warmup,
        )
        print(f"tok{tokens:<6d}          {max_diff:<10.4f} {ref_ms:<9.4f} {kernel_ms:<10.4f} {ref_ms / kernel_ms:>.2f}x")

    print("deepseek_v4_shared_experts_int8_marlin ok")


if __name__ == "__main__":
    main()

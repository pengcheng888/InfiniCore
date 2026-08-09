#!/usr/bin/env python3
"""Compare SGLang and InfiniCore fused_experts_impl_int8_marlin.

This micro benchmark uses random DeepSeek-V4-shaped W8A8 routed-expert weights,
constructs the same Marlin packed layout used by InfiniLM, and compares:

  * torch.ops.sglang.fused_experts_impl_int8_marlin, inplace=True
  * torch.ops.sglang.fused_experts_impl_int8_marlin, inplace=False
  * torch.ops.sglang.fused_experts_impl_int8_marlin, inplace=False + copy_
  * infinicore.deepseek_v4_fused_experts_impl_int8_marlin_

The precision check compares InfiniCore output against SGLang inplace=True.

Run this script with `source ~/.bashrc`. In the current Hygon container, adding
`source /.myenv.sh` before importing lmslim/SGLang may trigger vLLM's mixed
rocm/cuda platform-plugin conflict.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch


DEFAULT_SGLANG_REPO = "/workspace/sglang"
DEFAULT_INFINICORE_REPO = "/workspace_codex/InfiniCore"
DEFAULT_NUM_EXPERTS = 256
DEFAULT_HIDDEN = 4096
DEFAULT_INTERMEDIATE = 2048


def _prepend(path: str | None) -> None:
    if path and Path(path).exists() and path not in sys.path:
        sys.path.insert(0, path)


def _add_paths(args) -> None:
    for root in (args.sglang_repo, os.environ.get("SGLANG_REPO")):
        if root:
            _prepend(str(Path(root) / "python"))
            _prepend(root)
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


def _make_random_experts(args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if args.num_experts < 1:
        raise ValueError(f"num_experts must be positive, got {args.num_experts}")
    if args.tp_size < 1 or args.tp_rank < 0 or args.tp_rank >= args.tp_size:
        raise ValueError(f"invalid tp_size/tp_rank: {args.tp_size}/{args.tp_rank}")
    if args.intermediate % args.tp_size != 0:
        raise ValueError(f"intermediate={args.intermediate} is not divisible by tp_size={args.tp_size}")
    if not (-128 <= args.weight_low < args.weight_high <= 128):
        raise ValueError("weight range must satisfy -128 <= weight_low < weight_high <= 128")
    if args.scale_min <= 0.0 or args.scale_max <= args.scale_min:
        raise ValueError("scale range must satisfy 0 < scale_min < scale_max")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 1024)
    local_intermediate = args.intermediate // args.tp_size
    w1 = torch.randint(
        args.weight_low,
        args.weight_high,
        (args.num_experts, local_intermediate, args.hidden),
        dtype=torch.int8,
        generator=generator,
    )
    w3 = torch.randint(
        args.weight_low,
        args.weight_high,
        (args.num_experts, local_intermediate, args.hidden),
        dtype=torch.int8,
        generator=generator,
    )
    w2 = torch.randint(
        args.weight_low,
        args.weight_high,
        (args.num_experts, args.hidden, local_intermediate),
        dtype=torch.int8,
        generator=generator,
    ).contiguous()
    w13 = torch.cat([w1, w3], dim=1).contiguous()
    w13_scale = (
        torch.rand((args.num_experts, 2 * local_intermediate, 1), dtype=torch.float32, generator=generator)
        * (args.scale_max - args.scale_min)
        + args.scale_min
    ).contiguous()
    w2_scale = (
        torch.rand((args.num_experts, args.hidden, 1), dtype=torch.float32, generator=generator)
        * (args.scale_max - args.scale_min)
        + args.scale_min
    ).contiguous()
    return w13, w2, w13_scale, w2_scale


def _marlin_repack(weight: torch.Tensor, gemm2: bool) -> torch.Tensor:
    if weight.dim() != 3:
        raise ValueError(f"expected [E,N,K], got {tuple(weight.shape)}")
    expert_count, size_n, size_k = weight.shape
    transposed = weight.transpose(1, 2).contiguous()
    if not gemm2:
        if size_k % 64 != 0:
            raise ValueError(f"GEMM1 K must be divisible by 64, got {size_k}")
        return (
            transposed.reshape(expert_count, size_k // 64, 64, size_n)
            .transpose(2, 3)
            .contiguous()
            .view(expert_count, size_k // 64, size_n * 64)
        )
    if size_k % 64 != 0 or size_n % 16 != 0:
        raise ValueError(f"GEMM2 requires K%64==0 and N%16==0, got N={size_n}, K={size_k}")
    return (
        transposed.reshape(expert_count, size_k // 64, 64, size_n // 16, 16)
        .permute(0, 1, 3, 4, 2)
        .contiguous()
        .view(expert_count, size_k // 64, size_n // 16, 1, 16, 4, 16)
        .permute(0, 1, 2, 3, 5, 4, 6)
        .contiguous()
    )


_SGLANG_LIB = None


def _register_sglang_op() -> None:
    """Register torch.ops.sglang.fused_experts_impl_int8_marlin.

    Importing the full SGLang quantization package can pull vLLM platform
    discovery into this mixed Hygon environment and fail with both rocm/cuda
    plugins active. SGLang's wrapper only forwards to lmslim and registers a
    custom op, so this local registration mirrors that small piece.
    """
    global _SGLANG_LIB
    if hasattr(torch.ops, "sglang") and hasattr(torch.ops.sglang, "fused_experts_impl_int8_marlin"):
        return

    from torch.library import Library
    from lmslim.layers.fused_moe.fuse_moe_int8_marlin import fused_experts_impl_int8_marlin

    def fused_experts_impl_int8_marlin_wrapper(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        inplace: bool = False,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        use_fp8_w8a8: bool = False,
        use_int8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        use_int4_w4a16: bool = False,
        per_channel_quant: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        w1_zp: Optional[torch.Tensor] = None,
        w2_zp: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        block_shape: Optional[List[int]] = None,
        use_nn_moe: Optional[bool] = False,
        routed_scaling_factor: Optional[float] = 1.0,
        shared_output: Optional[torch.Tensor] = None,
        i_q: Optional[torch.Tensor] = None,
        i_s: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return fused_experts_impl_int8_marlin(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace=inplace,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=w1_zp,
            w2_zp=w2_zp,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape,
            use_nn_moe=use_nn_moe,
            routed_scaling_factor=routed_scaling_factor,
            shared_output=shared_output,
            i_q=i_q,
            i_s=i_s,
        )

    _SGLANG_LIB = Library("sglang", "FRAGMENT")
    schema = torch.library.infer_schema(fused_experts_impl_int8_marlin_wrapper, mutates_args=[])
    _SGLANG_LIB.define("fused_experts_impl_int8_marlin" + schema)
    _SGLANG_LIB.impl("fused_experts_impl_int8_marlin", fused_experts_impl_int8_marlin_wrapper, "CUDA")


def _sglang_call(
    hidden_states: torch.Tensor,
    w13_marlin: torch.Tensor,
    w2_marlin: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    inplace: bool,
    global_num_experts: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    return torch.ops.sglang.fused_experts_impl_int8_marlin(
        hidden_states,
        w13_marlin,
        w2_marlin,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=inplace,
        activation="silu",
        apply_router_weight_on_input=False,
        use_int8_w8a8=True,
        per_channel_quant=True,
        global_num_experts=global_num_experts,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        a1_scale=None,
        a2_scale=None,
        use_nn_moe=False,
        routed_scaling_factor=routed_scaling_factor,
    )


def _infinicore_call(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w13_marlin: torch.Tensor,
    w2_marlin: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    global_num_experts: int,
    routed_scaling_factor: float,
    inplace: bool = False,
    shared_output: torch.Tensor | None = None,
) -> torch.Tensor:
    import infinicore

    return infinicore.deepseek_v4_fused_experts_impl_int8_marlin_(
        infinicore.from_torch(output),
        infinicore.from_torch(hidden_states),
        infinicore.from_torch(w13_marlin),
        infinicore.from_torch(w2_marlin),
        infinicore.from_torch(topk_weights),
        infinicore.from_torch(topk_ids),
        infinicore.from_torch(w13_scale),
        infinicore.from_torch(w2_scale),
        global_num_experts,
        routed_scaling_factor,
        inplace,
        infinicore.from_torch(shared_output) if shared_output is not None else None,
    )


def _make_infinicore_raw_call(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w13_marlin: torch.Tensor,
    w2_marlin: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    global_num_experts: int,
    routed_scaling_factor: float,
    inplace: bool = False,
    shared_output: torch.Tensor | None = None,
):
    import infinicore
    from infinicore.lib import _infinicore

    keepalive = [
        infinicore.from_torch(output),
        infinicore.from_torch(hidden_states),
        infinicore.from_torch(w13_marlin),
        infinicore.from_torch(w2_marlin),
        infinicore.from_torch(topk_weights),
        infinicore.from_torch(topk_ids),
        infinicore.from_torch(w13_scale),
        infinicore.from_torch(w2_scale),
    ]
    shared_underlying = None
    if shared_output is not None:
        shared_tensor = infinicore.from_torch(shared_output)
        keepalive.append(shared_tensor)
        shared_underlying = shared_tensor._underlying
    raw_args = [item._underlying for item in keepalive]

    def call():
        _infinicore.deepseek_v4_fused_experts_impl_int8_marlin_(
            raw_args[0],
            raw_args[1],
            raw_args[2],
            raw_args[3],
            raw_args[4],
            raw_args[5],
            raw_args[6],
            raw_args[7],
            global_num_experts,
            routed_scaling_factor,
            inplace,
            shared_underlying,
        )
        return output

    call._keepalive = keepalive
    return call


def _bench(name: str, fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    avg_ms = (time.perf_counter() - start) * 1000.0 / max(iters, 1)
    print(f"  {name:<32} {avg_ms:>10.4f} ms")
    return avg_ms


def _diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    a32 = a.float()
    b32 = b.float()
    finite = torch.isfinite(a32) & torch.isfinite(b32)
    if not finite.all():
        return float("nan"), float("nan"), float("nan")
    d = (a32 - b32).abs()
    return d.max().item(), d.mean().item(), (d / b32.abs().clamp_min(1e-6)).max().item()


def _finite_text(name: str, tensor: torch.Tensor) -> str:
    finite = torch.isfinite(tensor.float())
    return f"{name}: finite={finite.sum().item()}/{tensor.numel()}"


def _make_inputs(tokens: int, hidden: int, topk: int, experts: int, device: str, dtype: torch.dtype):
    hidden_states = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    topk_weights = torch.rand(tokens, topk, device=device, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if experts < topk:
        raise ValueError(f"experts={experts} must be >= topk={topk}")
    topk_ids = torch.rand(tokens, experts, device=device).topk(topk, dim=-1).indices.to(dtype)
    return hidden_states, topk_weights, topk_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang-repo", default=DEFAULT_SGLANG_REPO)
    parser.add_argument("--infinicore-repo", default=DEFAULT_INFINICORE_REPO)
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--intermediate", type=int, default=DEFAULT_INTERMEDIATE)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--tokens", default="1,4,16,64,256")
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--routed-scaling-factor", type=float, default=1.0)
    parser.add_argument("--topk-dtype", choices=["int32", "int64"], default="int32")
    parser.add_argument("--weight-low", type=int, default=-64)
    parser.add_argument("--weight-high", type=int, default=64)
    parser.add_argument("--scale-min", type=float, default=1e-3)
    parser.add_argument("--scale-max", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--hygon", action="store_true", help="Accepted for consistency with other InfiniCore tests.")
    args = parser.parse_args()

    _add_paths(args)
    import infinicore  # noqa: F401

    _register_sglang_op()

    torch.manual_seed(args.seed)
    num_experts = args.num_experts
    topk_dtype = torch.int32 if args.topk_dtype == "int32" else torch.int64

    print("配置")
    print(f"  experts: {num_experts}")
    print(f"  hidden/intermediate/topk: {args.hidden}/{args.intermediate}/{args.topk}")
    print(f"  tp_size/tp_rank: {args.tp_size}/{args.tp_rank}")
    print(f"  local_intermediate: {args.intermediate // args.tp_size}")
    print(f"  random int8 range: [{args.weight_low}, {args.weight_high})")
    print(f"  random scale range: [{args.scale_min}, {args.scale_max})")
    print(f"  tokens: {args.tokens}")
    print(f"  iters/warmup: {args.iters}/{args.warmup}")
    print()

    w13, w2, w13_scale, w2_scale = _make_random_experts(args)
    w13_marlin = _marlin_repack(w13, gemm2=False).to(args.device)
    w2_marlin = _marlin_repack(w2, gemm2=True).to(args.device)
    w13_scale = w13_scale.to(args.device)
    w2_scale = w2_scale.to(args.device)

    print("权重")
    print(f"  w13_marlin: {tuple(w13_marlin.shape)} {w13_marlin.dtype}")
    print(f"  w2_marlin:  {tuple(w2_marlin.shape)} {w2_marlin.dtype}")
    print(f"  w13_scale:  {tuple(w13_scale.shape)} {w13_scale.dtype}")
    print(f"  w2_scale:   {tuple(w2_scale.shape)} {w2_scale.dtype}")
    print()

    for tokens in [int(x.strip()) for x in args.tokens.split(",") if x.strip()]:
        hidden_states, topk_weights, topk_ids = _make_inputs(
            tokens, args.hidden, args.topk, num_experts, args.device, topk_dtype
        )

        ref_true = _sglang_call(
            hidden_states.clone(),
            w13_marlin,
            w2_marlin,
            topk_weights,
            topk_ids,
            w13_scale,
            w2_scale,
            True,
            num_experts,
            args.routed_scaling_factor,
        )
        ref_false = _sglang_call(
            hidden_states.clone(),
            w13_marlin,
            w2_marlin,
            topk_weights,
            topk_ids,
            w13_scale,
            w2_scale,
            False,
            num_experts,
            args.routed_scaling_factor,
        )
        inf_out = torch.empty_like(hidden_states)
        _infinicore_call(
            inf_out,
            hidden_states.clone(),
            w13_marlin,
            w2_marlin,
            topk_weights,
            topk_ids,
            w13_scale,
            w2_scale,
            num_experts,
            args.routed_scaling_factor,
            False,
        )
        inf_inplace_out = hidden_states.clone()
        _infinicore_call(
            inf_inplace_out,
            inf_inplace_out,
            w13_marlin,
            w2_marlin,
            topk_weights,
            topk_ids,
            w13_scale,
            w2_scale,
            num_experts,
            args.routed_scaling_factor,
            True,
        )
        shared_output = torch.randn_like(hidden_states)
        inf_shared_out = torch.empty_like(hidden_states)
        _infinicore_call(
            inf_shared_out,
            hidden_states.clone(),
            w13_marlin,
            w2_marlin,
            topk_weights,
            topk_ids,
            w13_scale,
            w2_scale,
            num_experts,
            args.routed_scaling_factor,
            False,
            shared_output,
        )
        expected_shared_out = inf_out + shared_output
        _sync()

        diff_inf_true = _diff(inf_out, ref_true)
        diff_inf_inplace_true = _diff(inf_inplace_out, ref_true)
        diff_inf_false = _diff(inf_out, ref_false)
        diff_true_false = _diff(ref_true, ref_false)
        diff_shared = _diff(inf_shared_out, expected_shared_out)
        ok_shared = torch.allclose(inf_shared_out.float(), expected_shared_out.float(), atol=args.atol, rtol=args.rtol)
        ok_true = torch.allclose(inf_out.float(), ref_true.float(), atol=args.atol, rtol=args.rtol)
        ok_inplace_true = torch.allclose(inf_inplace_out.float(), ref_true.float(), atol=args.atol, rtol=args.rtol)
        ok_false = torch.allclose(inf_out.float(), ref_false.float(), atol=args.atol, rtol=args.rtol)
        ok_sglang = torch.allclose(ref_true.float(), ref_false.float(), atol=args.atol, rtol=args.rtol)

        print("=" * 88)
        print(f"tokens={tokens}")
        print("- 精度")
        print(f"  {_finite_text('SGLang inplace=True', ref_true)}")
        print(f"  {_finite_text('SGLang inplace=False', ref_false)}")
        print(f"  {_finite_text('InfiniCore', inf_out)}")
        print(f"  {_finite_text('InfiniCore inplace=True', inf_inplace_out)}")
        print(
            "  InfiniCore vs SGLang inplace=True:  "
            f"max_abs={diff_inf_true[0]:.6g} mean_abs={diff_inf_true[1]:.6g} "
            f"max_rel={diff_inf_true[2]:.6g} allclose={ok_true}"
        )
        print(
            "  InfiniCore inplace vs SGLang True:  "
            f"max_abs={diff_inf_inplace_true[0]:.6g} mean_abs={diff_inf_inplace_true[1]:.6g} "
            f"max_rel={diff_inf_inplace_true[2]:.6g} allclose={ok_inplace_true}"
        )
        print(
            "  InfiniCore vs SGLang inplace=False: "
            f"max_abs={diff_inf_false[0]:.6g} mean_abs={diff_inf_false[1]:.6g} "
            f"max_rel={diff_inf_false[2]:.6g} allclose={ok_false}"
        )
        print(
            "  SGLang True vs False:              "
            f"max_abs={diff_true_false[0]:.6g} mean_abs={diff_true_false[1]:.6g} "
            f"max_rel={diff_true_false[2]:.6g} allclose={ok_sglang}"
        )
        print(
            "  InfiniCore shared_output path:     "
            f"max_abs={diff_shared[0]:.6g} mean_abs={diff_shared[1]:.6g} "
            f"max_rel={diff_shared[2]:.6g} allclose={ok_shared}"
        )
        print("- 性能")

        inplace_input = hidden_states.clone()
        false_input = hidden_states.clone()
        false_copy_dst = torch.empty_like(hidden_states)
        inf_input = hidden_states.clone()
        inf_output = torch.empty_like(hidden_states)
        inf_inplace_input = hidden_states.clone()
        inf_raw_input = hidden_states.clone()
        inf_raw_output = torch.empty_like(hidden_states)
        inf_raw_inplace_input = hidden_states.clone()
        inf_raw_call = _make_infinicore_raw_call(
            inf_raw_output,
            inf_raw_input,
            w13_marlin,
            w2_marlin,
            topk_weights,
            topk_ids,
            w13_scale,
            w2_scale,
            num_experts,
            args.routed_scaling_factor,
            False,
        )
        inf_raw_inplace_call = _make_infinicore_raw_call(
            inf_raw_inplace_input,
            inf_raw_inplace_input,
            w13_marlin,
            w2_marlin,
            topk_weights,
            topk_ids,
            w13_scale,
            w2_scale,
            num_experts,
            args.routed_scaling_factor,
            True,
        )

        t_inplace = _bench(
            "SGLang inplace=True",
            lambda: _sglang_call(
                inplace_input,
                w13_marlin,
                w2_marlin,
                topk_weights,
                topk_ids,
                w13_scale,
                w2_scale,
                True,
                num_experts,
                args.routed_scaling_factor,
            ),
            args.warmup,
            args.iters,
        )
        t_false = _bench(
            "SGLang inplace=False",
            lambda: _sglang_call(
                false_input,
                w13_marlin,
                w2_marlin,
                topk_weights,
                topk_ids,
                w13_scale,
                w2_scale,
                False,
                num_experts,
                args.routed_scaling_factor,
            ),
            args.warmup,
            args.iters,
        )

        def _false_copy():
            result = _sglang_call(
                false_input,
                w13_marlin,
                w2_marlin,
                topk_weights,
                topk_ids,
                w13_scale,
                w2_scale,
                False,
                num_experts,
                args.routed_scaling_factor,
            )
            false_copy_dst.copy_(result)
            return false_copy_dst

        t_false_copy = _bench("SGLang inplace=False+copy", _false_copy, args.warmup, args.iters)
        t_inf = _bench(
            "InfiniCore op",
            lambda: _infinicore_call(
                inf_output,
                inf_input,
                w13_marlin,
                w2_marlin,
                topk_weights,
                topk_ids,
                w13_scale,
                w2_scale,
                num_experts,
                args.routed_scaling_factor,
                False,
            ),
            args.warmup,
            args.iters,
        )
        t_inf_inplace = _bench(
            "InfiniCore op inplace=True",
            lambda: _infinicore_call(
                inf_inplace_input,
                inf_inplace_input,
                w13_marlin,
                w2_marlin,
                topk_weights,
                topk_ids,
                w13_scale,
                w2_scale,
                num_experts,
                args.routed_scaling_factor,
                True,
            ),
            args.warmup,
            args.iters,
        )
        t_inf_raw = _bench(
            "InfiniCore raw prewrapped",
            inf_raw_call,
            args.warmup,
            args.iters,
        )
        t_inf_raw_inplace = _bench(
            "InfiniCore raw inplace=True",
            inf_raw_inplace_call,
            args.warmup,
            args.iters,
        )

        print("- 比值")
        print(f"  SGLang false / inplace:          {t_false / t_inplace:.4f}x")
        print(f"  SGLang false+copy / inplace:     {t_false_copy / t_inplace:.4f}x")
        print(f"  InfiniCore / SGLang inplace:     {t_inf / t_inplace:.4f}x")
        print(f"  InfiniCore / SGLang false+copy:  {t_inf / t_false_copy:.4f}x")
        print(f"  InfiniCore inplace / SGLang:     {t_inf_inplace / t_inplace:.4f}x")
        print(f"  InfiniCore raw / SGLang inplace: {t_inf_raw / t_inplace:.4f}x")
        print(f"  InfiniCore raw inplace / SGLang: {t_inf_raw_inplace / t_inplace:.4f}x")
        print(f"  InfiniCore py / raw:             {t_inf / t_inf_raw:.4f}x")
        print(f"  InfiniCore py inplace / raw:     {t_inf_inplace / t_inf_raw_inplace:.4f}x")
        print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare SGLang and InfiniCore fused_experts_impl_int8_marlin.

This micro benchmark uses real DeepSeek-V4 layer0 W8A8 routed-expert weights,
constructs the same Marlin packed layout used by InfiniLM, and compares:

  * torch.ops.sglang.fused_experts_impl_int8_marlin, inplace=True
  * torch.ops.sglang.fused_experts_impl_int8_marlin, inplace=False
  * torch.ops.sglang.fused_experts_impl_int8_marlin, inplace=False + copy_
  * _infinicore.deepseek_v4_fused_experts_impl_int8_marlin_

The precision check compares InfiniCore output against SGLang inplace=True.

Run this script with `source ~/.bashrc`. In the current Hygon container, adding
`source /.myenv.sh` before importing lmslim/SGLang may trigger vLLM's mixed
rocm/cuda platform-plugin conflict.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
from safetensors.torch import load_file


DEFAULT_MODEL = "/data/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8-layer0"
DEFAULT_SGLANG_REPO = "/workspace/sglang"
DEFAULT_INFINICORE_REPO = "/workspace_codex/InfiniCore"


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


def _load_state_dict(model_dir: str) -> dict[str, torch.Tensor]:
    model_path = Path(model_dir)
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        files = sorted({model_path / name for name in index["weight_map"].values()})
    else:
        files = sorted(Path(p) for p in glob.glob(str(model_path / "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No safetensors files found under {model_dir}")

    state: dict[str, torch.Tensor] = {}
    for file in files:
        state.update(load_file(str(file), device="cpu"))
    return state


def _find_key(state: dict[str, torch.Tensor], layer: int, expert: int, name: str) -> str:
    candidates = [
        f"layers.{layer}.ffn.experts.{expert}.{name}",
        f"model.layers.{layer}.mlp.experts.{expert}.{name}",
        f"model.layers.{layer}.ffn.experts.{expert}.{name}",
        f"layers.{layer}.mlp.experts.{expert}.{name}",
    ]
    for key in candidates:
        if key in state:
            return key
    suffix = f"experts.{expert}.{name}"
    matches = [key for key in state if f"layers.{layer}" in key and key.endswith(suffix)]
    if matches:
        return sorted(matches)[0]
    raise KeyError(f"Cannot find layer={layer} expert={expert} {name}")


def _detect_num_experts(state: dict[str, torch.Tensor], layer: int) -> int:
    prefix = f"layers.{layer}.ffn.experts."
    experts = set()
    for key in state:
        if key.startswith(prefix) and key.endswith("w1.weight"):
            experts.add(int(key[len(prefix):].split(".", 1)[0]))
    if experts:
        return max(experts) + 1
    return 256


def _load_experts(
    state: dict[str, torch.Tensor],
    layer: int,
    num_experts: int,
    tp_size: int,
    tp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    w13, w2, w13_s, w2_s = [], [], [], []
    for expert in range(num_experts):
        w1 = state[_find_key(state, layer, expert, "w1.weight")]
        w3 = state[_find_key(state, layer, expert, "w3.weight")]
        w2_e = state[_find_key(state, layer, expert, "w2.weight")]
        w1_s = state[_find_key(state, layer, expert, "w1.scale")]
        w3_s = state[_find_key(state, layer, expert, "w3.scale")]
        w2_se = state[_find_key(state, layer, expert, "w2.scale")]

        if tp_size < 1 or tp_rank < 0 or tp_rank >= tp_size:
            raise ValueError(f"invalid tp_size/tp_rank: {tp_size}/{tp_rank}")
        if w1.shape[0] % tp_size != 0:
            raise ValueError(f"w1 intermediate {w1.shape[0]} is not divisible by tp_size={tp_size}")
        part = w1.shape[0] // tp_size
        start = tp_rank * part
        end = start + part
        w1 = w1[start:end, :]
        w3 = w3[start:end, :]
        w2_e = w2_e[:, start:end]
        w1_s = w1_s[start:end, :]
        w3_s = w3_s[start:end, :]

        w13.append(torch.cat([w1, w3], dim=0).contiguous())
        w2.append(w2_e.contiguous())
        w13_s.append(torch.cat([w1_s, w3_s], dim=0).contiguous())
        w2_s.append(w2_se.contiguous())
    return torch.stack(w13), torch.stack(w2), torch.stack(w13_s), torch.stack(w2_s)


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
    from infinicore.lib import _infinicore

    output_ic = infinicore.from_torch(output)
    hidden_states_ic = infinicore.from_torch(hidden_states)
    w13_marlin_ic = infinicore.from_torch(w13_marlin)
    w2_marlin_ic = infinicore.from_torch(w2_marlin)
    topk_weights_ic = infinicore.from_torch(topk_weights)
    topk_ids_ic = infinicore.from_torch(topk_ids)
    w13_scale_ic = infinicore.from_torch(w13_scale)
    w2_scale_ic = infinicore.from_torch(w2_scale)
    shared_output_ic = infinicore.from_torch(shared_output) if shared_output is not None else None

    _infinicore.deepseek_v4_fused_experts_impl_int8_marlin_(
        output_ic._underlying,
        hidden_states_ic._underlying,
        w13_marlin_ic._underlying,
        w2_marlin_ic._underlying,
        topk_weights_ic._underlying,
        topk_ids_ic._underlying,
        w13_scale_ic._underlying,
        w2_scale_ic._underlying,
        global_num_experts,
        routed_scaling_factor,
        inplace,
        shared_output_ic._underlying if shared_output_ic is not None else None,
    )
    return output


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
    topk_ids = torch.randint(0, experts, (tokens, topk), device=device, dtype=dtype)
    return hidden_states, topk_weights, topk_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sglang-repo", default=DEFAULT_SGLANG_REPO)
    parser.add_argument("--infinicore-repo", default=DEFAULT_INFINICORE_REPO)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--num-experts", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--tokens", default="1,4,16,64,256")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--routed-scaling-factor", type=float, default=1.0)
    parser.add_argument("--topk-dtype", choices=["int32", "int64"], default="int32")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--hygon", action="store_true", help="Accepted for consistency with other InfiniCore tests.")
    args = parser.parse_args()

    _add_paths(args)
    import infinicore  # noqa: F401

    _register_sglang_op()

    torch.manual_seed(args.seed)
    state = _load_state_dict(args.model)
    num_experts = args.num_experts or _detect_num_experts(state, args.layer)
    topk_dtype = torch.int32 if args.topk_dtype == "int32" else torch.int64

    print("配置")
    print(f"  model: {args.model}")
    print(f"  layer: {args.layer}")
    print(f"  experts: {num_experts}")
    print(f"  hidden/topk: {args.hidden}/{args.topk}")
    print(f"  tp_size/tp_rank: {args.tp_size}/{args.tp_rank}")
    print(f"  tokens: {args.tokens}")
    print(f"  iters/warmup: {args.iters}/{args.warmup}")
    print()

    w13, w2, w13_scale, w2_scale = _load_experts(state, args.layer, num_experts, args.tp_size, args.tp_rank)
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

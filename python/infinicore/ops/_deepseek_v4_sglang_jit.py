from __future__ import annotations

import os

os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")

from typing import Any

import torch


def _deepseek_v4():
    from sglang.jit_kernel import deepseek_v4

    return deepseek_v4


_FP8_E4M3_MAX = 448.0


def _apply_swiglu(input: torch.Tensor, swiglu_limit: float | None) -> torch.Tensor:
    gate, up = input.chunk(2, dim=-1)
    if swiglu_limit is not None:
        gate = torch.minimum(gate, torch.tensor(float(swiglu_limit), device=gate.device, dtype=gate.dtype))
        up = torch.clamp(up, min=-float(swiglu_limit), max=float(swiglu_limit))
    gate_f = gate.float()
    return (gate_f / (1.0 + torch.exp(-gate_f))) * up.float()


def _fp8_group_quant_natural(values: torch.Tensor, output: torch.Tensor, output_scale: torch.Tensor, group_size: int) -> None:
    if values.shape[-1] % group_size != 0:
        raise ValueError("hidden dimension must be divisible by quant_group_size")
    flat = values.reshape(-1, values.shape[-1])
    out_flat = output.reshape(-1, output.shape[-1])
    scale_flat = output_scale.reshape(-1, output.shape[-1] // group_size)
    groups = flat.reshape(flat.shape[0], -1, group_size)
    absmax = torch.clamp(groups.abs().amax(dim=-1), min=1.0e-10)
    scale = absmax / _FP8_E4M3_MAX
    quant = torch.clamp(groups / scale.unsqueeze(-1), min=-_FP8_E4M3_MAX, max=_FP8_E4M3_MAX)
    out_flat[: flat.shape[0]].copy_(quant.reshape_as(out_flat[: flat.shape[0]]).to(output.dtype))
    scale_flat[: flat.shape[0]].copy_(scale.to(output_scale.dtype))


def _ue8m0_group_quant(values: torch.Tensor, output: torch.Tensor, output_scale: torch.Tensor, group_size: int) -> None:
    if values.shape[-1] % group_size != 0:
        raise ValueError("hidden dimension must be divisible by quant_group_size")
    flat = values.reshape(-1, values.shape[-1])
    out_flat = output.reshape(-1, output.shape[-1])
    groups = flat.reshape(flat.shape[0], -1, group_size).float()
    raw_scale = torch.clamp(groups.abs().amax(dim=-1), min=1.0e-10) / _FP8_E4M3_MAX
    exp = torch.ceil(torch.log2(raw_scale)).to(torch.int32) + 127
    exp = torch.clamp(exp, min=0, max=255)
    scale = torch.pow(2.0, exp.float() - 127.0)
    quant = torch.clamp(groups / scale.unsqueeze(-1), min=-_FP8_E4M3_MAX, max=_FP8_E4M3_MAX)
    out_flat[: flat.shape[0]].copy_(quant.reshape_as(out_flat[: flat.shape[0]]).to(output.dtype))

    scale_bytes = output_scale.view(torch.uint8).reshape(output_scale.shape[0], -1)
    scale_bytes[: flat.shape[0], : exp.shape[-1]].copy_(exp.to(torch.uint8))


def _silu_and_mul_quant_fallback(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    masked_m: torch.Tensor | None,
    quant_group_size: int,
    scale_ue8m0: bool,
    transposed: bool,
    swiglu_limit: float | None,
    swizzle: bool,
) -> None:
    if scale_ue8m0 or transposed or swizzle:
        raise RuntimeError("fallback supports natural fp32-scale layout only")
    values = _apply_swiglu(input, swiglu_limit)
    if masked_m is None:
        _fp8_group_quant_natural(values, output, output_scale, quant_group_size)
        return

    counts = masked_m.detach().cpu().tolist()
    for expert_id, count in enumerate(counts):
        valid = int(count)
        if valid <= 0:
            continue
        _fp8_group_quant_natural(
            values[expert_id, :valid],
            output[expert_id, :valid],
            output_scale[expert_id, :valid],
            quant_group_size,
        )


def _mega_moe_pre_dispatch_fallback(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    buf_x: torch.Tensor,
    buf_x_sf: torch.Tensor,
    buf_topk_idx: torch.Tensor,
    buf_topk_weights: torch.Tensor,
    quant_group_size: int,
) -> None:
    rows = x.shape[0]
    _ue8m0_group_quant(x, buf_x[:rows], buf_x_sf[:rows], quant_group_size)
    buf_topk_idx[:rows].copy_(topk_idx.to(buf_topk_idx.dtype))
    buf_topk_weights[:rows].copy_(topk_weights)
    if buf_topk_idx.shape[0] > rows:
        buf_topk_idx[rows:].fill_(-1)
        buf_topk_weights[rows:].zero_()

def silu_and_mul_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    masked_m: torch.Tensor | None,
    quant_group_size: int,
    scale_ue8m0: bool,
    topk: int,
    transposed: bool,
    swiglu_limit: float | None,
    swizzle: bool,
) -> None:
    try:
        deepseek_v4 = _deepseek_v4()
        if masked_m is None:
            deepseek_v4.silu_and_mul_contig_post_quant(
                input,
                output,
                output_scale,
                quant_group_size,
                scale_ue8m0=scale_ue8m0,
                transposed=transposed,
                swiglu_limit=swiglu_limit,
                swizzle=swizzle,
            )
        else:
            deepseek_v4.silu_and_mul_masked_post_quant(
                input,
                output,
                output_scale,
                quant_group_size,
                masked_m,
                scale_ue8m0=scale_ue8m0,
                topk=topk,
                transposed=transposed,
                swiglu_limit=swiglu_limit,
                swizzle=swizzle,
            )
    except Exception:
        _silu_and_mul_quant_fallback(
            input,
            output,
            output_scale,
            masked_m,
            quant_group_size,
            scale_ue8m0,
            transposed,
            swiglu_limit,
            swizzle,
        )


def mega_moe_pre_dispatch(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    buf_x: torch.Tensor,
    buf_x_sf: torch.Tensor,
    buf_topk_idx: torch.Tensor,
    buf_topk_weights: torch.Tensor,
    quant_group_size: int,
) -> None:
    try:
        _deepseek_v4().mega_moe_pre_dispatch(
            x,
            topk_idx,
            topk_weights,
            buf_x,
            buf_x_sf,
            buf_topk_idx,
            buf_topk_weights,
            quant_group_size=quant_group_size,
        )
    except Exception:
        _mega_moe_pre_dispatch_fallback(
            x,
            topk_idx,
            topk_weights,
            buf_x,
            buf_x_sf,
            buf_topk_idx,
            buf_topk_weights,
            quant_group_size,
        )


def compressed_attn_metadata(
    compress_ratio: int,
    num_q_tokens: int,
    seq_lens: torch.Tensor,
    extend_lens: torch.Tensor | None,
    device_ref: torch.Tensor,
):
    return _deepseek_v4().compress_plan(
        compress_ratio,
        num_q_tokens,
        seq_lens,
        extend_lens,
        device_ref.device,
    )


def compressed_attn_prefill(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    plan: Any,
    extra_data: torch.Tensor | None,
    head_dim: int,
    compress_ratio: int,
    seq_lens: torch.Tensor | None,
    extend_lens: torch.Tensor | None,
) -> None:
    _deepseek_v4().compress_forward(
        kv_score_buffer,
        kv_score_input,
        ape,
        indices,
        plan=None if plan is None else plan,
        extra_data=extra_data,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        out=output,
        seq_lens=seq_lens,
        extend_lens=extend_lens,
    )


def compressed_attn_decode(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    plan: Any,
    extra_data: torch.Tensor | None,
    head_dim: int,
    compress_ratio: int,
    seq_lens: torch.Tensor | None,
) -> None:
    _deepseek_v4().compress_forward(
        kv_score_buffer,
        kv_score_input,
        ape,
        indices,
        plan=None if plan is None else plan,
        extra_data=extra_data,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        out=output,
        seq_lens=seq_lens,
        extend_lens=None,
    )


def flashmla_metadata(
    cache_seqlens: torch.Tensor | None,
    num_heads_per_head_k: int,
    num_heads_k: int,
    dense_fp8: bool,
    num_q_heads: int | None,
):
    import flash_mla

    if dense_fp8:
        if cache_seqlens is None:
            raise ValueError("cache_seqlens is required when dense_fp8=True")
        return flash_mla.get_mla_decoding_metadata_dense_fp8(
            cache_seqlens,
            num_heads_per_head_k,
            num_heads_k,
        )
    if num_q_heads is not None:
        return flash_mla.get_mla_metadata(
            cache_seqlens,
            num_heads_per_head_k,
            num_heads_k,
            num_q_heads,
        )
    return flash_mla.get_mla_metadata(cache_seqlens, num_heads_per_head_k, num_heads_k)


def flashmla_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor | None,
    cache_seqlens: torch.Tensor | None,
    output: torch.Tensor,
    tile_scheduler_metadata,
    num_splits: torch.Tensor | None,
    head_dim_v: int,
    softmax_scale: float | None,
    causal: bool,
    is_fp8_kvcache: bool,
    indices: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    extra_k_cache: torch.Tensor | None,
    extra_indices_in_kvcache: torch.Tensor | None,
    topk_length: torch.Tensor | None,
    extra_topk_length: torch.Tensor | None,
) -> None:
    import flash_mla

    out, _ = flash_mla.flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        softmax_scale=softmax_scale,
        causal=causal,
        is_fp8_kvcache=is_fp8_kvcache,
        indices=indices,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices_in_kvcache,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
    )
    output.copy_(out)


def flashmla_decode_q_nope_pe(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    output: torch.Tensor,
    tile_scheduler_metadata,
    num_splits: torch.Tensor | None,
    head_dim_v: int,
    softmax_scale: float | None,
    causal: bool,
    k_scale: torch.Tensor | None,
    kv_cache_dtype: str | None,
) -> None:
    import flash_mla

    if k_scale is None and kv_cache_dtype is None:
        out, _ = flash_mla.flash_mla_with_kvcache_q_nope_pe(
            q_nope=q_nope,
            q_pe=q_pe,
            k_cache=k_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            head_dim_v=head_dim_v,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    else:
        out, _ = flash_mla.flash_mla_with_kvcache_quantization_q_nope_pe(
            q_nope=q_nope,
            q_pe=q_pe,
            k_cache=k_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            head_dim_v=head_dim_v,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            softmax_scale=softmax_scale,
            causal=causal,
            k_scale=k_scale,
            kv_cache_dtype=kv_cache_dtype,
        )
    output.copy_(out)


def flashmla_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    sm_scale: float,
    d_v: int,
    attn_sink: torch.Tensor | None,
    topk_length: torch.Tensor | None,
    max_logits: torch.Tensor | None,
    lse: torch.Tensor | None,
) -> None:
    from flash_mla.flash_mla_interface import flash_mla_sparse_fwd

    out, max_logits_result, lse_result = flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )
    output.copy_(out)
    if max_logits is not None:
        max_logits.copy_(max_logits_result)
    if lse is not None:
        lse.copy_(lse_result)

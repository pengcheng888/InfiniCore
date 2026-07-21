from __future__ import annotations

from typing import Any

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor, from_torch


def _u(tensor: Tensor | None):
    return None if tensor is None else tensor._underlying


def deepseek_v4_silu_and_mul_quant_(
    input: Tensor,
    output: Tensor,
    output_scale: Tensor,
    masked_m: Tensor | None = None,
    quant_group_size: int = 128,
    scale_ue8m0: bool = False,
    topk: int = 8,
    transposed: bool = False,
    swiglu_limit: float | None = None,
    swizzle: bool = False,
) -> tuple[Tensor, Tensor]:
    _infinicore.deepseek_v4_sglang_jit_call_(
        "silu_and_mul_quant",
        input._underlying,
        input._underlying,
        output._underlying,
        output_scale._underlying,
        _u(masked_m),
        quant_group_size,
        scale_ue8m0,
        topk,
        transposed,
        swiglu_limit,
        swizzle,
    )
    return output, output_scale


def deepseek_v4_mega_moe_pre_dispatch_(
    x: Tensor,
    topk_idx: Tensor,
    topk_weights: Tensor,
    buf_x: Tensor,
    buf_x_sf: Tensor,
    buf_topk_idx: Tensor,
    buf_topk_weights: Tensor,
    quant_group_size: int = 32,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _infinicore.deepseek_v4_sglang_jit_call_(
        "mega_moe_pre_dispatch",
        x._underlying,
        x._underlying,
        topk_idx._underlying,
        topk_weights._underlying,
        buf_x._underlying,
        buf_x_sf._underlying,
        buf_topk_idx._underlying,
        buf_topk_weights._underlying,
        quant_group_size,
    )
    return buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights


def deepseek_v4_compressed_attn_metadata_(
    compress_ratio: int,
    num_q_tokens: int,
    seq_lens: Tensor,
    extend_lens: Tensor | None = None,
    device_ref: Tensor | None = None,
):
    anchor = seq_lens if device_ref is None else device_ref
    return _infinicore.deepseek_v4_sglang_jit_call_(
        "compressed_attn_metadata",
        anchor._underlying,
        compress_ratio,
        num_q_tokens,
        seq_lens._underlying,
        _u(extend_lens),
        anchor._underlying,
    )


def deepseek_v4_compressed_attn_prefill_(
    kv_score_buffer: Tensor,
    kv_score_input: Tensor,
    ape: Tensor,
    indices: Tensor,
    output: Tensor,
    plan: Any = None,
    extra_data: Tensor | None = None,
    head_dim: int | None = None,
    compress_ratio: int = 4,
    seq_lens: Tensor | None = None,
    extend_lens: Tensor | None = None,
) -> Tensor:
    if head_dim is None:
        head_dim = output.shape[-1]
    _infinicore.deepseek_v4_sglang_jit_call_(
        "compressed_attn_prefill",
        kv_score_input._underlying,
        kv_score_buffer._underlying,
        kv_score_input._underlying,
        ape._underlying,
        indices._underlying,
        output._underlying,
        plan,
        _u(extra_data),
        head_dim,
        compress_ratio,
        _u(seq_lens),
        _u(extend_lens),
    )
    return output


def deepseek_v4_compressed_attn_decode_(
    kv_score_buffer: Tensor,
    kv_score_input: Tensor,
    ape: Tensor,
    indices: Tensor,
    output: Tensor,
    plan: Any = None,
    extra_data: Tensor | None = None,
    head_dim: int | None = None,
    compress_ratio: int = 4,
    seq_lens: Tensor | None = None,
) -> Tensor:
    if head_dim is None:
        head_dim = output.shape[-1]
    _infinicore.deepseek_v4_sglang_jit_call_(
        "compressed_attn_decode",
        kv_score_input._underlying,
        kv_score_buffer._underlying,
        kv_score_input._underlying,
        ape._underlying,
        indices._underlying,
        output._underlying,
        plan,
        _u(extra_data),
        head_dim,
        compress_ratio,
        _u(seq_lens),
    )
    return output


def deepseek_v4_flashmla_metadata_(
    cache_seqlens: Tensor | None = None,
    num_heads_per_head_k: int = 1,
    num_heads_k: int = 1,
    dense_fp8: bool = False,
    num_q_heads: int | None = None,
):
    if cache_seqlens is None:
        import torch

        cache_seqlens_t = torch.empty((1,), device="cuda", dtype=torch.int32)
        cache_seqlens = from_torch(cache_seqlens_t)
    return _infinicore.deepseek_v4_sglang_jit_call_(
        "flashmla_metadata",
        cache_seqlens._underlying,
        _u(cache_seqlens),
        num_heads_per_head_k,
        num_heads_k,
        dense_fp8,
        num_q_heads,
    )


def deepseek_v4_flashmla_decode_(
    q: Tensor,
    k_cache: Tensor,
    block_table: Tensor | None,
    cache_seqlens: Tensor | None,
    output: Tensor,
    tile_scheduler_metadata,
    num_splits: Tensor | None = None,
    head_dim_v: int = 512,
    softmax_scale: float | None = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Tensor | None = None,
    attn_sink: Tensor | None = None,
    extra_k_cache: Tensor | None = None,
    extra_indices_in_kvcache: Tensor | None = None,
    topk_length: Tensor | None = None,
    extra_topk_length: Tensor | None = None,
) -> Tensor:
    _infinicore.deepseek_v4_sglang_jit_call_(
        "flashmla_decode",
        q._underlying,
        q._underlying,
        k_cache._underlying,
        _u(block_table),
        _u(cache_seqlens),
        output._underlying,
        tile_scheduler_metadata,
        _u(num_splits),
        head_dim_v,
        softmax_scale,
        causal,
        is_fp8_kvcache,
        _u(indices),
        _u(attn_sink),
        _u(extra_k_cache),
        _u(extra_indices_in_kvcache),
        _u(topk_length),
        _u(extra_topk_length),
    )
    return output


def deepseek_v4_flashmla_decode_q_nope_pe_(
    q_nope: Tensor,
    q_pe: Tensor,
    k_cache: Tensor,
    block_table: Tensor,
    cache_seqlens: Tensor,
    output: Tensor,
    tile_scheduler_metadata,
    num_splits: Tensor | None = None,
    head_dim_v: int = 512,
    softmax_scale: float | None = None,
    causal: bool = False,
    k_scale: Tensor | None = None,
    kv_cache_dtype: str | None = None,
) -> Tensor:
    _infinicore.deepseek_v4_sglang_jit_call_(
        "flashmla_decode_q_nope_pe",
        q_nope._underlying,
        q_nope._underlying,
        q_pe._underlying,
        k_cache._underlying,
        block_table._underlying,
        cache_seqlens._underlying,
        output._underlying,
        tile_scheduler_metadata,
        _u(num_splits),
        head_dim_v,
        softmax_scale,
        causal,
        _u(k_scale),
        kv_cache_dtype,
    )
    return output


def deepseek_v4_flashmla_sparse_prefill_(
    q: Tensor,
    kv: Tensor,
    indices: Tensor,
    output: Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Tensor | None = None,
    topk_length: Tensor | None = None,
    max_logits: Tensor | None = None,
    lse: Tensor | None = None,
) -> Tensor:
    _infinicore.deepseek_v4_sglang_jit_call_(
        "flashmla_sparse_prefill",
        q._underlying,
        q._underlying,
        kv._underlying,
        indices._underlying,
        output._underlying,
        sm_scale,
        d_v,
        _u(attn_sink),
        _u(topk_length),
        _u(max_logits),
        _u(lse),
    )
    return output

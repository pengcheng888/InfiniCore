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
    _dsv4_dbg("exit")


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
    _dsv4_dbg("exit")


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
    _dsv4_dbg("exit")
    if max_logits is not None:
        max_logits.copy_(max_logits_result)
    if lse is not None:
        lse.copy_(lse_result)


_DSV4_FREQS_CACHE = {}


def _dsv4_dbg(msg):
    import os, sys
    if os.environ.get("INFINILM_DSV4_ATTN_DEBUG"):
        print("[InfiniLM-DSV4-Attn] " + str(msg), file=sys.stderr, flush=True)


def _dsv4_get_freqs(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow, device):
    key = (int(dim), int(seqlen), int(original_seq_len), float(base), float(factor), float(beta_fast), float(beta_slow), str(device))
    freqs = _DSV4_FREQS_CACHE.get(key)
    if freqs is None:
        from sglang.srt.layers.deepseek_v4_rope import precompute_freqs_cis

        freqs = precompute_freqs_cis(
            dim=int(dim),
            seqlen=int(seqlen),
            original_seq_len=int(original_seq_len),
            base=float(base),
            factor=float(factor),
            beta_fast=float(beta_fast),
            beta_slow=float(beta_slow),
        ).to(device=device)
        _DSV4_FREQS_CACHE[key] = freqs
    return freqs


def _dsv4_build_swa_indices(block_tables, positions, window_size: int = 128, block_size: int = 256):
    import torch

    num_tokens = positions.numel()
    offsets = torch.arange(window_size, device=positions.device, dtype=torch.int64)
    abs_pos = positions.to(torch.int64).unsqueeze(1) - offsets.unsqueeze(0)
    valid = abs_pos >= 0
    block_ids = torch.clamp(abs_pos // block_size, min=0)
    block_offsets = abs_pos % block_size
    # Current InfiniLM scheduler packs one request in the validation path; keep a clear error for future batching work.
    if block_tables.shape[0] != 1:
        raise RuntimeError('DeepSeekV4 attention bridge currently expects batch=1 metadata')
    physical_blocks = block_tables[0].to(torch.int64).gather(0, torch.clamp(block_ids.reshape(-1), max=block_tables.shape[1] - 1)).reshape(num_tokens, window_size)
    raw = physical_blocks * block_size + block_offsets
    raw = torch.where(valid, raw, torch.full_like(raw, -1))
    return raw.to(torch.int32).unsqueeze(1)


def _dsv4_build_extra_indices(positions, ratio: int, topk: int, page_size: int):
    import torch

    num_tokens = positions.numel()
    compressed_visible = torch.div(positions.to(torch.int64) + 1, ratio, rounding_mode='floor')
    ar = torch.arange(topk, device=positions.device, dtype=torch.int64).unsqueeze(0)
    valid = ar < compressed_visible.unsqueeze(1)
    raw = torch.where(valid, ar, torch.full_like(ar, -1))
    # The bridge stores compressed blocks densely, so physical compressed token ids equal logical ids.
    topk_len = torch.clamp(compressed_visible, min=1, max=topk).to(torch.int32)
    return raw.to(torch.int32).unsqueeze(1), topk_len


def infinilm_dsv4_attention_forward(
    q,
    kv,
    positions,
    block_tables,
    slot_mapping,
    total_sequence_lengths,
    input_offsets,
    dsv4_swa_indices,
    dsv4_swa_topk_lengths,
    dsv4_c4_indices,
    dsv4_c4_topk_lengths,
    dsv4_c128_indices,
    dsv4_c128_topk_lengths,
    dsv4_raw_out_loc,
    dsv4_page_table,
    dsv4_seq_lens_casual,
    dsv4_positions_casual,
    dsv4_c4_out_loc,
    dsv4_c4_positions,
    dsv4_c4_topk_lengths_raw,
    dsv4_c4_topk_lengths_clamp1,
    dsv4_c4_sparse_indices,
    dsv4_c4_sparse_topk_lengths,
    dsv4_c128_out_loc,
    dsv4_c128_positions,
    dsv4_c128_page_indices,
    dsv4_c128_topk_lengths_clamp1,
    dsv4_c4_compress_write_loc,
    dsv4_c4_compress_extra_loc,
    dsv4_c4_compress_state_indices,
    dsv4_c128_compress_write_loc,
    dsv4_c128_compress_state_indices,
    swa_cache_raw,
    c4_cache_raw,
    c128_cache_raw,
    c4_indexer_cache_raw,
    attn_sink,
    output,
    kv_scale,
    compressor_kv_score_input,
    compressor_ape,
    compressor_norm_weight,
    compressor_state,
    indexer_q,
    indexer_weights,
    indexer_weight_scale: float,
    indexer_num_heads: int,
    indexer_kv_score_input,
    indexer_ape,
    indexer_norm_weight,
    indexer_compressor_state,
    indexer_head_dim: int,
    compress_ratio: int,
    num_local_heads: int,
    head_dim: int,
    rope_dim: int,
    max_position_embeddings: int,
    rope_theta: float,
    compress_rope_theta: float,
    rope_factor: float,
    rope_beta_fast: float,
    rope_beta_slow: float,
    rope_original_seq_len: int,
    rms_norm_eps: float,
):
    import torch
    import flash_mla
    from sglang.jit_kernel.deepseek_v4 import (
        compress_forward,
        compress_fused_norm_rope_inplace,
        fused_rope,
        fused_store_cache,
        topk_transform_512,
    )
    from sglang.srt.layers.attention.compressed.indexer import fused_scale
    from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation
    from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

    def _as_i32(tensor, *, flatten: bool = False):
        if tensor is None:
            return None
        tensor = tensor.to(device=q.device, dtype=torch.int32)
        return tensor.reshape(-1) if flatten else tensor

    def _first_i32(*tensors, flatten: bool = False):
        for tensor in tensors:
            value = _as_i32(tensor, flatten=flatten)
            if value is not None:
                return value
        return None

    _dsv4_dbg(f"enter q={tuple(q.shape)} kv={tuple(kv.shape)} cr={compress_ratio}")
    if q.numel() == 0:
        return
    if q.dim() != 3 or kv.dim() != 2:
        raise RuntimeError(f'DeepSeekV4 attention bridge expects q [T,H,D], kv [T,D], got {q.shape=} {kv.shape=}')

    num_tokens = q.shape[0]
    positions = _as_i32(positions, flatten=True)
    slot_mapping = _as_i32(slot_mapping, flatten=True)
    raw_out_loc = _first_i32(dsv4_raw_out_loc, slot_mapping, flatten=True)
    positions_casual = _first_i32(dsv4_positions_casual, positions, flatten=True)
    seq_lens_casual = _first_i32(dsv4_seq_lens_casual, flatten=True)
    if seq_lens_casual is None:
        seq_lens_casual = positions_casual + 1
    page_table = _first_i32(dsv4_page_table, block_tables, flatten=False)

    def _full_to_swa(kv_indices):
        kv_indices = _as_i32(kv_indices, flatten=False)
        if kv_indices is None:
            return None
        if swa_cache_raw.shape[0] <= 0:
            raise RuntimeError('DeepSeekV4 SWA cache must have at least one page')
        invalid = kv_indices < 0
        safe = torch.where(invalid, torch.zeros_like(kv_indices), kv_indices)
        swa_pages = torch.remainder(safe // 256, int(swa_cache_raw.shape[0]))
        swa_loc = swa_pages * 256 + torch.remainder(safe, 256)
        return torch.where(invalid, torch.full_like(swa_loc, -1), swa_loc).to(torch.int32)

    def _full_locs_from_positions(pos_tensor):
        pos_tensor = _as_i32(pos_tensor, flatten=True)
        if pos_tensor is None or page_table is None:
            return None
        if pos_tensor.numel() != num_tokens or page_table.shape[0] != num_tokens:
            return None
        invalid = pos_tensor < 0
        safe_pos = torch.where(invalid, torch.zeros_like(pos_tensor), pos_tensor)
        page_idx = safe_pos // 256
        page_idx = torch.clamp(page_idx, min=0, max=max(int(page_table.shape[1]) - 1, 0))
        row_idx = torch.arange(num_tokens, device=q.device, dtype=torch.long)
        block = page_table[row_idx, page_idx.to(torch.long)]
        full_loc = block * 256 + torch.remainder(safe_pos, 256)
        invalid = invalid | (block < 0)
        return torch.where(invalid, torch.full_like(full_loc, -1), full_loc).to(torch.int32)

    def _state_indices_from_full_locs(full_locs, ratio_value: int):
        full_locs = _as_i32(full_locs, flatten=True)
        if full_locs is None:
            return None
        ring = 8 if int(ratio_value) == 4 else 128
        swa_locs = _full_to_swa(full_locs).reshape(-1)
        invalid = swa_locs < 0
        safe = torch.where(invalid, torch.zeros_like(swa_locs), swa_locs)
        state_loc = (safe // 256) * ring + torch.remainder(safe, ring)
        state_idx = state_loc // int(ratio_value)
        return torch.where(invalid, torch.full_like(state_idx, -1), state_idx).to(torch.int32)

    if positions_casual.numel() != num_tokens or raw_out_loc.numel() != num_tokens:
        raise RuntimeError(
            'DeepSeekV4 metadata length mismatch: '
            f'{num_tokens=} positions={tuple(positions_casual.shape)} raw_out={tuple(raw_out_loc.shape)}'
        )

    base = compress_rope_theta if int(compress_ratio) != 0 else rope_theta
    original = rope_original_seq_len if int(compress_ratio) != 0 else 0
    _dsv4_dbg(f"positions={tuple(positions_casual.shape)} page_table={None if page_table is None else tuple(page_table.shape)} max_pos={max_position_embeddings} base={base} original={original}")
    freqs = _dsv4_get_freqs(
        rope_dim,
        max_position_embeddings,
        original,
        base,
        rope_factor,
        rope_beta_fast,
        rope_beta_slow,
        q.device,
    )
    _dsv4_dbg(f"freqs={tuple(freqs.shape)} dtype={freqs.dtype}")

    _dsv4_dbg("before rope")
    fused_rope(q[..., -rope_dim:], kv[..., -rope_dim:].unsqueeze(1), freqs, positions=positions_casual)
    _dsv4_dbg("after rope")

    swa_out_loc = _full_to_swa(raw_out_loc).reshape(-1)
    _dsv4_dbg(f"before swa store cache={tuple(swa_cache_raw.shape)} raw_out={tuple(raw_out_loc.shape)} swa_out={tuple(swa_out_loc.shape)}")
    valid_raw = swa_out_loc >= 0
    if bool(valid_raw.any().item()):
        fused_store_cache(
            input=kv[valid_raw].contiguous(),
            cache=swa_cache_raw,
            indices=swa_out_loc[valid_raw].contiguous(),
            page_size=256,
            type='flashmla',
        )
    _dsv4_dbg("after swa store")
    fp8_dtype = torch.float8_e4m3fn
    swa_cache = swa_cache_raw.view(fp8_dtype).as_strided((swa_cache_raw.shape[0], 256, 1, 584), (swa_cache_raw.stride(0), 584, 584, 1))

    swa_indices = _full_to_swa(dsv4_swa_indices)
    if swa_indices.dim() == 2:
        swa_indices = swa_indices.unsqueeze(1)
    swa_topk = _as_i32(dsv4_swa_topk_lengths, flatten=True)

    extra_cache = None
    extra_indices = None
    extra_topk = None
    if int(compress_ratio) in (4, 128):
        if compressor_kv_score_input is None or compressor_ape is None or compressor_norm_weight is None or compressor_state is None:
            raise RuntimeError('DeepSeekV4 compressed attention requires compressor tensors')
        ratio = int(compress_ratio)
        total_lens = _as_i32(total_sequence_lengths, flatten=True)
        offsets = _as_i32(input_offsets, flatten=True)
        extend_lens = None
        if num_tokens != total_lens.numel():
            extend_lens = (offsets[1:] - offsets[:-1]).to(torch.int32)
        _dsv4_dbg("before compress plan")
        plan = __import__('sglang.jit_kernel.deepseek_v4', fromlist=['compress_plan']).compress_plan(
            ratio,
            num_tokens,
            total_lens,
            extend_lens,
            q.device,
        )
        fallback_state_indices = _state_indices_from_full_locs(raw_out_loc, ratio)
        extra_data = None
        compress_page_size = ratio
        if ratio == 4:
            c4_write_full_locs = _full_locs_from_positions(dsv4_c4_positions)
            c4_write_state_indices = _state_indices_from_full_locs(c4_write_full_locs, 4)
            compress_indices = _first_i32(
                c4_write_state_indices,
                dsv4_c4_compress_write_loc,
                dsv4_c4_compress_state_indices,
                fallback_state_indices,
                flatten=True,
            )
            if extend_lens is None:
                c4_prev_positions = None if dsv4_c4_positions is None else _as_i32(dsv4_c4_positions, flatten=True) - 4
                c4_prev_full_locs = _full_locs_from_positions(c4_prev_positions)
                c4_prev_state_indices = _state_indices_from_full_locs(c4_prev_full_locs, 4)
                if c4_prev_state_indices is not None:
                    c4_prev_state_indices = torch.clamp(c4_prev_state_indices, min=0).view(-1, 1)
                extra_data = _first_i32(c4_prev_state_indices, dsv4_c4_compress_extra_loc, flatten=False)
            else:
                extra_data = _as_i32(dsv4_c4_compress_extra_loc, flatten=False)
            expected_extra_width = 4 if extend_lens is not None else 1
            if extra_data is not None and (extra_data.dim() != 2 or extra_data.shape[-1] != expected_extra_width):
                _dsv4_dbg(
                    f"ignore c4 extra_data shape={tuple(extra_data.shape)} expected_width={expected_extra_width}"
                )
                extra_data = None
            compress_page_size = 4 if extra_data is not None else 8
        else:
            c128_write_full_locs = _full_locs_from_positions(dsv4_c128_positions)
            c128_write_state_indices = _state_indices_from_full_locs(c128_write_full_locs, 128)
            compress_indices = _first_i32(
                c128_write_state_indices,
                dsv4_c128_compress_write_loc,
                dsv4_c128_compress_state_indices,
                fallback_state_indices,
                flatten=True,
            )
        indexer_comp_out = None
        if ratio == 4 and c4_indexer_cache_raw is not None:
            if indexer_kv_score_input is None or indexer_ape is None or indexer_norm_weight is None or indexer_compressor_state is None:
                raise RuntimeError('DeepSeekV4 C4 indexer cache requires indexer compressor tensors')
            idx_head_dim = int(indexer_head_dim)
            idx_state_rows = (indexer_compressor_state.shape[0] // compress_page_size) * compress_page_size
            idx_state_for_compress = indexer_compressor_state[:idx_state_rows].view(
                -1,
                compress_page_size,
                indexer_compressor_state.shape[-1],
            )
            _dsv4_dbg(
                f"before indexer compress state={tuple(idx_state_for_compress.shape)} "
                f"inp={tuple(indexer_kv_score_input.shape)} extra={None if extra_data is None else tuple(extra_data.shape)}"
            )
            indexer_comp_out = compress_forward(
                kv_score_buffer=idx_state_for_compress,
                kv_score_input=indexer_kv_score_input,
                ape=indexer_ape.reshape(-1, idx_head_dim).to(torch.bfloat16),
                indices=compress_indices,
                plan=plan,
                extra_data=extra_data,
                head_dim=idx_head_dim,
                compress_ratio=4,
            )
            compress_fused_norm_rope_inplace(indexer_comp_out, indexer_norm_weight, float(rms_norm_eps), freqs, plan)
            indexer_comp_out = rotate_activation(indexer_comp_out)
            _dsv4_dbg(f"after indexer compress out={tuple(indexer_comp_out.shape)}")

        indexer_topk_ready = (
            ratio == 4
            and c4_indexer_cache_raw is not None
            and indexer_q is not None
            and indexer_weights is not None
            and dsv4_c4_topk_lengths_raw is not None
            and page_table is not None
        )

        state_rows = (compressor_state.shape[0] // compress_page_size) * compress_page_size
        state_for_compress = compressor_state[:state_rows].view(-1, compress_page_size, compressor_state.shape[-1])
        _dsv4_dbg(f"before compress state={tuple(state_for_compress.shape)} inp={tuple(compressor_kv_score_input.shape)} extra={None if extra_data is None else tuple(extra_data.shape)}")
        comp_out = compress_forward(
            kv_score_buffer=state_for_compress,
            kv_score_input=compressor_kv_score_input,
            ape=compressor_ape.reshape(-1, head_dim).to(torch.bfloat16),
            indices=compress_indices,
            plan=plan,
            extra_data=extra_data,
            head_dim=head_dim,
            compress_ratio=ratio,
        )
        _dsv4_dbg(f"after compress out={tuple(comp_out.shape)}")
        compress_fused_norm_rope_inplace(comp_out, compressor_norm_weight, float(rms_norm_eps), freqs, plan)
        _dsv4_dbg("after compress norm rope")
        if ratio == 4:
            c4_cache_loc = _first_i32(dsv4_c4_out_loc, raw_out_loc // 4, flatten=True)
            write_mask = ((positions_casual + 1) % 4) == 0
            if bool(write_mask.any().item()):
                fused_store_cache(
                    input=comp_out[write_mask].contiguous(),
                    cache=c4_cache_raw,
                    indices=c4_cache_loc[write_mask].contiguous(),
                    page_size=64,
                    type='flashmla',
                )
                if indexer_comp_out is not None:
                    fused_store_cache(
                        input=indexer_comp_out[write_mask].contiguous(),
                        cache=c4_indexer_cache_raw,
                        indices=c4_cache_loc[write_mask].contiguous(),
                        page_size=64,
                        type='indexer',
                    )
            extra_cache = c4_cache_raw.view(fp8_dtype).as_strided((c4_cache_raw.shape[0], 64, 1, 584), (c4_cache_raw.stride(0), 584, 584, 1))
            extra_indices = _first_i32(dsv4_c4_sparse_indices, dsv4_c4_indices, flatten=False)
            if indexer_topk_ready and extra_indices is not None:
                c4_seq_lens = _as_i32(dsv4_c4_topk_lengths_raw, flatten=True)
                q_indexer = indexer_q.contiguous()
                fused_rope(q_indexer[..., -rope_dim:], None, freqs, positions=positions_casual)
                q_indexer = rotate_activation(q_indexer)
                q_fp8, q_scale = act_quant(q_indexer)
                weights = fused_scale(indexer_weights.contiguous(), float(indexer_weight_scale), q_scale.contiguous()).squeeze(2)
                indexer_cache = c4_indexer_cache_raw.view(
                    c4_indexer_cache_raw.shape[0],
                    64,
                    1,
                    int(indexer_head_dim) + 4,
                )
                from lightop.gemmopt import paged_mqa_logits

                c4_seq_lens_arg = c4_seq_lens.unsqueeze(-1) if c4_seq_lens.dim() == 1 else c4_seq_lens
                logits = paged_mqa_logits(
                    q_fp8.unsqueeze(1),
                    indexer_cache,
                    weights,
                    c4_seq_lens_arg,
                    page_table,
                    None,
                    int(page_table.shape[1]) * 64,
                    False,
                )
                topk_transform_512(logits, c4_seq_lens, page_table, extra_indices, 64, None)
                _dsv4_dbg(f"after indexer topk indices={tuple(extra_indices.shape)} logits={tuple(logits.shape)}")
            if extra_indices.dim() == 2:
                extra_indices = extra_indices.unsqueeze(1)
            extra_topk = _first_i32(
                dsv4_c4_sparse_topk_lengths,
                dsv4_c4_topk_lengths_clamp1,
                dsv4_c4_topk_lengths,
                flatten=True,
            )
        else:
            c128_cache_loc = _first_i32(dsv4_c128_out_loc, raw_out_loc // 128, flatten=True)
            write_mask = ((positions_casual + 1) % 128) == 0
            if bool(write_mask.any().item()):
                fused_store_cache(
                    input=comp_out[write_mask].contiguous(),
                    cache=c128_cache_raw,
                    indices=c128_cache_loc[write_mask].contiguous(),
                    page_size=2,
                    type='flashmla',
                )
            extra_cache = c128_cache_raw.view(fp8_dtype).as_strided((c128_cache_raw.shape[0], 2, 1, 584), (c128_cache_raw.stride(0), 584, 584, 1))
            extra_indices = _first_i32(dsv4_c128_page_indices, dsv4_c128_indices, flatten=False)
            if extra_indices.dim() == 2:
                extra_indices = extra_indices.unsqueeze(1)
            extra_topk = _first_i32(dsv4_c128_topk_lengths_clamp1, dsv4_c128_topk_lengths, flatten=True)

    _dsv4_dbg("before flash metadata")
    meta, num_splits = flash_mla.get_mla_metadata(None, int(num_local_heads), 1, int(num_local_heads))
    _dsv4_dbg("before flashmla")
    out, _ = flash_mla.flash_mla_with_kvcache(
        q=q.reshape(num_tokens, 1, int(num_local_heads), int(head_dim)),
        k_cache=swa_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=meta,
        num_splits=num_splits,
        softmax_scale=float(head_dim) ** -0.5,
        causal=False,
        is_fp8_kvcache=True,
        indices=swa_indices,
        attn_sink=attn_sink,
        extra_k_cache=extra_cache,
        extra_indices_in_kvcache=extra_indices,
        topk_length=swa_topk,
        extra_topk_length=extra_topk,
    )
    _dsv4_dbg("after flashmla")
    out = out.reshape(num_tokens, int(num_local_heads), int(head_dim))
    fused_rope(out[..., -rope_dim:], None, freqs, positions=positions_casual, inverse=True)
    output.copy_(out)
    _dsv4_dbg("exit")

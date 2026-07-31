from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def deepseek_v4_compress_fused_norm_rope_(
    input: Tensor,
    norm_weight: Tensor,
    epsilon: float,
    freqs_cis: Tensor,
    positions: Tensor,
) -> Tensor:
    _infinicore.deepseek_v4_compress_fused_norm_rope_(
        input._underlying,
        norm_weight._underlying,
        epsilon,
        freqs_cis._underlying,
        positions._underlying,
    )
    return input

def deepseek_v4_c4_compress_stateful(
    kv_score_input: Tensor,
    ape: Tensor,
    compressor_state: Tensor,
    write_loc: Tensor,
    extra_loc: Tensor,
    positions: Tensor,
) -> Tensor:
    return Tensor(
        _infinicore.deepseek_v4_c4_compress_stateful(
            kv_score_input._underlying,
            ape._underlying,
            compressor_state._underlying,
            write_loc._underlying,
            extra_loc._underlying,
            positions._underlying,
        )
    )

def deepseek_v4_c128_compress_stateful(
    kv_score_input: Tensor,
    ape: Tensor,
    compressor_state: Tensor,
    write_loc: Tensor,
    positions: Tensor,
) -> Tensor:
    return Tensor(
        _infinicore.deepseek_v4_c128_compress_stateful(
            kv_score_input._underlying,
            ape._underlying,
            compressor_state._underlying,
            write_loc._underlying,
            positions._underlying,
        )
    )


def deepseek_v4_flashmla_sparse_attention_(
    q: Tensor,
    raw_cache: Tensor,
    indices: Tensor,
    topk_lengths: Tensor,
    attn_sink: Tensor | None,
    output: Tensor,
    softmax_scale: float,
    page_size: int = 256,
    head_dim_v: int = 512,
    extra_raw_cache: Tensor | None = None,
    extra_indices: Tensor | None = None,
    extra_topk_lengths: Tensor | None = None,
    extra_page_size: int = 0,
) -> Tensor:
    _infinicore.deepseek_v4_flashmla_sparse_attention_(
        q._underlying,
        raw_cache._underlying,
        indices._underlying,
        topk_lengths._underlying,
        None if attn_sink is None else attn_sink._underlying,
        output._underlying,
        softmax_scale,
        page_size,
        head_dim_v,
        None if extra_raw_cache is None else extra_raw_cache._underlying,
        None if extra_indices is None else extra_indices._underlying,
        None if extra_topk_lengths is None else extra_topk_lengths._underlying,
        extra_page_size,
    )
    return output


def deepseek_v4_flashmla_sparse_attention_with_metadata_(
    q: Tensor,
    raw_cache: Tensor,
    indices: Tensor,
    topk_lengths: Tensor,
    attn_sink: Tensor | None,
    output: Tensor,
    tile_scheduler_metadata: Tensor | None,
    num_splits: Tensor | None,
    softmax_scale: float,
    page_size: int = 256,
    head_dim_v: int = 512,
    extra_raw_cache: Tensor | None = None,
    extra_indices: Tensor | None = None,
    extra_topk_lengths: Tensor | None = None,
    extra_page_size: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    output_underlying, tile_scheduler_metadata_underlying, num_splits_underlying = (
        _infinicore.deepseek_v4_flashmla_sparse_attention_with_metadata_(
            q._underlying,
            raw_cache._underlying,
            indices._underlying,
            topk_lengths._underlying,
            None if attn_sink is None else attn_sink._underlying,
            output._underlying,
            None if tile_scheduler_metadata is None else tile_scheduler_metadata._underlying,
            None if num_splits is None else num_splits._underlying,
            softmax_scale,
            page_size,
            head_dim_v,
            None if extra_raw_cache is None else extra_raw_cache._underlying,
            None if extra_indices is None else extra_indices._underlying,
            None if extra_topk_lengths is None else extra_topk_lengths._underlying,
            extra_page_size,
        )
    )
    return Tensor(output_underlying), Tensor(tile_scheduler_metadata_underlying), Tensor(num_splits_underlying)


def deepseek_v4_flashmla_sparse_attention_out_workspace_(
    q: Tensor,
    raw_cache: Tensor,
    indices: Tensor,
    topk_lengths: Tensor,
    attn_sink: Tensor | None,
    output: Tensor,
    lse: Tensor,
    lse_accum: Tensor,
    o_accum: Tensor,
    tile_scheduler_metadata: Tensor,
    num_splits: Tensor,
    softmax_scale: float,
    page_size: int = 256,
    head_dim_v: int = 512,
    extra_raw_cache: Tensor | None = None,
    extra_indices: Tensor | None = None,
    extra_topk_lengths: Tensor | None = None,
    extra_page_size: int = 0,
) -> Tensor:
    _infinicore.deepseek_v4_flashmla_sparse_attention_out_workspace_(
        q._underlying,
        raw_cache._underlying,
        indices._underlying,
        topk_lengths._underlying,
        None if attn_sink is None else attn_sink._underlying,
        output._underlying,
        lse._underlying,
        lse_accum._underlying,
        o_accum._underlying,
        tile_scheduler_metadata._underlying,
        num_splits._underlying,
        softmax_scale,
        page_size,
        head_dim_v,
        None if extra_raw_cache is None else extra_raw_cache._underlying,
        None if extra_indices is None else extra_indices._underlying,
        None if extra_topk_lengths is None else extra_topk_lengths._underlying,
        extra_page_size,
    )
    return output


def deepseek_v4_flashmla_sparse_attention_metadata_(
    tile_scheduler_metadata: Tensor,
    num_splits: Tensor,
    topk_lengths: Tensor,
    topk: int,
    extra_topk_lengths: Tensor | None = None,
    extra_topk: int = -1,
) -> tuple[Tensor, Tensor]:
    _infinicore.deepseek_v4_flashmla_sparse_attention_metadata_(
        tile_scheduler_metadata._underlying,
        num_splits._underlying,
        topk_lengths._underlying,
        topk,
        None if extra_topk_lengths is None else extra_topk_lengths._underlying,
        extra_topk,
    )
    return tile_scheduler_metadata, num_splits

import ctypes

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _ensure_sparse_indexer_vendor_loaded() -> None:
    import deepgemm
    import deepgemm.op
    import lightop
    import lightop.op

    ctypes.CDLL(lightop.op.__file__, mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(deepgemm.op.__file__, mode=ctypes.RTLD_GLOBAL)


def deepseek_v4_sparse_attn_indexer_prefill_(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    cu_seqlen_ks: Tensor,
    cu_seqlen_ke: Tensor,
    logits: Tensor,
    topk_indices: Tensor,
    kv_scale: Tensor | None = None,
    topk_tokens: int = 2048,
    clean_logits: bool = True,
) -> Tensor:
    _ensure_sparse_indexer_vendor_loaded()
    _infinicore.deepseek_v4_sparse_attn_indexer_prefill_(
        q._underlying,
        k._underlying,
        weights._underlying,
        cu_seqlen_ks._underlying,
        cu_seqlen_ke._underlying,
        logits._underlying,
        topk_indices._underlying,
        None if kv_scale is None else kv_scale._underlying,
        topk_tokens,
        clean_logits,
    )
    return topk_indices


def deepseek_v4_sparse_attn_indexer_decode_(
    q: Tensor,
    fused_kv_cache: Tensor,
    weights: Tensor,
    context_lens: Tensor,
    block_table: Tensor,
    schedule_meta: Tensor,
    logits: Tensor,
    topk_indices: Tensor,
    max_context_len: int,
    next_n: int,
    topk_tokens: int = 2048,
    clean_logits: bool = True,
) -> Tensor:
    _ensure_sparse_indexer_vendor_loaded()
    _infinicore.deepseek_v4_sparse_attn_indexer_decode_(
        q._underlying,
        fused_kv_cache._underlying,
        weights._underlying,
        context_lens._underlying,
        block_table._underlying,
        schedule_meta._underlying,
        logits._underlying,
        topk_indices._underlying,
        max_context_len,
        next_n,
        topk_tokens,
        clean_logits,
    )
    return topk_indices



def deepseek_v4_c4_sparse_attn_indexer_(
    q: Tensor,
    indexer_weights: Tensor,
    indexer_kv_cache_raw: Tensor,
    c4_seq_lens: Tensor,
    page_table: Tensor,
    logits: Tensor,
    out_page_indices: Tensor,
    max_c4_seq_len: int,
    page_size: int = 64,
    weight_scale: float = 1.0,
    clean_logits: bool = False,
) -> Tensor:
    _infinicore.deepseek_v4_c4_sparse_attn_indexer_(
        q._underlying,
        indexer_weights._underlying,
        indexer_kv_cache_raw._underlying,
        c4_seq_lens._underlying,
        page_table._underlying,
        logits._underlying,
        out_page_indices._underlying,
        max_c4_seq_len,
        page_size,
        weight_scale,
        clean_logits,
    )
    return out_page_indices



def deepseek_v4_c4_act_quant_fused_scale_kernel_(
    q: Tensor,
    indexer_weights: Tensor,
    q_fp8: Tensor,
    q_scale: Tensor,
    fused_weights: Tensor,
    weight_scale: float = 1.0,
) -> Tensor:
    _infinicore.deepseek_v4_c4_act_quant_fused_scale_kernel_(
        q._underlying,
        indexer_weights._underlying,
        q_fp8._underlying,
        q_scale._underlying,
        fused_weights._underlying,
        weight_scale,
    )
    return q_fp8


def deepseek_v4_c4_paged_mqa_logits_(
    q_fp8: Tensor,
    fused_weights: Tensor,
    indexer_kv_cache_raw: Tensor,
    c4_seq_lens: Tensor,
    page_table: Tensor,
    logits: Tensor,
    max_c4_seq_len: int,
    page_size: int = 64,
    clean_logits: bool = False,
) -> Tensor:
    _ensure_sparse_indexer_vendor_loaded()
    _infinicore.deepseek_v4_c4_paged_mqa_logits_(
        q_fp8._underlying,
        fused_weights._underlying,
        indexer_kv_cache_raw._underlying,
        c4_seq_lens._underlying,
        page_table._underlying,
        logits._underlying,
        max_c4_seq_len,
        page_size,
        clean_logits,
    )
    return logits


def deepseek_v4_topk_transform_512_kernel_(
    scores: Tensor,
    seq_lens: Tensor,
    page_table: Tensor,
    out_page_indices: Tensor,
    page_size: int = 64,
) -> Tensor:
    _infinicore.deepseek_v4_topk_transform_512_kernel_(
        scores._underlying,
        seq_lens._underlying,
        page_table._underlying,
        out_page_indices._underlying,
        page_size,
    )
    return out_page_indices

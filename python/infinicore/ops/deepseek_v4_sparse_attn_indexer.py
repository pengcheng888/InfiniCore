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

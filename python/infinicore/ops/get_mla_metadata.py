from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def get_mla_metadata(
    cache_seqlens,
    num_q_tokens_per_head_k,
    num_heads_k,
    num_heads_q=None,
    is_fp8_kvcache=False,
    topk=None,
    sched_meta=None,
):
    if isinstance(cache_seqlens, Tensor):
        cache_seqlens = cache_seqlens._underlying
    return _infinicore.get_mla_metadata(
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
        num_heads_q,
        is_fp8_kvcache,
        topk,
        sched_meta,
    )

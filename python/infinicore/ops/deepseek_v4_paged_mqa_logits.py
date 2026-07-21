import ctypes

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _ensure_deepgemm_global_loaded() -> None:
    import deepgemm
    import deepgemm.op

    ctypes.CDLL(deepgemm.op.__file__, mode=ctypes.RTLD_GLOBAL)


def deepseek_v4_paged_mqa_logits_metadata_(
    context_lens: Tensor,
    schedule_meta: Tensor,
    block_kv: int,
    num_sms: int,
) -> Tensor:
    _ensure_deepgemm_global_loaded()
    _infinicore.deepseek_v4_paged_mqa_logits_metadata_(
        context_lens._underlying,
        schedule_meta._underlying,
        block_kv,
        num_sms,
    )
    return schedule_meta


def deepseek_v4_paged_mqa_logits_(
    q: Tensor,
    fused_kv_cache: Tensor,
    weights: Tensor,
    context_lens: Tensor,
    block_table: Tensor,
    schedule_meta: Tensor,
    logits: Tensor,
    max_context_len: int,
    clean_logits: bool = True,
) -> Tensor:
    _ensure_deepgemm_global_loaded()
    _infinicore.deepseek_v4_paged_mqa_logits_(
        q._underlying,
        fused_kv_cache._underlying,
        weights._underlying,
        context_lens._underlying,
        block_table._underlying,
        schedule_meta._underlying,
        logits._underlying,
        max_context_len,
        clean_logits,
    )
    return logits

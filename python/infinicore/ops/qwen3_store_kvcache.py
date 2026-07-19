from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def qwen3_store_kvcache_(
    k: Tensor,
    v: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    indices: Tensor,
) -> tuple[Tensor, Tensor]:
    _infinicore.qwen3_store_kvcache_(
        k._underlying,
        v._underlying,
        k_cache._underlying,
        v_cache._underlying,
        indices._underlying,
    )
    return k_cache, v_cache


def qwen3_store_kvcache(*args, **kwargs) -> tuple[Tensor, Tensor]:
    return qwen3_store_kvcache_(*args, **kwargs)


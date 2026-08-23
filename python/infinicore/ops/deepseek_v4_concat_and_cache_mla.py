from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _ensure_vllm_cache_ops_loaded() -> None:
    import vllm._C  # noqa: F401


def deepseek_v4_concat_and_cache_mla_(
    kv_c: Tensor,
    k_pe: Tensor,
    kv_cache: Tensor,
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    scale: Tensor,
) -> Tensor:
    _ensure_vllm_cache_ops_loaded()
    _infinicore.deepseek_v4_concat_and_cache_mla_(
        kv_c._underlying,
        k_pe._underlying,
        kv_cache._underlying,
        slot_mapping._underlying,
        kv_cache_dtype,
        scale._underlying,
    )
    return kv_cache

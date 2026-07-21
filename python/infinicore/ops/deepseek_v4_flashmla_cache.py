from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _ensure_vllm_cache_ops_loaded() -> None:
    import vllm._C  # noqa: F401


def _ensure_sgl_kernel_loaded() -> None:
    import sgl_kernel  # noqa: F401


def deepseek_v4_fused_store_flashmla_cache_(
    kv_c: Tensor,
    k_pe: Tensor,
    kv_cache: Tensor,
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    scale: Tensor,
) -> Tensor:
    _ensure_vllm_cache_ops_loaded()
    _infinicore.deepseek_v4_fused_store_flashmla_cache_(
        kv_c._underlying,
        k_pe._underlying,
        kv_cache._underlying,
        slot_mapping._underlying,
        kv_cache_dtype,
        scale._underlying,
    )
    return kv_cache


def deepseek_v4_flashmla_cache_indexer_(
    req_to_token: Tensor,
    req_pool_indices: Tensor,
    page_kernel_lens: Tensor,
    kv_start_idx: Tensor | None,
    kv_indices: Tensor,
    req_to_token_stride: int,
    kv_indices_stride: int,
    page_size: int = 64,
) -> Tensor:
    _ensure_sgl_kernel_loaded()
    _infinicore.deepseek_v4_flashmla_cache_indexer_(
        req_to_token._underlying,
        req_pool_indices._underlying,
        page_kernel_lens._underlying,
        None if kv_start_idx is None else kv_start_idx._underlying,
        kv_indices._underlying,
        req_to_token_stride,
        kv_indices_stride,
        page_size,
    )
    return kv_indices

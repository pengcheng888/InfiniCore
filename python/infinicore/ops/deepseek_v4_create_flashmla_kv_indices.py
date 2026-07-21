from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _ensure_sgl_kernel_loaded() -> None:
    import sgl_kernel  # noqa: F401


def deepseek_v4_create_flashmla_kv_indices_(
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
    _infinicore.deepseek_v4_create_flashmla_kv_indices_(
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

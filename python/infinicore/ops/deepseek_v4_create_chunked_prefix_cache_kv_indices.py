from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _ensure_sgl_kernel_loaded() -> None:
    import sgl_kernel  # noqa: F401


def deepseek_v4_create_chunked_prefix_cache_kv_indices_(
    req_to_token: Tensor,
    req_pool_indices: Tensor,
    chunk_starts: Tensor,
    chunk_seq_lens: Tensor,
    chunk_cu_seq_lens: Tensor,
    chunk_kv_indices: Tensor,
    col_num: int,
    bs: int,
) -> Tensor:
    _ensure_sgl_kernel_loaded()
    _infinicore.deepseek_v4_create_chunked_prefix_cache_kv_indices_(
        req_to_token._underlying,
        req_pool_indices._underlying,
        chunk_starts._underlying,
        chunk_seq_lens._underlying,
        chunk_cu_seq_lens._underlying,
        chunk_kv_indices._underlying,
        col_num,
        bs,
    )
    return chunk_kv_indices

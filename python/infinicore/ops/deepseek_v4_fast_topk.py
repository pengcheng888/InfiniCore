from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _ensure_sgl_kernel_loaded() -> None:
    import sgl_kernel  # noqa: F401


def deepseek_v4_fast_topk_(
    score: Tensor,
    indices: Tensor,
    lengths: Tensor,
    row_starts: Tensor | None = None,
) -> Tensor:
    _ensure_sgl_kernel_loaded()
    _infinicore.deepseek_v4_fast_topk_(
        score._underlying,
        indices._underlying,
        lengths._underlying,
        None if row_starts is None else row_starts._underlying,
    )
    return indices


def deepseek_v4_fast_topk_transform_fused_(
    score: Tensor,
    lengths: Tensor,
    dst_page_table: Tensor,
    src_page_table: Tensor,
    cu_seqlens_q: Tensor,
    row_starts: Tensor | None = None,
) -> Tensor:
    _ensure_sgl_kernel_loaded()
    _infinicore.deepseek_v4_fast_topk_transform_fused_(
        score._underlying,
        lengths._underlying,
        dst_page_table._underlying,
        src_page_table._underlying,
        cu_seqlens_q._underlying,
        None if row_starts is None else row_starts._underlying,
    )
    return dst_page_table


def deepseek_v4_fast_topk_transform_ragged_fused_(
    score: Tensor,
    lengths: Tensor,
    topk_indices_ragged: Tensor,
    topk_indices_offset: Tensor,
    row_starts: Tensor | None = None,
) -> Tensor:
    _ensure_sgl_kernel_loaded()
    _infinicore.deepseek_v4_fast_topk_transform_ragged_fused_(
        score._underlying,
        lengths._underlying,
        topk_indices_ragged._underlying,
        topk_indices_offset._underlying,
        None if row_starts is None else row_starts._underlying,
    )
    return topk_indices_ragged

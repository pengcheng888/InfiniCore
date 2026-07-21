from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _ensure_sgl_kernel_loaded() -> None:
    import sgl_kernel  # noqa: F401


def deepseek_v4_dcu_alloc_decode_kernel_(
    seq_lens: Tensor,
    last_loc: Tensor,
    free_page: Tensor,
    out_indices: Tensor,
    bs: int,
    page_size: int,
) -> Tensor:
    _ensure_sgl_kernel_loaded()
    _infinicore.deepseek_v4_dcu_alloc_decode_kernel_(
        seq_lens._underlying,
        last_loc._underlying,
        free_page._underlying,
        out_indices._underlying,
        bs,
        page_size,
    )
    return out_indices


def deepseek_v4_dcu_alloc_extend_kernel_(
    pre_lens: Tensor,
    seq_lens: Tensor,
    last_loc: Tensor,
    free_page: Tensor,
    out_indices: Tensor,
    bs: int,
    page_size: int,
) -> Tensor:
    _ensure_sgl_kernel_loaded()
    _infinicore.deepseek_v4_dcu_alloc_extend_kernel_(
        pre_lens._underlying,
        seq_lens._underlying,
        last_loc._underlying,
        free_page._underlying,
        out_indices._underlying,
        bs,
        page_size,
    )
    return out_indices

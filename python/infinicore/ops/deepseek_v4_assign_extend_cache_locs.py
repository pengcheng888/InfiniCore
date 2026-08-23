from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _ensure_sgl_kernel_loaded() -> None:
    import sgl_kernel  # noqa: F401


def deepseek_v4_assign_extend_cache_locs_(
    req_pool_indices: Tensor,
    req_to_token: Tensor,
    start_offset: Tensor,
    end_offset: Tensor,
    out_cache_loc: Tensor,
    pool_len: int,
    bs: int,
) -> Tensor:
    _ensure_sgl_kernel_loaded()
    _infinicore.deepseek_v4_assign_extend_cache_locs_(
        req_pool_indices._underlying,
        req_to_token._underlying,
        start_offset._underlying,
        end_offset._underlying,
        out_cache_loc._underlying,
        pool_len,
        bs,
    )
    return out_cache_loc

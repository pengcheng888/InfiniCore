from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _ensure_sgl_kernel_loaded() -> None:
    import sgl_kernel  # noqa: F401


def deepseek_v4_assign_req_to_token_pool_(
    req_pool_indices: Tensor,
    req_to_token: Tensor,
    allocate_lens: Tensor,
    new_allocate_lens: Tensor,
    out_cache_loc: Tensor,
    shape: int,
    bs: int,
) -> tuple[Tensor, Tensor]:
    _ensure_sgl_kernel_loaded()
    _infinicore.deepseek_v4_assign_req_to_token_pool_(
        req_pool_indices._underlying,
        req_to_token._underlying,
        allocate_lens._underlying,
        new_allocate_lens._underlying,
        out_cache_loc._underlying,
        shape,
        bs,
    )
    return req_to_token, out_cache_loc

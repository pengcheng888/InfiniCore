from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _ensure_sgl_kernel_loaded() -> None:
    import sgl_kernel  # noqa: F401


def deepseek_v4_transfer_kv_per_layer_(
    src_k: Tensor,
    dst_k: Tensor,
    src_v: Tensor,
    dst_v: Tensor,
    src_indices: Tensor,
    dst_indices: Tensor,
    item_size: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16,
) -> tuple[Tensor, Tensor]:
    _ensure_sgl_kernel_loaded()
    _infinicore.deepseek_v4_transfer_kv_per_layer_(
        src_k._underlying,
        dst_k._underlying,
        src_v._underlying,
        dst_v._underlying,
        src_indices._underlying,
        dst_indices._underlying,
        item_size,
        block_quota,
        num_warps_per_block,
    )
    return dst_k, dst_v


def deepseek_v4_transfer_kv_per_layer_pf_lf_(
    src_k: Tensor,
    dst_k: Tensor,
    src_v: Tensor,
    dst_v: Tensor,
    src_indices: Tensor,
    dst_indices: Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16,
) -> tuple[Tensor, Tensor]:
    _ensure_sgl_kernel_loaded()
    _infinicore.deepseek_v4_transfer_kv_per_layer_pf_lf_(
        src_k._underlying,
        dst_k._underlying,
        src_v._underlying,
        dst_v._underlying,
        src_indices._underlying,
        dst_indices._underlying,
        layer_id,
        item_size,
        src_layout_dim,
        block_quota,
        num_warps_per_block,
    )
    return dst_k, dst_v

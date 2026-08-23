from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _ensure_sgl_kernel_loaded() -> None:
    import sgl_kernel  # noqa: F401


def deepseek_v4_moe_align_block_size_(
    topk_ids: Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: Tensor,
    experts_ids: Tensor,
    num_tokens_post_pad: Tensor,
    cumsum_buffer: Tensor,
    pad_sorted_token_ids: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _ensure_sgl_kernel_loaded()
    _infinicore.deepseek_v4_moe_align_block_size_(
        topk_ids._underlying,
        num_experts,
        block_size,
        sorted_token_ids._underlying,
        experts_ids._underlying,
        num_tokens_post_pad._underlying,
        cumsum_buffer._underlying,
        pad_sorted_token_ids,
    )
    return sorted_token_ids, experts_ids, num_tokens_post_pad, cumsum_buffer

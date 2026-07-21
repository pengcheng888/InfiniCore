from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _ensure_sgl_kernel_loaded() -> None:
    import sgl_kernel  # noqa: F401


def deepseek_v4_moe_topk_softmax_(
    topk_weights: Tensor,
    topk_indices: Tensor,
    gating_output: Tensor,
    renormalize: bool = True,
    moe_softcapping: float = 0.0,
    correction_bias: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    _ensure_sgl_kernel_loaded()
    _infinicore.deepseek_v4_moe_topk_softmax_(
        topk_weights._underlying,
        topk_indices._underlying,
        gating_output._underlying,
        renormalize,
        moe_softcapping,
        None if correction_bias is None else correction_bias._underlying,
    )
    return topk_weights, topk_indices

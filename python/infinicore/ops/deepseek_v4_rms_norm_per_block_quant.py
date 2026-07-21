from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _ensure_vllm_ops_loaded() -> None:
    import vllm._C  # noqa: F401


def deepseek_v4_rms_norm_per_block_quant_(
    result: Tensor,
    input: Tensor,
    weight: Tensor,
    scale: Tensor,
    epsilon: float,
    scale_ub: Tensor | None = None,
    residual: Tensor | None = None,
    group_size: int = 128,
    is_scale_transposed: bool = False,
) -> tuple[Tensor, Tensor, Tensor | None]:
    _ensure_vllm_ops_loaded()
    _infinicore.deepseek_v4_rms_norm_per_block_quant_(
        result._underlying,
        input._underlying,
        weight._underlying,
        scale._underlying,
        epsilon,
        None if scale_ub is None else scale_ub._underlying,
        None if residual is None else residual._underlying,
        group_size,
        is_scale_transposed,
    )
    return result, scale, residual

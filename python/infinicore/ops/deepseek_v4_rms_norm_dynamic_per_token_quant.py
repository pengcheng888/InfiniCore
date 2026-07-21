from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _ensure_vllm_ops_loaded() -> None:
    import vllm._C  # noqa: F401


def deepseek_v4_rms_norm_dynamic_per_token_quant_(
    result: Tensor,
    input: Tensor,
    weight: Tensor,
    scale: Tensor,
    epsilon: float,
    scale_ub: Tensor | None = None,
    residual: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    _ensure_vllm_ops_loaded()
    _infinicore.deepseek_v4_rms_norm_dynamic_per_token_quant_(
        result._underlying,
        input._underlying,
        weight._underlying,
        scale._underlying,
        epsilon,
        None if scale_ub is None else scale_ub._underlying,
        None if residual is None else residual._underlying,
    )
    return result, scale, residual

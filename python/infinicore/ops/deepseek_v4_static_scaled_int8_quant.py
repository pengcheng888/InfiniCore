from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _ensure_vllm_ops_loaded() -> None:
    import vllm._C  # noqa: F401


def deepseek_v4_static_scaled_int8_quant_(
    result: Tensor,
    input: Tensor,
    scale: Tensor,
    azp: Tensor | None = None,
) -> Tensor:
    _ensure_vllm_ops_loaded()
    _infinicore.deepseek_v4_static_scaled_int8_quant_(
        result._underlying,
        input._underlying,
        scale._underlying,
        None if azp is None else azp._underlying,
    )
    return result

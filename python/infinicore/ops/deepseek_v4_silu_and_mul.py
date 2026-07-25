from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _ensure_sgl_kernel_loaded() -> None:
    import sgl_kernel  # noqa: F401


def deepseek_v4_silu_and_mul(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    if out is None:
        return Tensor(_infinicore.deepseek_v4_silu_and_mul(input._underlying))
    _infinicore.deepseek_v4_silu_and_mul_(out._underlying, input._underlying)
    return out


def deepseek_v4_silu_and_mul_(out: Tensor, input: Tensor) -> Tensor:
    _infinicore.deepseek_v4_silu_and_mul_(out._underlying, input._underlying)
    return out


def deepseek_v4_silu_and_mul_kernel_(out: Tensor, input: Tensor) -> Tensor:
    _infinicore.deepseek_v4_silu_and_mul_kernel_(out._underlying, input._underlying)
    return out


def deepseek_v4_silu_and_mul_dispatcher_(out: Tensor, input: Tensor) -> Tensor:
    _ensure_sgl_kernel_loaded()
    _infinicore.deepseek_v4_silu_and_mul_dispatcher_(out._underlying, input._underlying)
    return out

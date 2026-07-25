from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def deepseek_v4_linear_bf16_fp32(input: Tensor, weight: Tensor, *, out: Tensor | None = None) -> Tensor:
    if out is None:
        return Tensor(_infinicore.deepseek_v4_linear_bf16_fp32(input._underlying, weight._underlying))
    _infinicore.deepseek_v4_linear_bf16_fp32_(out._underlying, input._underlying, weight._underlying)
    return out



def deepseek_v4_linear_bf16_fp32_(out: Tensor, input: Tensor, weight: Tensor) -> Tensor:
    _infinicore.deepseek_v4_linear_bf16_fp32_(out._underlying, input._underlying, weight._underlying)
    return out


def deepseek_v4_linear_bf16_fp32_naive(input: Tensor, weight: Tensor, *, out: Tensor | None = None) -> Tensor:
    if out is None:
        return Tensor(_infinicore.deepseek_v4_linear_bf16_fp32_naive(input._underlying, weight._underlying))
    _infinicore.deepseek_v4_linear_bf16_fp32_naive_(out._underlying, input._underlying, weight._underlying)
    return out


def deepseek_v4_linear_bf16_fp32_naive_(out: Tensor, input: Tensor, weight: Tensor) -> Tensor:
    _infinicore.deepseek_v4_linear_bf16_fp32_naive_(out._underlying, input._underlying, weight._underlying)
    return out


def deepseek_v4_linear_bf16_fp32_kernel(input: Tensor, weight: Tensor, *, out: Tensor | None = None) -> Tensor:
    if out is None:
        return Tensor(_infinicore.deepseek_v4_linear_bf16_fp32_kernel(input._underlying, weight._underlying))
    _infinicore.deepseek_v4_linear_bf16_fp32_kernel_(out._underlying, input._underlying, weight._underlying)
    return out


def deepseek_v4_linear_bf16_fp32_kernel_(out: Tensor, input: Tensor, weight: Tensor) -> Tensor:
    _infinicore.deepseek_v4_linear_bf16_fp32_kernel_(out._underlying, input._underlying, weight._underlying)
    return out

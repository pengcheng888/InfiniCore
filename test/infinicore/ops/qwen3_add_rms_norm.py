import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase


_CASES = [(1, 128), (7, 128), (16, 1024)]
_DTYPES = [infinicore.float16, infinicore.bfloat16]
_EPS = 1e-6


def parse_test_cases():
    cases = []
    for shape in _CASES:
        for dtype in _DTYPES:
            cases.append(
                TestCase(
                    inputs=[
                        TensorSpec.from_tensor(shape, None, dtype),
                        TensorSpec.from_tensor(shape, None, dtype),
                        TensorSpec.from_tensor((shape[-1],), None, dtype),
                    ],
                    kwargs={"epsilon": _EPS},
                    tolerance={"atol": 2e-2, "rtol": 2e-2},
                    output_count=2,
                    description=f"Qwen3AddRMSNorm_{shape}_{dtype}",
                )
            )
    return cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("Qwen3AddRMSNorm")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, a, b, weight, epsilon=_EPS, **kwargs):
        residual = (a.float() + b.float()).to(a.dtype)
        y = residual.float()
        out = (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + epsilon) * weight.float()).to(a.dtype)
        return out, residual

    def infinicore_operator(self, a, b, weight, epsilon=_EPS, **kwargs):
        out = infinicore.qwen3_add_rms_norm(a, b, weight, epsilon)
        infinicore.sync_stream()
        return out


def main():
    GenericTestRunner(OpTest).run_and_exit()


if __name__ == "__main__":
    main()

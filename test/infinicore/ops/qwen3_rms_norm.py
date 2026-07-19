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
                        TensorSpec.from_tensor((shape[-1],), None, dtype),
                    ],
                    kwargs={"epsilon": _EPS},
                    tolerance={"atol": 2e-2, "rtol": 2e-2},
                    description=f"Qwen3RMSNorm_{shape}_{dtype}",
                )
            )
    return cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("Qwen3RMSNorm")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, x, weight, epsilon=_EPS, **kwargs):
        y = x.float()
        w = weight.float()
        return (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + epsilon) * w).to(x.dtype)

    def infinicore_operator(self, x, weight, epsilon=_EPS, **kwargs):
        out = infinicore.qwen3_rms_norm(x, weight, epsilon)
        infinicore.sync_stream()
        return out


def main():
    GenericTestRunner(OpTest).run_and_exit()


if __name__ == "__main__":
    main()

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase


_CASES = [(1, 128), (7, 128), (16, 512)]
_DTYPES = [infinicore.bfloat16, infinicore.float16]
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
                    description=f"DeepseekV4RMSNorm_{shape}_{dtype}",
                )
            )
    return cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("DeepseekV4RMSNorm")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, x, weight, epsilon=_EPS, **kwargs):
        y = x.float()
        w = weight.float()
        return (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + epsilon) * w).to(x.dtype)

    def infinicore_operator(self, x, weight, epsilon=_EPS, **kwargs):
        out = infinicore.deepseek_v4_rms_norm(x, weight, epsilon)
        infinicore.sync_stream()
        return out


def main():
    GenericTestRunner(OpTest).run_and_exit()


if __name__ == "__main__":
    main()

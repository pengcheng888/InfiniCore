import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import sgl_kernel
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase


_CASES = [(4, 256), (16, 1024)]
_DTYPES = [infinicore.bfloat16, infinicore.float16]


def parse_test_cases():
    cases = []
    for shape in _CASES:
        for dtype in _DTYPES:
            cases.append(
                TestCase(
                    inputs=[TensorSpec.from_tensor(shape, None, dtype)],
                    tolerance={"atol": 2e-2, "rtol": 2e-2},
                    description=f"DeepseekV4SiluAndMul_{shape}_{dtype}",
                )
            )
    return cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("DeepseekV4SiluAndMul")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, x, **kwargs):
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up

    def infinicore_operator(self, x, **kwargs):
        out = infinicore.deepseek_v4_silu_and_mul(x)
        infinicore.sync_stream()
        return out


def main():
    GenericTestRunner(OpTest).run_and_exit()


if __name__ == "__main__":
    main()

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase


_CASES = [(1, 6144), (7, 6144), (2, 4, 6144)]
_DTYPES = [infinicore.float16, infinicore.bfloat16]


def parse_test_cases():
    cases = []
    for shape in _CASES:
        for dtype in _DTYPES:
            cases.append(
                TestCase(
                    inputs=[TensorSpec.from_tensor(shape, None, dtype)],
                    tolerance={"atol": 2e-2, "rtol": 2e-2},
                    description=f"Qwen3SiluAndMul_{shape}_{dtype}",
                )
            )
    return cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("Qwen3SiluAndMul")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, input, **kwargs):
        half = input.shape[-1] // 2
        gate, up = torch.split(input, [half, half], dim=-1)
        return (torch.nn.functional.silu(gate.float()) * up.float()).to(input.dtype)

    def infinicore_operator(self, input, **kwargs):
        out = infinicore.qwen3_silu_and_mul(input)
        infinicore.sync_stream()
        return out


def main():
    GenericTestRunner(OpTest).run_and_exit()


if __name__ == "__main__":
    main()

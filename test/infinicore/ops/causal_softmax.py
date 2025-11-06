import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

# Test cases format: (operation_mode, shape, input_strides, output_strides)
# Causal softmax is a single-input function that applies causal masking before softmax
_TEST_CASES_DATA = [
    # Basic 2D causal softmax
    (TestCase.BOTH, (3, 3), None, None),
    (TestCase.BOTH, (32, 512), None, None),
    # Strided tensors
    (TestCase.BOTH, (32, 512), (1024, 1), (1024, 1)),
    # 3D causal softmax
    (TestCase.BOTH, (32, 5, 5), None, None),
    (TestCase.BOTH, (32, 20, 512), None, None),
    (TestCase.BOTH, (32, 20, 512), (20480, 512, 1), None),
    (TestCase.BOTH, (28, 15, 15), None, None),
]


def parse_test_cases(data):
    """
    Parse causal_softmax test case data according to format:
    (operation_mode, shape, input_strides, output_strides)
    """
    operation_mode = data[0]
    shape = data[1]
    input_strides = data[2] if len(data) > 2 else None
    output_strides = data[3] if len(data) > 3 else None

    # Create input specifications
    inputs = []

    # Tensor input
    if input_strides is not None:
        inputs.append(TensorSpec.from_strided_tensor(shape, input_strides))
    else:
        inputs.append(TensorSpec.from_tensor(shape))

    # Output tensor
    if output_strides is not None:
        output = TensorSpec.from_strided_tensor(shape, output_strides)
    else:
        output = TensorSpec.from_tensor(shape)

    return TestCase(operation_mode, inputs, output)


# Parse test cases
_TEST_CASES = [parse_test_cases(data) for data in _TEST_CASES_DATA]

# Data types
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]

# Tolerance
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 3e-5, "rtol": 1e-5},
    infinicore.bfloat16: {"atol": 5e-3, "rtol": 5e-2},
}


class OpTest(BaseOperatorTest):
    """CausalSoftmax test with simplified test case parsing"""

    def __init__(self):
        super().__init__("CausalSoftmax")

    def get_test_cases(self):
        return _TEST_CASES

    def get_tensor_dtypes(self):
        return _TENSOR_DTYPES

    def get_tolerance_map(self):
        return _TOLERANCE_MAP

    def torch_operator(self, input, out=None, **kwargs):
        # Causal softmax implementation: apply causal mask then softmax
        dtype = input.dtype

        # Create causal mask
        mask = torch.tril(torch.ones_like(input), diagonal=-1).flip(dims=[-2, -1])
        masked = torch.where(mask == 1, -torch.inf, input.to(torch.float32))

        result = torch.nn.functional.softmax(masked, dim=-1, dtype=dtype)

        if out is not None:
            out.copy_(result)
            return out
        return result

    def infinicore_operator(self, input, out=None, **kwargs):
        return infinicore.nn.functional.causal_softmax(input, out=out)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()

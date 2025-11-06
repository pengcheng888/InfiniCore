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
# SiLU is a single-input activation function: output = input * sigmoid(input)
_TEST_CASES_DATA = [
    # Basic 2D SiLU
    (TestCase.BOTH, (2, 4), None, None),
    (TestCase.BOTH, (128, 64), None, None),
    # 3D SiLU
    (TestCase.BOTH, (2, 4, 8), None, None),
    (TestCase.BOTH, (4, 48, 6), None, None),
    # Strided tensors
    (TestCase.BOTH, (1, 2048), (4096, 1), (4096, 1)),
    (TestCase.BOTH, (6, 2560), (2048, 1), (2560, 1)),
    # Mixed cases
    (TestCase.BOTH, (8, 16, 32), None, None),
    # Large tensors
    (TestCase.BOTH, (16, 5632), None, None),
    (TestCase.BOTH, (4, 4, 5632), None, None),
]


def parse_test_cases(data):
    """
    Parse silu test case data according to format:
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
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.bfloat16: {"atol": 5e-3, "rtol": 1e-2},
}


class OpTest(BaseOperatorTest):
    """SiLU test with simplified test case parsing"""

    def __init__(self):
        super().__init__("SiLU")

    def get_test_cases(self):
        return _TEST_CASES

    def get_tensor_dtypes(self):
        return _TENSOR_DTYPES

    def get_tolerance_map(self):
        return _TOLERANCE_MAP

    def torch_operator(self, input, out=None, **kwargs):
        # SiLU implementation: input * sigmoid(input)
        sigmoid_input = torch.sigmoid(input)
        result = input * sigmoid_input
        if out is not None:
            out.copy_(result)
            return out
        return result

    def infinicore_operator(self, input, out=None, **kwargs):
        return infinicore.nn.functional.silu(input, out=out)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()

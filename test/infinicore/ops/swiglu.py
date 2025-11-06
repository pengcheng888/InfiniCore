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

# Test cases format: (operation_mode, shape, a_strides, b_strides, c_strides)
# SwiGLU operates element-wise on two tensors of the same shape
_TEST_CASES_DATA = [
    # Basic 2D SwiGLU
    (TestCase.BOTH, (2, 4), None, None, None),
    (TestCase.BOTH, (128, 64), None, None, None),
    # 3D SwiGLU
    (TestCase.BOTH, (2, 4, 8), None, None, None),
    (TestCase.BOTH, (4, 48, 6), None, None, None),
    # Strided tensors
    (TestCase.BOTH, (1, 2048), (4096, 1), (4096, 1), (4096, 1)),
    (TestCase.BOTH, (6, 2560), (2048, 1), (1, 2048), (2560, 1)),
    # Mixed cases
    (TestCase.BOTH, (8, 16, 32), None, None, None),
    # Large tensors
    (TestCase.BOTH, (16, 5632), None, None, None),
    (TestCase.BOTH, (4, 4, 5632), None, None, None),
]


def parse_test_cases(data):
    """
    Parse swiglu test case data according to format:
    (operation_mode, shape, a_strides, b_strides, c_strides)
    """
    operation_mode = data[0]
    shape = data[1]
    a_strides = data[2] if len(data) > 2 else None
    b_strides = data[3] if len(data) > 3 else None
    c_strides = data[4] if len(data) > 4 else None

    # Create input specifications
    inputs = []

    # Tensor a
    if a_strides is not None:
        inputs.append(TensorSpec.from_strided_tensor(shape, a_strides))
    else:
        inputs.append(TensorSpec.from_tensor(shape))

    # Tensor b
    if b_strides is not None:
        inputs.append(TensorSpec.from_strided_tensor(shape, b_strides))
    else:
        inputs.append(TensorSpec.from_tensor(shape))

    # Output tensor
    if c_strides is not None:
        output = TensorSpec.from_strided_tensor(shape, c_strides)
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
    """SwiGLU test with simplified test case parsing"""

    def __init__(self):
        super().__init__("SwiGLU")

    def get_test_cases(self):
        return _TEST_CASES

    def get_tensor_dtypes(self):
        return _TENSOR_DTYPES

    def get_tolerance_map(self):
        return _TOLERANCE_MAP

    def torch_operator(self, a, b, out=None, **kwargs):
        # SwiGLU implementation: a * b * sigmoid(b)
        sigmoid_b = torch.sigmoid(b)
        result = a * b * sigmoid_b
        if out is not None:
            out.copy_(result)
            return out
        return result

    def infinicore_operator(self, a, b, out=None, **kwargs):
        return infinicore.nn.functional.swiglu(a, b, out=out)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()

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

# Test cases format: (mode, shape, x_strides, y_strides)
_TEST_CASES_DATA = [
    (TestCase.BOTH, (1, 32, 128), None, None),
    # (TestCase.BOTH, (2, 4), (2, 4), (4,), None, None),
    # (TestCase.BOTH, (2, 2, 4), (2, 2, 4), (4,), None, None),
    # (TestCase.BOTH, (2, 2, 4), (2, 2, 4), (4,), (12, 8, 1), (12, 8, 1)),
    # (TestCase.BOTH, (16, 2048), (16, 2048), (2048,), None, None),
    # (TestCase.BOTH, (16, 2048), (16, 2048), (2048,), (4096, 1), (4096, 1)),
]



def parse_rope_norm_test_case(data):
    """
    Parse RoPE test case data according to format:
    (shape, x_strides, y_strides)
    """
    operation_mode = data[0]  
    shape = data[1]  # shape
    x_strides = data[2]  # x_strides
    y_strides = data[3]  # y_strides

    # Create input specifications
    inputs = []

    # Input tensor x
    if x_strides is not None:
        inputs.append(TensorSpec.from_strided_tensor(shape, x_strides))
    else:
        inputs.append(TensorSpec.from_tensor(shape))

    # Weight tensor (1D, always contiguous)
    pos = shape[0]
    dim = shape[2]

    table_shape = (pos, dim//2)
 
    inputs.append(TensorSpec.from_tensor(table_shape)) # sin_table
    inputs.append(TensorSpec.from_tensor(table_shape)) # cos_table


    # Output tensor
    if y_strides is not None:
        output = TensorSpec.from_strided_tensor(shape, y_strides)
    else:
        output = TensorSpec.from_tensor(shape)

    return TestCase(operation_mode, inputs, output)


# Parse test cases
_TEST_CASES = [parse_rope_norm_test_case(data) for data in _TEST_CASES_DATA]

# Data types for individual tensors
_INPUT_DTYPES = [infinicore.float16, 
                #  infinicore.bfloat16
                 ]

# Generate all dtype combinations
_DTYPE_COMBINATIONS = []
for input_dtype in _INPUT_DTYPES:
    _DTYPE_COMBINATIONS.append(
        {
            "input_0": input_dtype,  # x tensor
            "input_1": input_dtype,  # weight tensor
            "input_2": input_dtype,  # weight tensor
            "output": input_dtype,  # output tensor (same as input)
        }
    )

# Base data types
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16]

# Tolerance
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 2e-3, "rtol": 2e-3},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
}


from enum import Enum, auto
class Algorithm(Enum):
    GPT_J = 0
    GPT_NEOX = 1


def rotary_embedding(ans, t, sin, cos, algo):
    def _torch_rope(sin, cos, t1, t2):
        cos = cos.unsqueeze(1)  # [seq_len, 1, dh // 2]
        sin = sin.unsqueeze(1)  # [seq_len, 1, dh // 2]
        # if device == InfiniDeviceEnum.CPU:
        #     t1, t2, cos, sin = t1.float(), t2.float(), cos.float(), sin.float()
     
        t_out_1 = t1 * cos - t2 * sin
        t_out_2 = t2 * cos + t1 * sin

        return t_out_1, t_out_2

    dh = t.shape[-1]
    dt = t.dtype
    assert dh % 2 == 0, "Embedding dimension must be even."

    if algo == Algorithm.GPT_J:
        t_even = t[..., 0::2]  # [seq_len, n_head, dh // 2]
        t_odd = t[..., 1::2]  # [seq_len, n_head, dh // 2]

        t_out_even, t_out_odd = _torch_rope(sin, cos, t_even, t_odd)

        ans[..., 0::2] = t_out_even.to(dt)
        ans[..., 1::2] = t_out_odd.to(dt)
    else:
        half_dim = dh // 2
        t_first = t[..., :half_dim]
        t_second = t[..., half_dim:]

        t_out_first, t_out_second = _torch_rope(sin, cos, t_first, t_second)

        ans[..., :half_dim] = t_out_first.to(dt)
        ans[..., half_dim:] = t_out_second.to(dt)


class OpTest(BaseOperatorTest):
    """RoPE test with simplified test case parsing"""

    def __init__(self):
        super().__init__("rope")

    def get_test_cases(self):
        return _TEST_CASES

    def get_tensor_dtypes(self):
        return _TENSOR_DTYPES 

    def get_tolerance_map(self):
        return _TOLERANCE_MAP

    def get_dtype_combinations(self):
        return _DTYPE_COMBINATIONS

    def torch_operator(self, x,  sin_table, cos_table, out=None, **kwargs):
        if out is  None:
            out =  x.clone()
        
        rotary_embedding(out, x, sin_table, cos_table, Algorithm.GPT_J)
        return out

    def infinicore_operator(self, x, sin_table,cos_table, out=None, **kwargs):
  
        from framework import infinicore_tensor_from_torch

        pos = torch.arange(0, x.shape[0],dtype=torch.int32)
        pos_ids = infinicore_tensor_from_torch(pos)
        
        return infinicore.rope(x, pos_ids, sin_table, cos_table, out=out)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()

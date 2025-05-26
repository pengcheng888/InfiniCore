import torch
import ctypes
from ctypes import POINTER, Structure, c_int32, c_void_p, c_uint64
from libinfiniop import (
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    open_lib,
    to_tensor,
    get_test_devices,
    check_error,
    rearrange_if_needed,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    create_workspace,
)
from enum import Enum, auto

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # shape, a_stride, b_stride, c_stride
    ((13, 4), None, None, None),
    ((13, 4), (10, 1), (10, 1), (10, 1)),
    ((13, 4), (0, 1), None, None),
    ((13, 4, 4), None, None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
    ((13, 4, 4), (4, 0, 1), (0, 4, 1), None),
    ((16, 5632), None, None, None),
    ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
    ((4, 4, 5632), None, None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
]

_TEST_CASES_ = [
    # shape, a_stride, b_stride, c_stride
    ((1e1, 10, 10), (100, 10, 1), (100, 10, 1), (100, 10, 1)),
    ((1e2, 10, 10), (100, 10, 1), (100, 10, 1), (100, 10, 1)),
    ((1e3, 10, 10), (100, 10, 1), (100, 10, 1), (100, 10, 1)),
    ((1e4, 10, 10), (100, 10, 1), (100, 10, 1), (100, 10, 1)),
    ((1e5, 10, 10), (100, 10, 1), (100, 10, 1), (100, 10, 1)),
    # ((1e6, 10, 10), (100,10, 1), (100,10, 1), (100,10, 1)),
    # ((2e6, 10, 10), (100,10, 1), (100,10, 1), (100,10, 1)),
]

# Data types used for testing
_TENSOR_DTYPES = [torch.float16, torch.float32]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_A = auto()
    INPLACE_B = auto()


def _rearrange(tensor, strides):
    # print('====> ')
    # print(tensor)

    if strides and 0 in strides:  # stride中有0，表示这广播吗
        # print(tensor[0, 0, 0])
        # print(tensor[0, 0, 1])
        # print(tensor[0, 0, 2])

        # 这句话的意思是将tensor中默认的strides，修改为strides，其他的东西保持不变
        # 此时再通过 tensor[0, 1, 0]索引访问，数据就不同了
        tensor.set_(tensor.untyped_storage(), 0, tensor.shape, strides)  # 这行代码会修改什么呢

        # print(tensor[0, 1, 0])
        # print(tensor[0, 1, 1])
        # print(tensor[0, 1, 2])
        return tensor
    else:
        return rearrange_if_needed(tensor, strides)


def process_tensors(c, c_strides, a, a_stride, b, b_stride, inplace):
    """
    rearrange the tensors if needed and apply the inplace config.
    if inplace is true and the output (i.e., c) is placed to the broadcasted input,
    the inplace config is ignored and out-of-place is used
    """
    original_c_strides = c_strides if c_strides else c.stride()

    a, b, c = [
        _rearrange(tensor, stride) for tensor, stride in zip([a, b, c],
                                                             [a_stride, b_stride, c_strides])
    ]
    c = (
        c
        if inplace == Inplace.OUT_OF_PLACE
        else (a if inplace == Inplace.INPLACE_A else b)
    )
    # if inplace is true and c has broadcasted config, reset it to the original unbroadcasted strides
    if 0 in c.stride():
        c.set_(c.untyped_storage(), 0, c.shape, original_c_strides)

    return a, b, c


def func():
    # 设置随机种子
    torch.manual_seed(1234)  # 只设置主设备的随机种子

    shape = (2, 3, 4)
    a_stride = None
    dtype = torch.float16

    data = torch.rand(shape, dtype=dtype).to("cuda")

    exit()

    print(data.untyped_storage())
    print(len(data.untyped_storage()))

    print(type(data.storage()))

    print('=========>')

    print(data.stride())
    print(data[0, 0, 0], data[0, 0, 1])

    data.set_(data.untyped_storage(), 0, data.shape, (12, 100, 1))  # 这行代码会修改什么呢

    print(data[0, 1, 0], data[0, 1, 1])
    print(data.stride())


def func2():
    torch.manual_seed(1234)  # 只设置主设备的随机种子

    t = torch.arange(6).reshape(2, 3)
    print(t)
    print("shape:", t.shape, "\t stride:", t.stride())

    t2 = t.transpose(0, 1)
    print(t2)
    print("shape:", t2.shape, "\t stride:", t2.stride())

    print(t.data_ptr() == t2.data_ptr())  #

    print(t.is_contiguous(), t2.is_contiguous())


if __name__ == "__main__":
    # func2()
    # exit()
    # shape, a_stride, b_stride, c_stride

    shape = (2, 4, 10)
    # a_stride = (2, 4, 1)
    # b_stride = (2, 4, 1)
    # c_stride = (2, 4, 1)

    a_stride = (40, 5, 1)
    b_stride = (40, 5, 1)
    c_stride = (40, 10, 1)

    dtype = torch.float16

    torch_device = "cuda"

    a = torch.rand(shape, dtype=dtype).to(torch_device)
    b = torch.rand(shape, dtype=dtype).to(torch_device)
    c = torch.rand(shape, dtype=dtype).to(torch_device)

    a, b, c = process_tensors(c, c_stride,
                              a, a_stride,
                              b, b_stride,
                              Inplace.OUT_OF_PLACE)

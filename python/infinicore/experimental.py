import torch

import infinicore
import numpy as np


def to_torch_dtype(infini_dtype):
    """Convert infinicore data type to PyTorch data type"""
    if infini_dtype == infinicore.float16:
        return torch.float16
    elif infini_dtype == infinicore.float32:
        return torch.float32
    elif infini_dtype == infinicore.float64:
        return torch.float64
    elif infini_dtype == infinicore.bfloat16:
        return torch.bfloat16
    elif infini_dtype == infinicore.int8:
        return torch.int8
    elif infini_dtype == infinicore.int16:
        return torch.int16
    elif infini_dtype == infinicore.int32:
        return torch.int32
    elif infini_dtype == infinicore.int64:
        return torch.int64
    elif infini_dtype == infinicore.uint8:
        return torch.uint8
    else:
        raise ValueError(f"Unsupported infinicore dtype: {infini_dtype}")


def to_infinicore_dtype(torch_dtype):
    """Convert PyTorch data type to infinicore data type"""
    if torch_dtype == torch.float32:
        return infinicore.float32
    elif torch_dtype == torch.float64:
        return infinicore.float64
    elif torch_dtype == torch.float16:
        return infinicore.float16
    elif torch_dtype == torch.bfloat16:
        return infinicore.bfloat16
    elif torch_dtype == torch.int8:
        return infinicore.int8
    elif torch_dtype == torch.int16:
        return infinicore.int16
    elif torch_dtype == torch.int32:
        return infinicore.int32
    elif torch_dtype == torch.int64:
        return infinicore.int64
    elif torch_dtype == torch.uint8:
        return infinicore.uint8
    else:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


def torch_2_infini_tensor_ref(torch_tensor: torch.Tensor):
    infini_device = infinicore.device(torch_tensor.device.type, 0)
    ref = infinicore.from_blob(
        torch_tensor.data_ptr(),
        list(torch_tensor.shape),
        dtype=to_infinicore_dtype(torch_tensor.dtype),
        device=infini_device,
    )
    return ref


def convert_torch_to_infini_tensor(torch_tensor):
    infini_device = infinicore.device(torch_tensor.device.type, 0)
    if torch_tensor.is_contiguous():
        ref = infinicore.from_blob(
            torch_tensor.data_ptr(),
            list(torch_tensor.shape),
            dtype=to_infinicore_dtype(torch_tensor.dtype),
            device=infini_device,
        )
    else:
        ref = infinicore.strided_from_blob(
            torch_tensor.data_ptr(),
            list(torch_tensor.shape),
            list(torch_tensor.stride()),
            dtype=to_infinicore_dtype(torch_tensor.dtype),
            device=infini_device,
        )

    infini_tensor = infinicore.empty(
        list(torch_tensor.shape),
        dtype=to_infinicore_dtype(torch_tensor.dtype),
        device=infini_device,
    )
    infini_tensor.copy_(ref)

    return infini_tensor


def convert_np_to_infini_tensor(np_tensor):
    infini_device = infinicore.device("cpu", 0)
    ptr = np_tensor.__array_interface__["data"][0]
    ref = infinicore.from_blob(
        ptr,
        list(np_tensor.shape),
        dtype=infinicore.int32,
        device=infini_device,
    )

    infini_tensor = infinicore.empty(
        list(np_tensor.shape), dtype=infinicore.int32, device=infini_device
    )
    infini_tensor.copy_(ref)

    return infini_tensor


def convert_infini_to_torch_tensor(infini_result, torch_reference=None):
    """
    Convert infinicore tensor to PyTorch tensor for comparison

    Args:
        infini_result: infinicore tensor result
        torch_reference: PyTorch tensor reference (for shape and device)

    Returns:
        torch.Tensor: PyTorch tensor with infinicore data
    """

    torch_tensor = torch.zeros(
        infini_result.shape,
        dtype=to_torch_dtype(infini_result.dtype),
        device=infini_result.device.type,
    )

    infini_device = infinicore.device(torch_tensor.device.type, 0)
    temp_tensor = infinicore.from_blob(
        torch_tensor.data_ptr(),
        list(torch_tensor.shape),
        dtype=to_infinicore_dtype(torch_tensor.dtype),
        device=infini_device,
    )
    temp_tensor.copy_(infini_result)
    return torch_tensor


# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
def py_to_ctype(py_dtype):
    """Convert PyTorch data type to infinicore data type"""
    from ctypes import POINTER, addressof, byref, c_float, c_int, c_uint, c_void_p

    if py_dtype == int:
        return c_int
    elif py_dtype == float:
        return c_float
    else:
        raise ValueError(f"Unsupported py_dtype: {py_dtype}")


def py_to_infinicore_dtype(py_dtype):
    """Convert PyTorch data type to infinicore data type"""
    from ctypes import POINTER, addressof, byref, c_float, c_int, c_uint, c_void_p

    import infinicore

    if py_dtype == int:
        return infinicore.int32
    elif py_dtype == float:
        return infinicore.float32
    else:
        raise ValueError(f"Unsupported py_dtype: {py_dtype}")


def convert_list_to_infini_tensor(
    data_list: list,
    shape=None,
    infini_device=infinicore.device("cpu", 0),
    infini_dtype=infinicore.int32,
):
    from ctypes import POINTER, addressof, byref, c_float, c_int, c_uint, c_void_p

    num_count = len(data_list)
    address = (py_to_ctype(type(data_list[0])) * num_count)(*data_list)
    ptr = addressof(address)

    if shape is None:
        shape = [num_count]

    infini_type = py_to_infinicore_dtype(type(data_list[0]))

    ref = infinicore.from_blob(
        ptr,
        shape,
        dtype=infini_dtype,
        device=infinicore.device("cpu", 0),
    )

    infini_tensor = infinicore.empty(
        shape, dtype=infini_type, device=infinicore.device("cpu", 0)
    )
    infini_tensor.copy_(ref)

    if infini_device.type != "cpu":
        return infini_tensor.to(infini_device)

    return infini_tensor


# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #


def get_index_value(infini_tensor: infinicore.Tensor, index: list):
    torch_tensor = infinicore.convert_infini_to_torch_tensor(infini_tensor)
    value = torch_tensor[index].cpu().item()
    return value


# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
def infini__str__(self):
    self_torch = infinicore.convert_infini_to_torch_tensor(self)
    return "infinicore::\n" + self_torch.__str__()


infinicore.Tensor.__str__ = infini__str__


import torch  # noqa: E402


def to_infini(self) -> infinicore.Tensor:
    """
    将torch.Tensor的数据转为 infinicore.Tensor
    """
    return infinicore.from_torch(self)


torch.Tensor.to_infini = to_infini

torch.Tensor._infini_ret = None


def to_torch(self) -> torch.Tensor:
    # torch_tensor = torch.zeros(
    #     infini_result.shape,
    #     dtype=to_torch_dtype(infini_result.dtype),
    #     device=infini_result.device.type,
    # )

    # infini_device = infinicore.device(torch_tensor.device.type, 0)
    # temp_tensor = infinicore.from_blob(
    #     torch_tensor.data_ptr(),
    #     list(torch_tensor.shape),
    #     dtype=to_infinicore_dtype(torch_tensor.dtype),
    #     device=infini_device,
    # )
    # temp_tensor.copy_(infini_result)

    return infinicore.convert_infini_to_torch_tensor(self)


infinicore.Tensor.to_torch = to_torch


def from_infini(data: infinicore.Tensor) -> torch.Tensor:
    """
    将 infinicore.Tensor 的数据 转为 torch.Tensor
    """
    return infinicore.convert_infini_to_torch_tensor(data)


torch.from_infini = from_infini

# -------------------------------------------------------------------
# -------------------------------------------------------------------


# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
def infini_to_ctype_dtype(infini_dtype):
    """Convert PyTorch data type to infinicore data type"""
    import ctypes
    from ctypes import POINTER, addressof, byref, c_float, c_int, c_uint, c_void_p

    if infini_dtype == infinicore.int32:
        return ctypes.c_int32
    elif infini_dtype == infinicore.float32:
        return ctypes.c_float
    else:
        raise ValueError(f"Unsupported py_dtype: {infini_dtype}")


def infini_to_numpy(infini_tensor: infinicore.Tensor):
    if infini_tensor.device.type != "cpu":
        infini_tensor_cpu = infini_tensor.to(infinicore.device("cpu", 0))
    else:
        infini_tensor_cpu = infini_tensor

    # 获取数据指针和形状信息
    data_ptr = infini_tensor_cpu.data_ptr()
    num_elements = infini_tensor_cpu.numel()
    original_shape = infini_tensor_cpu.shape

    # 创建1D NumPy数组（共享内存）
    ArrayType = infinicore.infini_to_ctype_dtype(infini_tensor_cpu.dtype) * num_elements
    array = ArrayType.from_address(data_ptr)
    np_flat = np.ctypeslib.as_array(array)

    # 重塑为原始形状
    np_array = np_flat.reshape(original_shape)

    return np.copy(np_array)


infinicore.Tensor.to_numpy = infini_to_numpy
import torch
import infinicore


def to_torch_dtype(infini_dtype):
    """Convert infinicore data type to PyTorch data type"""
    if infini_dtype == infinicore.float16:
        return torch.float16
    elif infini_dtype == infinicore.float32:
        return torch.float32
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

    infini_tensor = infinicore.empty(torch_tensor.shape,
                                     dtype=to_infinicore_dtype(torch_tensor.dtype),
                                     device=infini_device
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


def infini__str__(self):
    self_torch = infinicore.convert_infini_to_torch_tensor(self)
    return "infinicore::\n" + self_torch.__str__()


infinicore.Tensor.__str__ = infini__str__


def infini__mul__(self, other):
    self_torch = infinicore.convert_infini_to_torch_tensor(self)
    other_torch = infinicore.convert_infini_to_torch_tensor(other)

    # 先暂时使用 pytorch 实现逐元素相乘
    output_torch = self_torch * other_torch
    # 
    output_infinicore = infinicore.convert_torch_to_infini_tensor(output_torch)
    return output_infinicore


infinicore.Tensor.__mul__ = infini__mul__


def infini__add__(self, other):
    return infinicore.add(self, other)


infinicore.Tensor.__add__ = infini__add__

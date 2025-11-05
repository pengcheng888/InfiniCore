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
    

def is_integer_dtype(dtype):
    """Check if dtype is integer type"""
    return dtype in [
        infinicore.int8,
        infinicore.int16,
        infinicore.int32,
        infinicore.int64,
        infinicore.uint8,
    ]


def is_float_dtype(dtype):
    """Check if dtype is floating point type"""
    return dtype in [infinicore.float16, infinicore.float32, infinicore.bfloat16]


def convert_torch_to_infini_tensor(torch_tensor):
    infini_device = infinicore.device(torch_tensor.device.type, 0)
    if torch_tensor.is_contiguous():
        ref =  infinicore.from_blob(
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


def convert_infini_to_torch_tensor(infini_result, torch_reference = None):
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


def rearrange_tensor(tensor, new_strides):
    """
    Given a PyTorch tensor and a list of new strides, return a new PyTorch tensor with the given strides.
    """
    import torch

    shape = tensor.shape

    new_size = [0] * len(shape)
    left = 0
    right = 0
    for i in range(len(shape)):
        if new_strides[i] > 0:
            new_size[i] = (shape[i] - 1) * new_strides[i] + 1
            right += new_strides[i] * (shape[i] - 1)
        else:  # TODO: Support negative strides in the future
            # new_size[i] = (shape[i] - 1) * (-new_strides[i]) + 1
            # left += new_strides[i] * (shape[i] - 1)
            raise ValueError("Negative strides are not supported yet")

    # Create a new tensor with zeros
    new_tensor = torch.zeros(
        (right - left + 1,), dtype=tensor.dtype, device=tensor.device
    )

    # Generate indices for original tensor based on original strides
    indices = [torch.arange(s) for s in shape]
    mesh = torch.meshgrid(*indices, indexing="ij")

    # Flatten indices for linear indexing
    linear_indices = [m.flatten() for m in mesh]

    # Calculate new positions based on new strides
    new_positions = sum(
        linear_indices[i] * new_strides[i] for i in range(len(shape))
    ).to(tensor.device)
    offset = -left
    new_positions += offset

    # Copy the original data to the new tensor
    new_tensor.view(-1).index_add_(0, new_positions, tensor.view(-1))
    new_tensor.set_(new_tensor.untyped_storage(), offset, shape, tuple(new_strides))

    return new_tensor

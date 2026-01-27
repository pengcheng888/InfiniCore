from ctypes import c_uint64
import ctypes
import sys
import os
import torch
import math
from torch.nn import functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================



# Test cases: (input_shape, kernel_size_h, kernel_size_w, stride_h, stride_w, 
#              padding_h, padding_w, dilation_h, dilation_w, ceil_mode)
_TEST_CASES = [
    # Basic cases
    ((1, 512, 7, 7), 7, 7, 1, 1, 0, 0, 1, 1, 0),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def avg_pool2d_ref(input, kernel_size_h, kernel_size_w, stride_h, stride_w, 
                   padding_h, padding_w, dilation_h, dilation_w, ceil_mode):
    """
    Reference implementation of avg_pool2d with dilation support.
    For dilation=1, uses PyTorch's built-in avg_pool2d.
    For dilation>1, implements dilated average pooling manually.
    """
    if dilation_h == 1 and dilation_w == 1:
        # Use PyTorch's built-in function for non-dilated case
        return F.avg_pool2d(
            input,
            kernel_size=(kernel_size_h, kernel_size_w),
            stride=(stride_h, stride_w),
            padding=(padding_h, padding_w),
            ceil_mode=bool(ceil_mode),
        )
    else:
        # Manual implementation for dilated case
        # Create a kernel with dilation
        N, C, H, W = input.shape
        
        # Calculate effective kernel size with dilation
        eff_kernel_h = (kernel_size_h - 1) * dilation_h + 1
        eff_kernel_w = (kernel_size_w - 1) * dilation_w + 1
        
        # Calculate output size
        if ceil_mode:
            out_h = math.ceil((H + 2 * padding_h - eff_kernel_h) / stride_h + 1)
            out_w = math.ceil((W + 2 * padding_w - eff_kernel_w) / stride_w + 1)
        else:
            out_h = math.floor((H + 2 * padding_h - eff_kernel_h) / stride_h + 1)
            out_w = math.floor((W + 2 * padding_w - eff_kernel_w) / stride_w + 1)
        
        # Pad input
        if padding_h > 0 or padding_w > 0:
            input_padded = F.pad(input, (padding_w, padding_w, padding_h, padding_h), mode='constant', value=0)
        else:
            input_padded = input
        
        # Initialize output
        output = torch.zeros(N, C, out_h, out_w, dtype=input.dtype, device=input.device)
        
        # Perform dilated average pooling
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride_h
                w_start = j * stride_w
                
                # Collect values for dilated sampling
                sum_val = torch.zeros(N, C, dtype=input.dtype, device=input.device)
                count = 0
                for kh in range(kernel_size_h):
                    for kw in range(kernel_size_w):
                        h_idx = h_start + kh * dilation_h
                        w_idx = w_start + kw * dilation_w
                        if h_idx < input_padded.shape[2] and w_idx < input_padded.shape[3]:
                            sum_val += input_padded[:, :, h_idx, w_idx]
                            count += 1
                
                if count > 0:
                    output[:, :, i, j] = sum_val / count
                else:
                    output[:, :, i, j] = 0
        
        return output


def infer_output_shape(input_shape, kernel_size_h, kernel_size_w, stride_h, stride_w,
                       padding_h, padding_w, dilation_h, dilation_w, ceil_mode):
    """Infer output shape for avg_pool2d"""
    N, C, H, W = input_shape
    eff_kernel_h = (kernel_size_h - 1) * dilation_h + 1
    eff_kernel_w = (kernel_size_w - 1) * dilation_w + 1
    
    if ceil_mode:
        out_h = math.ceil((H + 2 * padding_h - eff_kernel_h) / stride_h + 1)
        out_w = math.ceil((W + 2 * padding_w - eff_kernel_w) / stride_w + 1)
    else:
        out_h = math.floor((H + 2 * padding_h - eff_kernel_h) / stride_h + 1)
        out_w = math.floor((W + 2 * padding_w - eff_kernel_w) / stride_w + 1)
    
    return (N, C, out_h, out_w)


def test(
    handle,
    device,
    input_shape,
    kernel_size_h,
    kernel_size_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    ceil_mode,
    tensor_dtype=InfiniDtype.F16,
    sync=None,
):
    # Create input tensor
    x = TestTensor(input_shape, None, dt=tensor_dtype, device=device, scale=0.01)
    
    # Infer output shape
    output_shape = infer_output_shape(
        input_shape, kernel_size_h, kernel_size_w, stride_h, stride_w,
        padding_h, padding_w, dilation_h, dilation_w, ceil_mode
    )
    y = TestTensor(output_shape, None, dt=tensor_dtype, device=device)
    
    print(
        f"Testing AvgPool2d on {InfiniDeviceNames[device]} with "
        f"input_shape: {input_shape}, kernel_size: ({kernel_size_h}, {kernel_size_w}), "
        f"stride: ({stride_h}, {stride_w}), padding: ({padding_h}, {padding_w}), "
        f"dilation: ({dilation_h}, {dilation_w}), ceil_mode: {ceil_mode}, "
        f"dtype: {InfiniDtypeNames[tensor_dtype]}"
    )
    
    # Run reference implementation
    y_ref = avg_pool2d_ref(
        x.torch_tensor(),
        kernel_size_h, kernel_size_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        ceil_mode,
    )
    
    if sync is not None:
        sync()
    
    # Create operator descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateAvgPool2dDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            x.descriptor,
            ctypes.c_int32(kernel_size_h),
            ctypes.c_int32(kernel_size_w),
            ctypes.c_int32(stride_h),
            ctypes.c_int32(stride_w),
            ctypes.c_int32(padding_h),
            ctypes.c_int32(padding_w),
            ctypes.c_int32(dilation_h),
            ctypes.c_int32(dilation_w),
            ctypes.c_int32(ceil_mode),
        )
    )
    
    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x.destroy_desc()
    y.destroy_desc()
    
    # Get workspace size
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetAvgPool2dWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, y.device)
    
    def lib_avg_pool2d():
        check_error(
            LIBINFINIOP.infiniopAvgPool2d(
                descriptor,
                workspace.data(),
                workspace_size.value,
                y.data(),
                x.data(),
                None,
            )
        )
    
    lib_avg_pool2d()
    
    if sync is not None:
        sync()
    
    # Verify correctness
    atol, rtol = get_tolerance(_TOLERANCE_MAP, tensor_dtype)
    if DEBUG:
        debug(y.actual_tensor(), y_ref, atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), y_ref, atol=atol, rtol=rtol)
    
    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation(
            "PyTorch",
            lambda: avg_pool2d_ref(
                x.torch_tensor(), kernel_size_h, kernel_size_w,
                stride_h, stride_w, padding_h, padding_w,
                dilation_h, dilation_w, ceil_mode
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation("    lib", lib_avg_pool2d, device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    
    check_error(LIBINFINIOP.infiniopDestroyAvgPool2dDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    
    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations
    
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
    
    print("\033[92mTest passed!\033[0m")
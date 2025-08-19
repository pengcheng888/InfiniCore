import torch
import ctypes
from ctypes import c_uint64

import torch.nn.functional as F

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
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # values_shape, indices_shape, x_shape, values_stride, indices_stride, x_stride
    ((10, 8), (10, 8), (10, 80), None, None, None),
]

# w (weight) types
# Note: 'None' means the same as input dtype
_X_DTYPES = [InfiniDtype.BF16]
# x types used for testing
_VALUE_DTYPES = [InfiniDtype.BF16]

# Form the test cases by appending each element of _X_DTYPES to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (x_dtype,) for test_case in _TEST_CASES_ for x_dtype in _X_DTYPES
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 2e-3, "rtol": 2e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def topksoftmax_torch(router_logits, top_k, norm_topk_prob=False):
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    if norm_topk_prob:  # only diff with mixtral sparse moe block!
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

    return routing_weights, selected_experts


def tensorInfo(name, data):
    print(name, data.is_contiguous(), data.device, data.dtype, data.shape, data.stride(), data.data_ptr(), hex(data.data_ptr()))


def test(
        handle,
        device,
        values_shape,
        indices_shape,
        x_shape,
        values_stride,
        indices_stride,
        x_stride,
        x_dtype=InfiniDtype.F32,
        dtype=InfiniDtype.F16,
        sync=None,
):
    print(
        f"Testing topksoftmax on {InfiniDeviceNames[device]} with values_shape:{values_shape} indices_shape:{indices_shape} x_shape:{x_shape}"
        f" y_stride:{values_stride} x_stride:{indices_stride} w_dtype:{InfiniDtypeNames[x_dtype]} dtype:{InfiniDtypeNames[dtype]}"
    )

    values = TestTensor(values_shape, values_stride, x_dtype, device, mode="zeros")
    indices = TestTensor(indices_shape, indices_stride, InfiniDtype.I32, device, mode="zeros")
    x = TestTensor(x_shape, x_stride, x_dtype, device, mode="random")

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()

    check_error(
        LIBINFINIOP.infiniopCreateTopksoftmaxDescriptor(
            handle,
            ctypes.byref(descriptor),
            values.descriptor,
            indices.descriptor,
            x.descriptor
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [values, indices, x]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetTopksoftmaxWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_topksoftmax():
        check_error(
            LIBINFINIOP.infiniopTopksoftmax(
                descriptor,
                workspace.data(),
                workspace_size.value,
                values.data(),
                indices.data(),
                x.data(),
                None,
            )
        )

    # print("actual_tensor:", x.actual_tensor())

    ans = topksoftmax_torch(x.actual_tensor().clone(), 8, norm_topk_prob=False)
    print("ans: ", ans)

    lib_topksoftmax()

    # tensorInfo("values:: ", values.actual_tensor())
    # tensorInfo("indices:: ", indices.actual_tensor())
    # tensorInfo("x:  ", x.actual_tensor())

    # print("actual_tensor:", x.actual_tensor())
    print(values.actual_tensor())
    print(indices.actual_tensor())
    exit()

    def tensorInfo(data):
        print("data:  ", data.is_contiguous(), data.device, data.dtype, data.shape, data.stride(), data.data_ptr(), hex(data.data_ptr()))

    tensorInfo(y.actual_tensor())
    tensorInfo(x.actual_tensor())
    tensorInfo(w.actual_tensor())
    exit()
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: topksoftmax_torch(y.torch_tensor(), x.torch_tensor(), w.torch_tensor(), eps), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_topksoftmax(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyTopksoftmaxDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    args.nvidia = True

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _VALUE_DTYPES)

    print("\033[92mTest passed!\033[0m")

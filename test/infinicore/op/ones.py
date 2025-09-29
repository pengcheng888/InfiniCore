import torch
import infinicore
import sys
import os

# Framework path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework import (
    TestConfig,
    TestRunner,
    TestCase,
    create_infinicore_tensor,
    compare_results,
    get_args,
    get_test_devices,
    profile_operation,
    to_torch_dtype,
    InfiniDeviceNames,
    torch_device_map,
)

# ==============================================================================
# Test Setup
# ==============================================================================

# Test cases
_TEST_CASES = [
    # (shape,  stride)
    TestCase((2, 3), None),
]

# Data types - now using infinicore native types
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]

# Tolerance
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

# ==============================================================================
# Test Method
# ==============================================================================


# def test_matmul(device, test_case, dtype, config):
#     """
#     Test matmul operation

#     Args:
#         device: device enum
#         test_case: test case
#         dtype: infinicore data type
#         config: test config
#     """
#     a_shape, b_shape, result_shape, a_stride, b_stride, c_stride = test_case.args

#     print(
#         f"Testing Matmul on {InfiniDeviceNames[device]} with "
#         f"a_shape:{a_shape}, b_shape:{b_shape}, result_shape:{result_shape}, "
#         f"a_stride:{a_stride}, b_stride:{b_stride}, c_stride:{c_stride}, "
#         f"dtype:{dtype}"
#     )

#     # Create PyTorch tensors
#     device_str = torch_device_map[device]
#     torch_dtype = to_torch_dtype(dtype)

#     torch_a = torch.rand(a_shape, dtype=torch_dtype, device=device_str)
#     torch_b = torch.rand(b_shape, dtype=torch_dtype, device=device_str)

#     # Calculate PyTorch reference result
#     def torch_matmul():
#         return torch.matmul(torch_a, torch_b)

#     torch_result = torch_matmul()

#     # Create infinicore tensors
#     infini_a = create_infinicore_tensor(torch_a, device)
#     infini_b = create_infinicore_tensor(torch_b, device)

#     # Out-of-place matmul
#     def infini_matmul():
#         return infinicore.matmul(infini_a, infini_b)

#     infini_result = infini_matmul()

#     # Validate results using common method
#     is_valid = compare_results(
#         infini_result, torch_result, dtype, config, device_str, device
#     )
#     assert is_valid, "Matmul test failed"

#     # Performance test
#     if config.bench:
#         profile_operation(
#             "PyTorch",
#             torch_matmul,
#             device_str,
#             config.num_prerun,
#             config.num_iterations,
#         )
#         profile_operation(
#             "Infinicore",
#             infini_matmul,
#             device_str,
#             config.num_prerun,
#             config.num_iterations,
#         )


def test_ones_inplace(device, test_case, dtype, config):
    """
    Test in-place ones operation

    Args:
        device: device enum
        test_case: test case
        dtype: infinicore data type
        config: test config
    """
    x_shape,  stride = test_case.args

    print(
        f"Testing In-place Matmul on {InfiniDeviceNames[device]} with "
        f"x_shape:{x_shape}, stride:{stride}, "
        f"dtype:{dtype}"
    )

    device_str = torch_device_map[device]
    torch_dtype = to_torch_dtype(dtype)

    # Create PyTorch tensors
    torch_x = torch.rand(x_shape, dtype=torch_dtype, device=device_str)


    # Create pre-allocated result tensor
    torch_preallocated = torch.zeros(x_shape, dtype=torch_dtype, device=device_str)

    # Calculate PyTorch reference result using in-place operation
    def torch_ones_inplace():
        torch.ones(x_shape, out=torch_preallocated)

    # Execute in-place operation
    torch_ones_inplace()

    # Create infinicore tensors
    infini_x = create_infinicore_tensor(torch_x, device)

    infini_y = infinicore.empty(
        x_shape, dtype, infinicore.device(device_str, 0), False
    )

    # Test in-place matmul
    def infini_ones_inplace():
        infinicore.ones_(infini_y, infini_x)

    # Execute in-place operation
    infini_ones_inplace()

    # Validate results using common method
    is_valid = compare_results(
        infini_y, torch_preallocated, dtype, config, device_str, device
    )
    assert is_valid, "In-place matmul test failed"

    # Performance test
    if config.bench:
        profile_operation(
            "PyTorch In-place",
            torch_ones_inplace,
            device_str,
            config.num_prerun,
            config.num_iterations,
        )
        profile_operation(
            "Infinicore In-place",
            infini_ones_inplace,
            device_str,
            config.num_prerun,
            config.num_iterations,
        )


# ==============================================================================
# Main Execution Function
# ==============================================================================


def main():
    args = get_args()

    # Create test configuration
    config = TestConfig(
        tensor_dtypes=_TENSOR_DTYPES,
        tolerance_map=_TOLERANCE_MAP,
        debug=args.debug,
        bench=args.bench,
        num_prerun=args.num_prerun,
        num_iterations=args.num_iterations,
    )

    # Create test runner
    runner = TestRunner(_TEST_CASES, config)

    # Get test devices
    devices = get_test_devices(args)

    print("Starting matmul tests...")

    all_passed = True

    # Run out-of-place tests
    # print("\n--- Testing Out-of-place Matmul ---")
    # out_of_place_passed = runner.run_tests(devices, test_matmul)
    # all_passed = all_passed and out_of_place_passed

    # Run in-place tests
    print("\n--- Testing In-place Matmul ---")
    in_place_passed = runner.run_tests(devices, test_ones_inplace)
    all_passed = all_passed and in_place_passed

    runner.print_summary()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

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
    TestCase((2, 4, 6), (10, 1)),
    TestCase((1, 2048), None),
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

def test_ones(device, test_case, dtype, config):
    """
    Test ones operation

    Args:
        device: device enum
        test_case: test case
        dtype: infinicore data type
        config: test config
    """
    x_shape, stride = test_case.args

    print(
        f"Testing ones on {InfiniDeviceNames[device]} with "
        f"x_shape:{x_shape}, stride:{stride}, "
        f"dtype:{dtype}"
    )

    device_str = torch_device_map[device]
    torch_dtype = to_torch_dtype(dtype)

    # Create pre-allocated result tensor
    torch_preallocated = torch.zeros(x_shape, dtype=torch_dtype, device=device_str)

    # Calculate PyTorch reference result
    def torch_ones():
        torch.ones(x_shape, out=torch_preallocated)

    # Execute operation
    torch_ones()

    # Calculate infini result
    def infini_ones():
        return infinicore.ones(x_shape, dtype=dtype, device=infinicore.device(device_str, 0))

    # Execute operation
    infini_y = infini_ones()

    def print_infini_tensor(infini_tensor):
        torch_tensor = torch.rand(infini_tensor.shape, dtype=torch_dtype, device=device_str)
        temp_tensor = create_infinicore_tensor(torch_tensor, device_str)
        temp_tensor.copy_(infini_tensor)
        print(torch_tensor)

    # print_infini_tensor(infini_y)

    # Validate results using common method
    is_valid = compare_results(infini_y, torch_preallocated, dtype, config, device_str)
    assert is_valid, "ones test failed"

    # Performance test
    if config.bench:
        profile_operation(
            "PyTorch ones",
            torch_ones,
            device_str,
            config.num_prerun,
            config.num_iterations,
        )
        profile_operation(
            "Infinicore ones",
            infini_ones,
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

    print("Starting ones tests...")

    all_passed = True

    # Run ones tests
    print("\n--- Testing ones ---")
    in_passed = runner.run_tests(devices, test_ones)
    all_passed = all_passed and in_passed

    runner.print_summary()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

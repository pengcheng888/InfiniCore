import sys
import argparse
from pathlib import Path

# Import components from the unified framework package
from framework.loader import TestDiscoverer
from framework.driver import TestDriver
from framework.printer import ConsolePrinter
from framework.types import TestTiming
from framework import get_hardware_args_group, add_common_test_args

def generate_help_epilog(ops_dir=None):
    """
    Generate dynamic help epilog containing available operators and hardware platforms.
    Maintains the original output format for backward compatibility.
    """
    # === Adapter: Use TestDiscoverer to get operator list ===
    # Temporarily instantiate a Discoverer just to fetch the list
    discoverer = TestDiscoverer(ops_dir)
    operators = discoverer.get_available_operators()

    # Build epilog text (fully replicating original logic)
    epilog_parts = []

    # Examples section
    epilog_parts.append("Examples:")
    epilog_parts.append("  # Run all operator tests on CPU")
    epilog_parts.append("  python run.py --cpu")
    epilog_parts.append("")
    epilog_parts.append("  # Run specific operators")
    epilog_parts.append("  python run.py --ops add matmul --nvidia")
    epilog_parts.append("")
    epilog_parts.append("  # Run with debug mode on multiple devices")
    epilog_parts.append("  python run.py --cpu --nvidia --debug")
    epilog_parts.append("")
    epilog_parts.append(
        "  # Run with verbose mode to stop on first error with full traceback"
    )
    epilog_parts.append("  python run.py --cpu --nvidia --verbose")
    epilog_parts.append("")
    epilog_parts.append("  # Run with benchmarking (both host and device timing)")
    epilog_parts.append("  python run.py --cpu --bench")
    epilog_parts.append("")
    epilog_parts.append("  # Run with host timing only")
    epilog_parts.append("  python run.py --nvidia --bench host")
    epilog_parts.append("")
    epilog_parts.append("  # Run with device timing only")
    epilog_parts.append("  python run.py --nvidia --bench device")
    epilog_parts.append("")
    epilog_parts.append("  # List available tests without running")
    epilog_parts.append("  python run.py --list")
    epilog_parts.append("")

    # Available operators section
    if operators:
        epilog_parts.append("Available Operators:")
        # Group operators for better display
        operators_per_line = 4
        for i in range(0, len(operators), operators_per_line):
            line_ops = operators[i : i + operators_per_line]
            epilog_parts.append(f"  {', '.join(line_ops)}")
        epilog_parts.append("")
    else:
        epilog_parts.append("Available Operators: (none detected)")
        epilog_parts.append("")

    # Additional notes
    epilog_parts.append("Note:")
    epilog_parts.append(
        "  - Use '--' to pass additional arguments to individual test scripts"
    )
    epilog_parts.append(
        "  - Operators are automatically discovered from the ops directory"
    )
    epilog_parts.append(
        "  - --bench mode now shows cumulative timing across all operators"
    )
    epilog_parts.append(
        "  - --bench host/device/both controls host/device timing measurement"
    )
    epilog_parts.append(
        "  - --verbose mode stops execution on first error and shows full traceback"
    )

    return "\n".join(epilog_parts)

def main():
    """Main entry point for the InfiniCore Operator Test Runner."""
    parser = argparse.ArgumentParser(
        description="Run InfiniCore operator tests across multiple hardware platforms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=generate_help_epilog()
    )
    parser.add_argument("--ops-dir", type=str, help="Path to the ops directory (default: auto-detect)")
    parser.add_argument("--ops", nargs="+", help="Run specific operators only (e.g., --ops add matmul)")
    parser.add_argument("--list", action="store_true", help="List all available test files without running them")
    
    # Add common test arguments (including --save, --bench, etc.)
    add_common_test_args(parser)
    get_hardware_args_group(parser)
    
    args, unknown_args = parser.parse_known_args()
    # Show what extra arguments will be passed
    if unknown_args:
        print(f"Passing extra arguments to test scripts: {unknown_args}")

    # 1. Discovery
    discoverer = TestDiscoverer(args.ops_dir)
    if args.list:
        print("Available operators:", discoverer.get_available_operators())
        return

    if args.verbose:
        print(f"Verbose mode: ENABLED (will stop on first error with full traceback)")

    if args.bench:
        bench_mode = args.bench if args.bench != "both" else "both"
        print(f"Benchmark mode: {bench_mode.upper()} timing")
    
    target_ops = None

    if args.ops:
        # Get all available operator names
        available_ops = set(discoverer.get_available_operators())
        requested_ops = set(args.ops)
        
        # Classify using set operations
        valid_ops = list(requested_ops & available_ops)   # Intersection: Valid ops
        invalid_ops = list(requested_ops - available_ops) # Difference: Invalid ops
        
        # Warn if there are invalid operators
        if invalid_ops:
            print(f"⚠️  Warning: The following requested operators were not found:")
            print(f"   {', '.join(invalid_ops)}")
            print(f"   (Use --list to see available operators)")
            
        if not valid_ops:
            # Case A: User input provided, but ALL were invalid.
            print(f"⚠️  No valid operators remained from your list.")
            print(f"🔄 Fallback: Proceeding to run ALL available tests...")
            target_ops = None 
        else:
            # Case B: At least some valid operators found.
            print(f"🎯 Targeted operators: {', '.join(valid_ops)}")
            target_ops = valid_ops

        target_ops = valid_ops

    test_files = discoverer.scan(target_ops)
    if not test_files:
        print("No tests found.")
        sys.exit(0)

    # 2. Preparation
    dirver = TestDriver()
    cumulative_timing = TestTiming()
    printer = ConsolePrinter()
    results = []
    
    printer.print_header(discoverer.ops_dir, len(test_files))

    # 3. Execution Loop
    for f in test_files:
        result = dirver.drive(f)
        results.append(result)
        
        # Real-time reporting and printing of stdout
        printer.print_live_result(result, verbose=args.verbose)

        # Accumulate timing
        if result.success:
            cumulative_timing.torch_host += result.timing.torch_host
            cumulative_timing.infini_host += result.timing.infini_host
            cumulative_timing.torch_device += result.timing.torch_device
            cumulative_timing.infini_device += result.timing.infini_device

        # Fail fast in verbose mode
        if args.verbose and not result.success:
            print("\nStopping due to failure in verbose mode.")
            break

    # 4. Final Report & Save
    all_passed = printer.print_summary(
        results, 
        cumulative_timing if args.bench else None,
        ops_dir=discoverer.ops_dir,
        total_expected=len(test_files),
        verbose=args.verbose
    )

    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()

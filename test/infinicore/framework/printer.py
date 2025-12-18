# lib/printer.py
import sys
from .types import OperatorTestResult, TestTiming

class ConsolePrinter:
    """
    Handles all console output logic.
    Acts as the 'View' in the application structure.
    """

    def list_tests(self, discoverer):
        """
        Intelligently list available tests.
        If no valid operators are found, it falls back to listing raw Python files 
        to assist with debugging (e.g., typos in class inheritance).
        """
        ops_dir = discoverer.ops_dir
        operators = discoverer.get_available_operators()

        if operators:
            print(f"Available operator test files in {ops_dir}:")
            for operator in operators:
                print(f"  - {operator}")
            print(f"\nTotal: {len(operators)} operators")
        else:
            print(f"No valid operator tests found in {ops_dir}")
            
            # === Fallback Debug Logic ===
            raw_files = discoverer.get_raw_python_files()
            if raw_files:
                print(f"\n💡 Debug Hint: Found the following Python files (but they are not valid tests):")
                print(f"   {raw_files}")
                print("   (Ensure they inherit from 'BaseOperatorTest' and contain 'infinicore')")

    def print_header(self, ops_dir, count):
        print(f"InfiniCore Operator Test Runner")
        print(f"Directory: {ops_dir}")
        print(f"Tests found: {count}\n")

    def print_live_result(self, result, verbose=False):
        """Print single-line result in real-time."""
        
        print(f"{result.status_icon}  {result.name}: {result.status_text} (code: {result.return_code})")
        
        # Only print details if verbose or if the test failed/had output
        if result.stdout:
            print(result.stdout.rstrip())
            
        if result.stderr:
            print("\nSTDERR:", result.stderr.rstrip())
            
        if result.error_message:
            print(f"💥 Error: {result.error_message}")

        if result.stdout or result.stderr or verbose:
            print("-" * 40)

    def print_summary(
            self, 
            results, 
            cumulative_timing, 
            ops_dir, 
            total_expected=0, 
            verbose=False,
            bench_mode="both"
        ):
        """Prints the final comprehensive test summary and statistics, ensuring consistency with original output."""
        print(f"\n{'='*80}\nCUMULATIVE TEST SUMMARY\n{'='*80}")
        
        passed = [r for r in results if r.return_code == 0]
        failed = [r for r in results if r.return_code == -1]
        skipped = [r for r in results if r.return_code == -2]
        partial = [r for r in results if r.return_code == -3]

        total = len(results)
        print(f"Total tests run: {total}")
        if total_expected > 0 and total < total_expected:
             print(f"Total tests expected: {total_expected}")
             print(f"Tests not executed: {total_expected - total}")

        print(f"Passed: {len(passed)}")
        print(f"Failed: {len(failed)}")
        if skipped: print(f"Skipped: {len(skipped)}")
        if partial: print(f"Partial: {len(partial)}")

        # 1. Print Benchmark data
        if cumulative_timing:
            # Call the internal helper method
            self._print_timing(cumulative_timing, bench_mode=bench_mode)

        # 2. Print Detailed Lists
        
        # PASSED
        if passed:
            self._print_op_list("✅ PASSED OPERATORS", passed)
        else:
            print(f"\n✅ PASSED OPERATORS: None")

        # FAILED
        if failed:
            self._print_op_list("❌ FAILED OPERATORS", failed)

        # SKIPPED
        if skipped:
             self._print_op_list("⏭️  SKIPPED OPERATORS", skipped)

        # PARTIAL
        if partial:
             self._print_op_list("⚠️  PARTIAL IMPLEMENTATIONS", partial)

        # 3. Restore Success Rate
        if total > 0:
            # Calculate success rate based on actually executed tests (excluding skipped)
            executed_tests = total - len(skipped)
            if executed_tests > 0:
                success_rate = len(passed) / executed_tests * 100
                print(f"\nSuccess rate: {success_rate:.1f}%")
        
        if not failed:
            if skipped or partial:
                print(f"\n⚠️  Tests completed with some operators not fully implemented")
            else:
                print(f"\n🎉 All tests passed!")
        else:
            print(f"\n❌ {len(failed)} tests failed")

        if not failed and (skipped or partial):
            print(f"\n⚠️  Note: Some operators are not fully implemented")
            print(f"   Run individual tests for details on missing implementations")
        
        if verbose and failed:
            print(f"\n💡 Verbose mode tip: Use individual test commands for detailed debugging:")
            # Show first 3 failed operators to avoid spamming
            for r in failed[:3]:
                # Construct file path: ops_dir / filename.py
                file_path = ops_dir / (r.name + ".py")
                print(f"   python {file_path} --verbose")
            
            if len(failed) > 3:
                print(f"   ... (and {len(failed) - 3} others)")

        return len(failed) == 0

    # --- Internal Helpers ---
    def _print_op_list(self, title, result_list):
        """Helper to print a formatted list of operator names."""
        print(f"\n{title} ({len(result_list)}):")
        names = [r.name for r in result_list]
        # Group by 10 per line
        for i in range(0, len(names), 10):
            print("  " + ", ".join(names[i : i + 10]))
            
    def _print_timing(self, t, bench_mode="both"):
        """Prints detailed timing breakdown for host and device, based on bench_mode."""
        
        print(f"{'-'*40}")
        
        # Restore Operators Tested field using the dataclass field
        if hasattr(t, 'operators_tested') and t.operators_tested > 0:
            print(f"BENCHMARK SUMMARY:")
            print(f"  Operators Tested: {t.operators_tested}")
        
        # Restore detailed Host/Device distinction
        if bench_mode in ["host", "both"]:
            print(
                f"  PyTorch Host Total Time:     {t.torch_host:12.3f} ms"
            )
            print(
                f"  InfiniCore Host Total Time:  {t.infini_host:12.3f} ms"
            )
        
        if bench_mode in ["device", "both"]:
            print(
                f"  PyTorch Device Total Time:   {t.torch_device:12.3f} ms"
            )
            print(
                f"  InfiniCore Device Total Time: {t.infini_device:12.3f} ms"
            )
        
        print(f"{'-'*40}")

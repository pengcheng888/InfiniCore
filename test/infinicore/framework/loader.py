from pathlib import Path

class TestDiscoverer:
    def __init__(self, ops_dir_path=None):
        self.ops_dir = self._resolve_dir(ops_dir_path)

    def _resolve_dir(self, path):
        if path:
            p = Path(path)
            if p.exists(): return p
        
        # Default fallback logic: 'ops' directory under the parent of the current file's parent.
        # Note: Since this file is in 'framework/', we look at parent.parent.
        # It is recommended to pass an explicit path in run.py.
        fallback = Path(__file__).parent.parent / "ops" 
        return fallback if fallback.exists() else None

    def get_available_operators(self):
        """Returns a list of names of all available operators."""
        if not self.ops_dir: return []
        files = self.scan()
        return sorted([f.stem for f in files])

    def get_raw_python_files(self):
        """
        Get all .py files in the directory (excluding run.py) without content validation.
        Used for debugging: helps identify files that exist but failed validation.
        """
        if not self.ops_dir or not self.ops_dir.exists():
            return []
            
        files = list(self.ops_dir.glob("*.py"))
        # Exclude run.py itself and __init__.py
        return [f.name for f in files if f.name != "run.py" and not f.name.startswith("__")]
        
    def scan(self, specific_ops=None):
        """Scans and returns a list of Path objects that meet the criteria."""
        if not self.ops_dir or not self.ops_dir.exists():
            return []

        # 1. Find all .py files
        files = list(self.ops_dir.glob("*.py"))
        
        target_ops_set = set(specific_ops) if specific_ops else None

        # 2. Filter out non-test files (via content check)
        valid_files = []
        for f in files:
            # A. Basic Name Filtering
            if f.name.startswith("_") or f.name == "run.py":
                continue

            # B. Specific Ops Filtering
            if target_ops_set and f.stem not in target_ops_set:
                continue

            # C. Content Check (Expensive I/O)
            # Only perform this check if the file passed the name filters above.
            if self._is_operator_test(f):
                valid_files.append(f)
        
        return valid_files

    def _is_operator_test(self, file_path):
        """Checks if the file content contains operator test characteristics."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                return "infinicore" in content and (
                    "BaseOperatorTest" in content or "GenericTestRunner" in content
                )
        except:
            return False

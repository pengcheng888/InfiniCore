<!-- Copilot / AI agent instructions for InfiniCore -->
# InfiniCore — Quick AI coding guide

This file collects focused, actionable notes for AI coding agents working in InfiniCore so they can be productive immediately.

## 1) Big picture
- **Purpose**: cross-platform core runtime and operator library exposing C API and Python bindings (pybind11) for multiple hardware backends.
- **Major components**:
  - `src/infinirt` — runtime implementations and platform adapters (CPU, CUDA, Ascend, Cambricon, etc.).
  - `src/infiniop` — operator implementations with device-specific code (see `src/infiniop/ops/*/operator.cc` for pattern).
  - `src/infiniccl` — communication library (InfiniCCL).
  - `src/infinicore` + `python/infinicore` — C++ glue with pybind11 and the Python package.
  - `xmake.lua` — canonical build configuration: targets, platform options, and install locations.
- **Data flow**: Core C libraries (InfiniRT/InfiniOP/InfiniCCL) → pybind11 bindings (`_infinicore`) → Python package (`python/infinicore`).

## 2) Build & install (exact commands)
- **Submodules**: clone with `--recursive` or run `git submodule update --init --recursive`.
- **Set environment**: `source scripts/set_env_linux.sh` (sets `INFINI_ROOT=$HOME/.infini` and `LD_LIBRARY_PATH`).
- **One-step install** (recommended):
  - `python scripts/install.py [XMAKE_CONFIG_FLAGS]`
  - Example: `python scripts/install.py --omp=y -y` (used in CI)
- **Manual xmake flow** (useful for adding backends):
  - Configure: `xmake f --nv-gpu=true --cuda=$CUDA_HOME -cv`
  - Build & install: `xmake build && xmake install`
- **Python bindings & editable dev**:
  - Build extension: `xmake build _infinicore && xmake install _infinicore`
  - Install package: `pip install -e .`

## 3) Test workflows
- **Python operator tests**: `python scripts/python_test.py --cpu` or `python test/infinicore/run.py --cpu`
- **Native C++ tests**: `xmake build infinirt-test && ./build/linux/x86_64/release/infinirt-test --cpu`
- **InfiniCCL tests**: `xmake build infiniccl-test && ./build/linux/x86_64/release/infiniccl-test --nvidia`

## 4) Code patterns & conventions

### Platform-specific conditional compilation
Each backend (NVIDIA, Ascend, Cambricon, etc.) is guarded by `#ifdef ENABLE_*_API`:
```cpp
// In src/infiniop/ops/gemm/operator.cc (pattern for all operators)
#ifdef ENABLE_CPU_API
  #include "cpu/gemm_cpu.h"
#endif
#ifdef ENABLE_NVIDIA_API
  #include "nvidia/gemm_nvidia.cuh"
#endif
```
Backend options defined in `xmake.lua`: `--nv-gpu`, `--ascend-npu`, `--cambricon-mlu`, `--metax-gpu`, etc.

### Adding an operator
1. Create `src/infiniop/ops/<op_name>/` with `operator.cc` and `<device>/<op_name>_<device>.{cc,cu,h}`
2. In `operator.cc`: implement C API using macro pattern (see `CREATE` macro in `gemm/operator.cc`)
3. Create Python binding in `python/infinicore/ops/<op_name>.py`
4. Register in `python/infinicore/__init__.py` and CI test under `test/infinicore/ops/`

### C++ naming conventions
- Internal types: `UpperCamelCase` (e.g., `InfiniopMatmulCudaDescriptor`)
- C API types: `infinixx[Xxx]_t` (e.g., `infiniopMatmulCudaDescriptor_t`, `infiniDtype_t`)
- Constants: `INFINI_UPPER_SNAKE_CASE` (e.g., `INFINI_DEVICE_NVIDIA`)
- Variables/functions: `snake_case` for params, `lowerCamelCase` for methods

### Install order matters
1. Configure xmake with backend flags
2. Build and install core libraries (InfiniRT/InfiniOP/InfiniCCL) to `INFINI_ROOT`
3. Build and install `_infinicore` Python extension
4. Install Python package with `pip install -e .`

## 5) CI and formatting
- **CI pipeline** (`.github/workflows/build.yml`):
  - Runs on Windows and Ubuntu with `debug` and `release` modes
  - Steps: format check → install xmake → `python scripts/install.py --omp=y -y` → `python scripts/python_test.py --cpu` → native C++ tests
- **Formatting**: `python3 scripts/format.py --path src --check` (uses `clang-format-16` for C++ and `black` for Python)

## 6) Integration points & external deps
- **Hardware SDKs**: conditional compilation for CUDA, Ascend, Cambricon (expect runtime linking)
- **Optional**: `ntops` (NineToothed) operator library; use `scripts/build_ntops.py` to AOT build
- **Install locations**: native libs → `INFINI_ROOT/lib`, headers → `INFINI_ROOT/include`, Python → `python/infinicore/lib/`

## 7) Quick reference for AI changes
- **Modifying Python bindings**: edit `src/infinicore/pybind11/`, rebuild with `xmake build _infinicore && xmake install _infinicore`
- **Adding backend option**: update `xmake.lua` with `option("new-gpu")` section, add `xmake/<new-gpu>.lua`, add platform dir `src/*/new-gpu/`
- **Updating build config**: xmake is the single source of truth; all platform logic lives in `xmake.lua` and `xmake/*.lua`
- **Testing locally**: `pip install -e . && python scripts/python_test.py --cpu` for quick iteration

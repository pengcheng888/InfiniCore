# NVIDIA RMSNorm dynamic block reproduction

This directory records the NVIDIA A100 version of the Moore S5000 RMSNorm dynamic-block fix.

## Files

- `0001-nvidia-rmsnorm-dynamic-block.patch`
  - Dynamic block-size fix for NVIDIA RMSNorm.
- `bench_nvidia_rmsnorm.py`
  - Python benchmark for Qwen3 q_norm/k_norm and multiple normalized dimensions.
- `bench_nvidia_rmsnorm.sh`
  - Environment wrapper for the benchmark.
- `nvidia-a100-rmsnorm-before-after.csv`
  - Measured before/after results.
- `nvidia-a100-rmsnorm-recheck-20261127.csv`
  - Recheck run after adding this reproduction directory.
- `2026-11-27-nvidia-a100-rmsnorm-summary.md`
  - Full test and fix report.

## Apply patch

From the InfiniCore repository root:

```bash
git apply fix-rmsnorm-nvidia/0001-nvidia-rmsnorm-dynamic-block.patch
```

## Build

```bash
export CUTLASS_HOME=/workspace_40/InfiniCore/third_party/cutlass/
xmake build -y infiniop-nvidia
xmake build -y infinicore_cpp_api
xmake install -y infinicore_cpp_api
xmake build -y _infinicore
xmake install -y _infinicore
```

## Benchmark

```bash
bash fix-rmsnorm-nvidia/bench_nvidia_rmsnorm.sh
```

With CSV output:

```bash
bash fix-rmsnorm-nvidia/bench_nvidia_rmsnorm.sh \
  --output fix-rmsnorm-nvidia/nvidia-a100-rmsnorm-current.csv
```

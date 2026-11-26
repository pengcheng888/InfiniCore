# RMSNorm dynamic block-size reproduction

This directory reproduces the Moore S5000 RMSNorm benchmark.

## Files

- `0001-rmsnorm-dynamic-block.patch`
  - Dynamic block-size fix for Moore RMSNorm
- `bench-qwen3-rmsnorm.sh`
  - Qwen3 q_norm/k_norm RMSNorm benchmark
- `2026-11-26-moore-s5000-rmsnorm-summary.md`
  - Full analysis and performance report

## Usage

Apply the patch if your source tree does not already contain it:

```bash
cd /workspace/InfiniCore
git apply fix-rmsnorm/0001-rmsnorm-dynamic-block.patch
```

Build InfiniCore:

```bash
source scripts/set_env_linux.sh
xmake build -y infinicore_cpp_api
xmake install -y infinicore_cpp_api
```

Run the benchmark:

```bash
bash fix-rmsnorm/bench-qwen3-rmsnorm.sh
```

Run only one case:

```bash
CASE=q bash fix-rmsnorm/bench-qwen3-rmsnorm.sh
CASE=k bash fix-rmsnorm/bench-qwen3-rmsnorm.sh
```

# RMSNorm dynamic block-size reproduction on Hygon BW1000

This directory records the Hygon BW1000 verification of the RMSNorm launch
configuration fix. Hygon uses the CUDA-compatible `rms_norm::nvidia`
implementation in InfiniCore.

## Files

- `0001-hygon-rmsnorm-dynamic-block.patch`
  Dynamic block-size fix.
- `bench_bw1000_rmsnorm.py`
  Standalone RMSNorm benchmark and correctness check.
- `bench_bw1000_rmsnorm.sh`
  BW1000 environment wrapper.
- `hygon-bw1000-rmsnorm-before.csv`
  Fixed-maximum-block baseline.
- `hygon-bw1000-rmsnorm-after.csv`
  Dynamic-block result.
- `2026-11-27-hygon-bw1000-rmsnorm-summary.md`
  Full analysis and performance report.
- `hygon-bw1000-rmsnorm-before.log`
  Raw baseline benchmark output.
- `hygon-bw1000-rmsnorm-after.log`
  Raw dynamic-block benchmark output.
- `hygon-bw1000-rmsnorm-operator-test.log`
  Raw `test/infiniop/rms_norm.py --hygon` output.

## Usage

```bash
cd /workspace_v0.2.9/InfiniCore
bash fix-rmsnorm-hygon/bench_bw1000_rmsnorm.sh \
  --warmup 20 --iterations 100 \
  --output fix-rmsnorm-hygon/hygon-bw1000-rmsnorm-after.csv
```

Run the InfiniCore RMSNorm operator test:

```bash
export PYTHONPATH="$PWD/python:$PYTHONPATH"
export LD_LIBRARY_PATH="$HOME/.infini/lib:$PWD/python/infinicore/lib:$LD_LIBRARY_PATH"
python3 test/infiniop/rms_norm.py --hygon
```

The benchmark loads libraries from `$HOME/.infini/lib` first so that before and
after runs use the library version installed by `xmake install`.

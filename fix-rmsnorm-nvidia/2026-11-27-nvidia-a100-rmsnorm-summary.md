# NVIDIA A100 RMSNorm Dynamic Block 修复记录

日期：2026-11-27  
测试日期：2026-11-26
复验日期：2026-11-27

## 1. 背景

Moore S5000 的最新提交 `201a62f4fba593df46c6dc0ba52b9090efedee99` 修复了 RMSNorm 在小维度上固定使用设备最大 block 的问题。NVIDIA A100 的 RMSNorm 实现存在同样的问题：

```text
A100 maxThreadsPerBlock = 1024
Qwen3 head_dim = 128
```

原实现固定使用 1024 线程处理 128 个元素，造成线程浪费和调度开销。

## 2. 测试环境

```text
GPU: NVIDIA A100-SXM4-80GB
Compute capability: 8.0
maxThreadsPerBlock: 1024
PyTorch: 2.13.0+cu130
CUDA: 13.0
Python: 3.12
InfiniCore branch: pd-v0.2.9
InfiniCore source: /workspace_40/InfiniCore
```

## 3. Moore S5000 测试口径

本次参考 Moore S5000 的测试方法：

```text
dtype: BF16
warmup: 20
iterations: 100
device synchronization: torch.cuda.synchronize()
reference: torch.nn.functional.rms_norm
correctness tolerance: rtol=1e-2, atol=1e-2
```

测试用例：

```text
Qwen3 q_norm: (4096, 32, 128)
Qwen3 k_norm: (4096, 8, 128)
dim 256:      (2048, 32, 256)
dim 512:      (1024, 32, 512)
dim 1024:     (512, 32, 1024)
dim 2048:     (256, 32, 2048)
dim 4096:     (128, 32, 4096)
```

Moore S5000 还测试了 Qwen3-8B 端到端 prefill；本次按要求只处理并测试 InfiniCore 算子层。

## 4. 修改内容

修改文件：

```text
src/infiniop/devices/nvidia/nvidia_kernel_common.cuh
src/infiniop/ops/rms_norm/nvidia/rms_norm_nvidia.cu
```

核心策略：

```text
block_size = min(maxThreadsPerBlock, next_pow2(dim))
```

新增 block 常量：

```text
64 / 128 / 256
```

当 `dim` 大于设备最大 block 时，回退到最大可用 block。该修改只改变 kernel launch 线程数，不改变 RMSNorm 数值计算逻辑。

补丁文件：

```text
0001-nvidia-rmsnorm-dynamic-block.patch
```

## 5. 修复前后性能

| 用例 | Shape | 修复前 | 修复后 | 加速比 |
|---|---|---:|---:|---:|
| q_norm | `(4096, 32, 128)` | 1.537703 ms | 0.211523 ms | 7.270x |
| k_norm | `(4096, 8, 128)` | 0.342377 ms | 0.055655 ms | 6.152x |
| dim 256 | `(2048, 32, 256)` | 0.666756 ms | 0.212893 ms | 3.132x |
| dim 512 | `(1024, 32, 512)` | 0.357817 ms | 0.201770 ms | 1.773x |
| dim 1024 | `(512, 32, 1024)` | 0.213677 ms | 0.213706 ms | 1.000x |
| dim 2048 | `(256, 32, 2048)` | 0.123171 ms | 0.123067 ms | 1.001x |
| dim 4096 | `(128, 32, 4096)` | 0.084332 ms | 0.083969 ms | 1.004x |

规律与 Moore S5000 一致：

1. `dim` 越小，收益越明显。
2. `dim` 达到 A100 的最大 block 1024 后，修复前后基本不变。
3. `dim` 大于 1024 时回退到 1024，因此性能也基本不变。

### Qwen3 单层 q_norm + k_norm

```text
修复前: 1.880080 ms
修复后: 0.267178 ms
加速:   7.037x
```

按 36 层估算：

```text
修复前: 67.683 ms
修复后: 9.618 ms
节省:   58.065 ms
```

### 2026-11-27 复验

保存于 `nvidia-a100-rmsnorm-recheck-20261127.csv`：

| 用例 | InfiniCore | PyTorch | 吞吐 |
|---|---:|---:|---:|
| q_norm | 0.211304 ms | 0.157968 ms | 79.398 Gelem/s |
| k_norm | 0.055457 ms | 0.038529 ms | 75.631 Gelem/s |
| dim 256 | 0.212643 ms | 0.086362 ms | 78.898 Gelem/s |
| dim 512 | 0.196219 ms | 0.051703 ms | 85.502 Gelem/s |
| dim 1024 | 0.232030 ms | 0.048321 ms | 72.306 Gelem/s |
| dim 2048 | 0.133331 ms | 0.049295 ms | 125.832 Gelem/s |
| dim 4096 | 0.091084 ms | 0.052725 ms | 184.194 Gelem/s |

## 6. 构建命令

```bash
cd /workspace_40/InfiniCore

export CUTLASS_HOME=/workspace_40/InfiniCore/third_party/cutlass/

xmake build -y infiniop-nvidia
xmake build -y infinicore_cpp_api
xmake install -y infinicore_cpp_api
xmake build -y _infinicore
xmake install -y _infinicore
```

## 7. 基准测试

```bash
cd /workspace_40/InfiniCore
bash fix-rmsnorm-nvidia/bench_nvidia_rmsnorm.sh
```

输出 CSV：

```bash
bash fix-rmsnorm-nvidia/bench_nvidia_rmsnorm.sh \
  --output fix-rmsnorm-nvidia/nvidia-a100-rmsnorm-current.csv
```

## 8. 正确性验证

已运行：

```bash
cd /workspace_40/InfiniCore
env CUDA_VISIBLE_DEVICES=0 \
  INFINI_ROOT=/workspace_40/InfiniCore/python/infinicore \
  LD_LIBRARY_PATH=/workspace_40/InfiniCore/python/infinicore/lib \
  python3 test/infiniop/rms_norm.py --nvidia
```

结果：

```text
Test passed!
```

`git diff --check` 也已通过。

## 9. 结论

NVIDIA A100 上确实存在 Moore S5000 同类的 RMSNorm 小维度 block 浪费问题。动态 block 修复后：

```text
Qwen3 q_norm: 7.270x
Qwen3 k_norm: 6.152x
单层 q_norm + k_norm: 7.037x
```

修复后 Qwen3 `q_norm` 与 PyTorch 的差距从约 `9.7x` 缩小到约 `1.3x`，`k_norm` 从约 `10.8x` 缩小到约 `1.4x`。

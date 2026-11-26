# Moore S5000 RMSNorm 问题总结

日期：2026-11-26

## 1. 环境

- 平台：Moore S5000
- 架构：`mp_31`
- 容器：`wangpengcheng_new`
- InfiniCore 分支：`pd-v0.2.9`
- InfiniLM 分支：`pd-v0.2.9-v1-tp`
- `mate`：`0.1.3+mu436torch2.7`
- MUSA SDK：`4.3.6`
- 模型：Qwen3-8B
- 测试卡：GPU 0

## 1.1 Qwen3 结构参数

```text
num_hidden_layers = 36
num_attention_heads = 32
num_key_value_heads = 8
head_dim = 128
seq_len = 4096
```

## 2. 问题现象

在 Qwen3-8B 上，注释掉 `q_norm_` 和 `k_norm_` 后，prefill 性能提升约一倍：

| 配置 | Prefill TTFT | Prefill 吞吐 |
|---|---:|---:|
| RMSNorm 启用 | 689.48 ms | 5940.73 tok/s |
| RMSNorm 禁用 | 312.40 ms | 13111.59 tok/s |

提速幅度约：

```text
689.48 / 312.40 = 2.21x
```

这说明 RMSNorm 是当前 prefill 路径的重要瓶颈。

## 3. 根因分析

问题不在 `infinicore::nn::RMSNorm` 模块本身，而在 Moore 平台的 InfiniOp RMSNorm kernel。

相关文件：

```text
/workspace/InfiniCore/src/infiniop/ops/rms_norm/moore/rms_norm_moore.mu
/workspace/InfiniCore/src/infiniop/ops/rms_norm/cuda/kernel.cuh
```

原实现有三个关键特征：

1. 每个 block 处理一行
2. block size 固定取设备最大线程数，通常是 1024
3. Qwen3 的 `head_dim=128`

这导致：

```text
1024 threads / 128 elements = 8x 线程浪费
```

也就是说，在主循环中只有 `128 / 1024` 的线程承担有效工作。

同时，`q_norm` 的实际 shape 是：

```text
(4096, 32, 128)
```

因此会启动：

```text
4096 * 32 = 131072 blocks
```

`k_norm` 的 shape 是：

```text
(4096, 8, 128)
```

因此会启动：

```text
4096 * 8 = 32768 blocks
```

这会造成严重的线程浪费和调度开销。

### 3.1 每层耗时估算

修复前，单层 `q_norm + k_norm` 耗时约为：

```text
7.98 ms + 2.00 ms = 9.98 ms
```

Qwen3-8B 共 36 层，因此总耗时约为：

```text
36 * 9.98 ms = 359.3 ms
```

这与端到端 prefill 中约 `377 ms` 的性能差距基本吻合，进一步确认 RMSNorm 是主要瓶颈。

## 4. 修改方案

采用的修复是：按 `dim` 动态选择 block size。

核心逻辑：

```text
block_size = min(maxThreadsPerBlock, next_pow2(dim))
```

实际实现中按以下顺序选择：

```text
64 / 128 / 256 / 512 / 1024 / 2048 / 4096
```

同时保留两类保护：

1. 不超过设备 `maxThreadsPerBlock`
2. 当 `dim > maxThreadsPerBlock` 时，回退到最大可用 block

这次修改不改变 RMSNorm 的数值计算逻辑，只改变 kernel launch 的线程配置。

补丁位置：

```text
/home/wangpengcheng/infinicore-moore-s5000-repro/patches/rmsnorm-dynamic-block.patch
```

## 5. 单测结果

### 5.1 Qwen3 q_norm

Shape：

```text
(4096, 32, 128), BF16
```

结果：

| 版本 | 耗时 | 吞吐 | 相对 PyTorch |
|---|---:|---:|---:|
| 修复前 | 7.980 ms | 2.10 Gelem/s | 182.68x |
| 修复后 | 1.108 ms | 15.14 Gelem/s | 25.31x |

性能提升：

```text
7.980 / 1.108 = 7.20x
```

### 5.2 Qwen3 k_norm

Shape：

```text
(4096, 8, 128), BF16
```

结果：

| 版本 | 耗时 | 相对 PyTorch |
|---|---:|---:|
| 修复前 | 2.0006 ms | 约 83x |
| 修复后 | 0.2802 ms | 15.31x |

性能提升：

```text
2.0006 / 0.2802 = 7.14x
```

### 5.3 不同 dim 的性能变化

以下为固定元素数、不同 `dim` 的对照结果，并给出 PyTorch 参考值：

| Shape | 修复前 InfiniCore | 修复后 InfiniCore | PyTorch | 修复后 / PyTorch |
|---|---:|---:|---:|---:|
| `(4096, 32, 128)` | 7.981 ms | 1.108 ms | 0.0442 ms | 25.07x |
| `(2048, 32, 256)` | 3.809 ms | 0.998 ms | 0.0480 ms | 20.78x |
| `(1024, 32, 512)` | 1.845 ms | 0.942 ms | 0.0425 ms | 22.19x |
| `(512, 32, 1024)` | 0.937 ms | 0.936 ms | 0.0483 ms | 19.37x |
| `(256, 32, 2048)` | 0.492 ms | 0.488 ms | 0.0460 ms | 10.62x |
| `(128, 32, 4096)` | 0.289 ms | 0.284 ms | 0.0507 ms | 5.61x |

可以看出：

- `dim` 越小，收益越明显
- `dim` 越大，修复前后差异越小

这符合“固定 1024 线程导致小维度浪费”的判断。

## 6. 端到端性能

测试条件：

```text
Qwen3-8B
TP=1
flash-attn
graph
batch=4
input=1024
output=4
```

结果：

| 配置 | Prefill TTFT | Prefill 吞吐 |
|---|---:|---:|
| RMSNorm 禁用 | 312.40 ms | 13111.59 tok/s |
| RMSNorm 启用，原 kernel | 689.48 ms | 5940.73 tok/s |
| RMSNorm 启用，动态 block | 368.48 ms | 11115.80 tok/s |

RMSNorm 带来的额外开销：

```text
原 kernel：689.48 - 312.40 = 377.08 ms
动态 block：368.48 - 312.40 = 56.08 ms
```

开销减少：

```text
(377.08 - 56.08) / 377.08 = 85.1%
```

## 7. 结论

1. RMSNorm 的问题已确认并复现
2. 根因是 Moore RMSNorm kernel 固定使用设备最大 block size
3. 动态 block size 修复有效
4. 单算子性能提升约 `7.2x`
5. 端到端 prefill 性能从 `5940.73 tok/s` 提升到 `11115.80 tok/s`
6. RMSNorm 引入的额外开销减少约 `85%`
7. 修复后仍比 PyTorch 慢约 `25x`

剩余瓶颈：

```text
一行一个 block + BlockReduce
```

下一步优化方向：

- 对 `dim <= 128` 增加 warp-level fast path
- 使用向量化访存
- 让一个 block 处理多行

## 8. 复现方式

单测脚本：

```bash
bash /home/wangpengcheng/infinicore-moore-s5000-repro/container-bench-qwen3-rmsnorm.sh
```

端到端脚本：

```bash
cd /workspace/InfiniLM
source /workspace/InfiniCore/scripts/set_env_linux.sh

BUILD=0 \
ATTN=flash-attn \
TP=1 \
ENABLE_GRAPH=1 \
BENCHMARK_MODES=generate \
BATCH_SIZES=4 \
INPUT_LEN=1024 \
OUTPUT_LEN=4 \
BENCHMARK_REPEATS=30 \
WARMUP=1 \
RESULT_DIR=benchmark-results/moore-s5000-qwen3-rmsnorm-dynamic-block \
bash ./run_bench_old.sh
```


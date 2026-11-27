# Hygon BW1000 RMSNorm 问题总结

## 1. 测试环境

- 平台：Hygon BW1000 / C-3000
- 架构：`gfx936`
- HIP 属性：`maxThreadsPerBlock=1024`，`wavefrontSize=64`，`CU=80`
- 数据类型：BF16
- 随机种子：`torch.manual_seed(0)`
- 计时方式：20 次 warmup 后计时 100 次，`torch.cuda.synchronize()` 包首尾
- 正确性：与 `torch.nn.functional.rms_norm` 比较，`rtol=1e-2`，`atol=1e-2`

## 2. 测试方法

参考 Moore S5000 的独立 RMSNorm 算子测试。每个用例创建相同形状的 `x`、
`weight` 和 `out`，用 `infinicore.from_torch` 转成 InfiniCore tensor，先执行一次
InfiniCore RMSNorm 并和 PyTorch 参考结果比较，然后分别计时
`infinicore.nn.functional.rms_norm` 与 PyTorch RMSNorm。

Moore 原始固定对比用例是 Qwen3 的两个归一化形状：

| 名称 | Shape |
|---|---:|
| q_norm | `(4096, 32, 128)` |
| k_norm | `(4096, 8, 128)` |

为了观察随维度收敛的趋势，本次沿用后续 NVIDIA 版基准增加以下对照：

| 名称 | Shape |
|---|---:|
| dim256 | `(2048, 32, 256)` |
| dim512 | `(1024, 32, 512)` |
| dim1024 | `(512, 32, 1024)` |
| dim2048 | `(256, 32, 2048)` |
| dim4096 | `(128, 32, 4096)` |

所有用例元素数固定为 `16777216`。

## 3. 根因

BW1000 的 Hygon RMSNorm 走 InfiniCore `rms_norm::nvidia` 后端。修复前 host 侧只根据
设备 `maxThreadsPerBlock` 选择 block：

```text
BW1000 maxThreadsPerBlock = 1024
Qwen3 head_dim          = 128
```

因此一个 128 元素的归一化行会启动 1024 线程。按 wavefront size 64 计算，每 block
启动 16 个 wavefront，其中只有 2 个 wavefront 有有效元素工作。

修复后选择最小可覆盖行宽的 2 的幂：

```text
block_size = min(maxThreadsPerBlock, next_pow2(dim))
```

因此 `dim=128` 时启动 128 线程，也就是 2 个 wavefront。该修改不改变 RMSNorm 数值
计算逻辑，只改变 launch 配置。

## 4. 算子性能

| Shape | 修复前 InfiniCore | 修复后 InfiniCore | 加速比 | 修复前吞吐 | 修复后吞吐 |
|---|---:|---:|---:|---:|---:|
| `(4096, 32, 128)` q_norm | 1.999349 ms | 0.443020 ms | 4.51x | 8.391 Gelem/s | 37.870 Gelem/s |
| `(4096, 8, 128)` k_norm | 0.503093 ms | 0.114657 ms | 4.39x | 8.337 Gelem/s | 36.581 Gelem/s |
| `(2048, 32, 256)` | 0.934502 ms | 0.320178 ms | 2.92x | 17.953 Gelem/s | 52.400 Gelem/s |
| `(1024, 32, 512)` | 0.545767 ms | 0.329409 ms | 1.66x | 30.741 Gelem/s | 50.931 Gelem/s |
| `(512, 32, 1024)` | 0.386322 ms | 0.386288 ms | 1.00x | 43.428 Gelem/s | 43.432 Gelem/s |
| `(256, 32, 2048)` | 0.255369 ms | 0.255633 ms | 1.00x | 65.698 Gelem/s | 65.630 Gelem/s |
| `(128, 32, 4096)` | 0.170896 ms | 0.170912 ms | 1.00x | 98.172 Gelem/s | 98.163 Gelem/s |

PyTorch 参考耗时分别为 q_norm `0.452059 ms`、k_norm `0.118357 ms`。修复后 InfiniCore
在该两项上略快于当前 PyTorch 参考。

结果符合预期：`dim < 1024` 时收益明显，维度越小收益越大；`dim >= 1024` 时新旧配置
都回退或使用 1024 block，耗时基本不变。

## 5. InfiniCore 算子测试

执行：

```bash
python3 test/infiniop/rms_norm.py --hygon
```

测试覆盖 16 组形状/stride 组合、F16/BF16 activation、F16/BF16/F32 weight 组合，
包含 Qwen3 q_norm 和 k_norm 的大形状用例。结果：

```text
Test passed!
```

## 6. 原始记录

完整 benchmark 输出保存在：

- `hygon-bw1000-rmsnorm-before.log`
- `hygon-bw1000-rmsnorm-after.log`

InfiniCore 算子测试完整输出保存在：

- `hygon-bw1000-rmsnorm-operator-test.log`

## 7. 结论

1. Hygon BW1000 存在 Moore S5000 同类的小维度 RMSNorm block 浪费问题。
2. 动态 block 配置在 BW1000 上有效，Qwen3 q_norm 提升 4.51x，k_norm 提升 4.39x。
3. 修改只影响独立 RMSNorm launch 配置；fused `add_rms_norm` 是独立算子，未在本报告中修改。

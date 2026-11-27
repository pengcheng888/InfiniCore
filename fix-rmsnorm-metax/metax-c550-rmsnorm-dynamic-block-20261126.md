# InfiniCore MetaX C550 RMSNorm Dynamic Block-Size Fix

更新时间：2026-11-27（Asia/Shanghai）

测试时间：2026-11-26（Asia/Shanghai）

## 范围

- 仓库：`/workspace_codex/InfiniCore`
- 分支：`pd-v0.2.9`
- 平台：MetaX C550
- 测试对象：InfiniOp RMSNorm 算子
- 参考问题：Moore S5000 RMSNorm 固定使用设备最大 block size 的问题

本记录只覆盖 InfiniCore 的 RMSNorm 算子级修复和测试，不包含 InfiniLM 端到端模型测试。

## 问题与修改

MetaX 原实现和 Moore 问题类似：当设备 `maxThreadsPerBlock` 为 1024 时，无论归一化维度 `dim` 是多少，都固定用 1024 线程启动 kernel。Qwen3 的 `head_dim=128` 会造成约 8 倍线程浪费。

修复策略是选择能覆盖一行的最小 2 的幂 block size，并保留设备最大线程数约束：

```text
64 / 128 / 256 / 512 / 1024 / 2048 / 4096
```

当 `dim` 大于设备最大可用 block size 时，回退到最大可用 block size。数值计算逻辑不变，只改变 kernel launch 的线程配置。

修改文件：

- `src/infiniop/devices/metax/metax_kernel_common.h`
- `src/infiniop/ops/rms_norm/metax/rms_norm_metax.maca`
- `test/infiniop/rms_norm.py`

源码差异见：

```text
infinicore-metax-rmsnorm-dynamic-block-20261126.patch
```

补丁在 InfiniCore 仓库根目录使用 `git apply` 应用。

## 构建与数值验证

构建命令等价于：

```bash
cd /workspace_codex/InfiniCore
xmake build _infinicore
xmake install infiniop
```

结果：

- `xmake build _infinicore` 成功，用时 162.417 s。
- 动态修复版 `libinfiniop.so` SHA-256：`045c0672f13edf6c160f5b1ee2cf69abf780d469`。
- 已安装到 `/root/.infini/lib/libinfiniop.so`。
- 默认路径执行 `python test/infiniop/rms_norm.py --metax`，全量数值测试通过。

测试覆盖原有 shape、stride、dtype 组合，并新增以下 `dim=128` 用例：

```text
(16, 128)
(4, 4, 128)
(4096, 32, 128)
(4096, 8, 128)
```

## 性能测试

测试口径参考 Moore S5000 的 `fix-rmsnorm/bench-qwen3-rmsnorm.sh`：

- dtype：BF16
- weight dtype：BF16
- warmup：20
- timed iterations：100
- 计时前后同步设备
- 先做数值校验，再计时
- fixed-1024 与 dynamic-block 使用相同设备、shape 和参数

只测试 Qwen3-8B 的两个关键 RMSNorm 形状：

| 用例 | shape | fixed-1024 | dynamic-block | 提升 | 耗时降低 |
| --- | --- | ---: | ---: | ---: | ---: |
| q_norm | `(4096, 32, 128)` | 0.008242 ms | 0.006742 ms | 1.222x | 18.199% |
| k_norm | `(4096, 8, 128)` | 0.007198 ms | 0.006251 ms | 1.151x | 13.156% |

吞吐：

| 用例 | fixed-1024 | dynamic-block |
| --- | ---: | ---: |
| q_norm | 2035.576 Gelem/s | 2488.463 Gelem/s |
| k_norm | 582.704 Gelem/s | 670.981 Gelem/s |

MetaX 提升幅度小于 Moore S5000 的约 7.2x。原因是 MetaX C550 的 fixed-1024 基线明显更快，但 `dim=128` 下的线程浪费仍然存在，动态 block size 修复有效。

## Moore S5000 参考

Moore 参考脚本为：

```text
/workspace_codex/InfiniCore/fix-rmsnorm/bench-qwen3-rmsnorm.sh
```

其测试方法为 BF16、seed 0、warmup 20、100 iterations、前后同步、先校验再计时，并输出 InfiniCore、PyTorch、ratio 和 Gelem/s。

主要用例：

```text
q_norm: (4096, 32, 128)
k_norm: (4096, 8, 128)
```

固定元素数 dim sweep 还包括：

```text
(4096, 32, 128)
(2048, 32, 256)
(1024, 32, 512)
(512, 32, 1024)
(256, 32, 2048)
(128, 32, 4096)
```

Moore S5000 参考结果：

| 用例 | 修复前 | 修复后 | 提升 |
| --- | ---: | ---: | ---: |
| q_norm | 7.980 ms | 1.108 ms | 7.20x |
| k_norm | 2.0006 ms | 0.2802 ms | 7.14x |


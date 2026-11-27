# InfiniCore MetaX RMSNorm Dynamic Block-Size Fix

更新时间：2026-11-27（Asia/Shanghai）

测试时间：2026-11-26（Asia/Shanghai）

## 内容

本目录记录 InfiniCore MetaX C550 RMSNorm 算子的动态 block-size 修复、数值验证和 q_norm/k_norm A/B 性能测试结果。

| 文件 | 说明 |
| --- | --- |
| `infinicore-metax-rmsnorm-dynamic-block-20261126.patch` | InfiniCore 源码与测试用例补丁 |
| `metax-c550-rmsnorm-dynamic-block-20261126.csv` | q_norm/k_norm A/B 性能数据 |
| `metax-c550-rmsnorm-dynamic-block-20261126.md` | 完整分析、构建、验证与测试报告 |
| `SHA256SUMS` | 本目录文件校验值 |

## 结果摘要

| 用例 | shape | fixed-1024 | dynamic-block | 提升 |
| --- | --- | ---: | ---: | ---: |
| q_norm | `(4096, 32, 128)` | 0.008242 ms | 0.006742 ms | 1.222x |
| k_norm | `(4096, 8, 128)` | 0.007198 ms | 0.006251 ms | 1.151x |

默认安装路径下执行：

```bash
cd /workspace_codex/InfiniCore
python test/infiniop/rms_norm.py --metax
```

全量 RMSNorm 数值测试通过。

## 应用补丁

补丁基于 InfiniCore `pd-v0.2.9` 分支生成，在仓库根目录执行：

```bash
cd /workspace_codex/InfiniCore
git apply fix-rmsnorm-metax/infinicore-metax-rmsnorm-dynamic-block-20261126.patch
```

当前工作区已经包含该修改时，可用反向校验确认补丁与源码一致：

```bash
git apply --check --reverse fix-rmsnorm-metax/infinicore-metax-rmsnorm-dynamic-block-20261126.patch
```

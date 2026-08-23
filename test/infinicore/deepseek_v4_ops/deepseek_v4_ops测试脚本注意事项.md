# deepseek_v4_ops 测试脚本注意事项

本文档用于指导后续在 `test/infinicore/deepseek_v4_ops` 目录下新增 DeepSeek V4 单算子测试脚本。当前目录中的四个脚本可以作为模板参考：

- `deepseek_v4_silu_and_mul.py`
- `deepseek_v4_rmsnorm_self.py`
- `deepseek_v4_mhc_post.py`
- `deepseek_v4_mhc_fused_post_pre.py`

## 目录层级

新增测试脚本统一放在：

```text
test/infinicore/deepseek_v4_ops/
```

命名使用算子名：

```text
deepseek_v4_<op_name>.py
```

原则上一个文件只测试一个 public 算子，例如 `deepseek_v4_mhc_fused_post_pre.py` 只测试 `_infinicore.deepseek_v4_mhc_fused_post_pre_`。如果需要测试 `_aten_`、`_dispatcher_`、`_kernel_` 等内部或对照变体，应在脚本中明确把它们作为 reference 或 baseline，不要把多个 public 算子的测试混在同一个文件里。

## 文件组织

脚本整体结构建议保持如下顺序：

```text
import 区域
DEFAULT_TOKENS
解析和 dtype helper
tensor 构造 helper
reference helper
diff/allclose helper
_bench
_run_case
_print_header / _print_row
main
```

标准 import 写法：

```python
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
from infinicore.lib import _infinicore
```

默认 tokens 范围使用：

```python
DEFAULT_TOKENS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"
```

## 函数命名

公共 helper 命名尽量与现有脚本一致：

- `_parse_tokens(text)`：解析 `--tokens` 参数。
- `_torch_dtype(name)`：需要多 dtype 时使用。
- `_dtype_names(name)`：支持 `--dtype all` 时使用。
- `_as_core(tensor)`：统一调用 `infinicore.from_torch(tensor)`。
- `_make_input(...)` 或 `_make_inputs(...)`：构造输入 tensor。
- `_make_outputs(...)`：多输出算子使用。
- `_max_diff(lhs, rhs)`：返回 `max_abs` 和 `max_rel`。
- `_tuple_max_diff(...)`、`_tuple_allclose(...)`：多输出算子使用。
- `_bench(fn, warmup, iters)`：统一性能计时。
- `_run_case(...)`：单个 tokens/shape case 的构造、计时、精度对比。
- `_print_header(...)`、`_print_row(result)`：统一输出格式。

reference helper 的命名要体现来源：

- `_aten_ref`：ATen 实现作为 reference。
- `_naive_ref`：已有 naive 算子作为 reference。
- `_sglang_dispatcher_ref`：SGLang dispatcher bridge 作为 reference。
- `_ref_separate`：用多个已验证算子串联作为 fused 算子的 reference。

## 参数风格

所有 deepseek_v4_ops 测试脚本建议保留这些通用参数：

```python
parser.add_argument("--hygon", action="store_true")
parser.add_argument("--nvidia", action="store_true")
parser.add_argument("--tokens", default=DEFAULT_TOKENS)
parser.add_argument("--hidden", type=int, default=<op_default_hidden>)
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--iters", type=int, default=40)
parser.add_argument("--seed", type=int, default=<fixed_seed>)
parser.add_argument("--atol", type=float, default=2e-2)
parser.add_argument("--rtol", type=float, default=2e-2)
```

其中：

- `--hygon` 和 `--nvidia` 作为平台选择/兼容参数保留，即使当前脚本内部不分支，也方便命令行风格一致。
- `--seed` 必须保留，便于复现精度和性能异常。
- op 自身参数按需添加，例如 `--dtype`、`--eps`、`--hc`、`--rms-eps`、`--sinkhorn-repeat` 等。
- 默认超参数优先贴近 DeepSeek V4 推理真实 shape，例如 MHC 相关脚本默认 `hidden=4096`、`hc=4`。

## 随机输入

每个 `_run_case` 内部设置随机种子：

```python
torch.manual_seed(args.seed + tokens * 17 + hidden)
```

如果算子的输入还强依赖 `dtype`、`hc` 或其他 shape 参数，可以在这个基础上加入额外 offset，但要保持可复现。

输入和输出 tensor 默认都应是 contiguous，除非该脚本专门测试非连续 tensor 行为。构造 InfiniCore tensor 时统一使用：

```python
core_x = _as_core(x)
```

调用 `_infinicore` 算子时传入：

```python
core_x._underlying
```

## 性能计时

`_bench` 统一使用 `torch.cuda.Event` 记录 GPU 时间：

```python
def _bench(fn, warmup, iters):
    warmup_value = None
    for _ in range(warmup):
        warmup_value = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    return {
        "avg_ms": total_ms / iters,
        "total_ms": total_ms,
        "warmup_value": warmup_value,
    }
```

注意事项：

- 不使用 `time.perf_counter()` 统计 GPU kernel 时间。
- `start.record()` 放在计时循环外，`end.record()` 放在计时循环后。
- 默认不打印 `median_ms`，除非用户明确要求中位数。
- 默认不做 CSV 输出，保持命令行输出简洁。
- timed section 内不要加入 CPU 同步、CPU copy 或打印。

## 精度检查

每个 case 都需要做精度检查，不再额外提供 `--check` 或 `--no-check` 参数。

推荐做法：

1. reference 和待测 op 都通过 `_bench` 执行。
2. 使用 `_bench` 返回的 `warmup_value` 做精度对比。
3. 打印 `max_abs`、`max_rel`、`allclose`。
4. 任意 case 失败时设置 `ok = False`，脚本最后 `raise SystemExit(1)`。

单输出算子使用：

```python
max_abs, max_rel = _max_diff(got, ref)
allclose = torch.allclose(got.float(), ref.float(), atol=args.atol, rtol=args.rtol)
```

多输出算子使用 tuple helper，输出顺序必须固定，并且 reference 与 op 的 tuple 顺序一致。

## 输出格式

输出风格保持表格化。header 先打印当前 shape/dtype 信息，例如：

```text
dtype=bf16 hidden=1536 input_hidden=3072
tokens | ref avg | op avg | speedup | max_abs | max_rel | allclose
```

列名根据 reference 来源调整：

- ATen reference：`aten avg` / `kernel avg`
- naive reference：`naive avg` / `kernel avg`
- dispatcher reference：`disp avg` / `kernel avg`
- 多算子串联 reference：`ref avg` / `op avg`

新增脚本默认应包含：

- `tokens`
- reference 平均耗时
- op 平均耗时
- `speedup`
- `max_abs`
- `max_rel`
- `allclose`

## Reference 选择

reference 的优先级按语义可靠性选择：

1. 已验证 ATen/naive 算子。
2. 已验证 dispatcher/SGLang bridge。
3. 已验证的多个 public 算子串联。
4. 纯 PyTorch 公式实现。

如果 fused 算子的 reference 是多个算子串联，应像 `deepseek_v4_mhc_fused_post_pre.py` 一样命名为 `_ref_separate`，并在 `_run_case` 中分别 benchmark reference 链路和 fused op。

## 不建议的写法

- 不依赖 `python/infinicore/ops/deepseek_v4_<op>.py` 包装层，优先直接通过 `_infinicore.deepseek_v4_<op>_` 调用。
- 不在同一个脚本中混合 graph 测试；graph 测试需要单独命名和说明。
- 不默认生成 CSV。
- 不默认打印 median。
- 不保留长期不用的调试参数。
- 不在 timed section 里调用 `.cpu()`、`.item()`、`print()` 或显式 `torch.cuda.synchronize()`。
- 不把 correctness-only 脚本和 perf 脚本拆得过细；该目录当前风格是同一个脚本同时覆盖精度和不同 tokens 下性能。

## 运行方式

在 Hygon/CoreX 环境中运行前，需要先加载环境：

```bash
source ~/.bashrc
source /.myenv.sh
cd /workspace_codex/InfiniCore
python test/infinicore/deepseek_v4_ops/deepseek_v4_<op_name>.py --hygon
```

如果只想快速检查小 shape：

```bash
python test/infinicore/deepseek_v4_ops/deepseek_v4_<op_name>.py --hygon --tokens 1,2,4 --iters 10 --warmup 1
```

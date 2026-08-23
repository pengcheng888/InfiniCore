# deepseek_v4_ops 注意事项

本文档参考当前 `src/infinicore/deepseek_v4_ops` 目录下已有的三个算子：

- `deepseek_v4_silu_and_mul`
- `deepseek_v4_rmsnorm_self`
- `deepseek_v4_mhc_fused_post_pre`

后续新增 DeepSeek V4 专用算子时，优先沿用这里的目录层级、命名方式和 graph 接入风格。

## 目录层级

推荐每个算子独立一个目录：

```text
src/infinicore/deepseek_v4_ops/<op_name>/
  <op_name>.cc
  kernel/
    <op_name>.cc
    <op_name>_kernel.cu
    <op_name>_kernel.hpp
  aten/
    <op_name>_aten.cc
  dispatcher/
    <op_name>_dispatcher.cc
```

其中 `aten/` 和 `dispatcher/` 是可选目录：

- 有 ATen naive/reference 实现时，放到 `aten/`。
- 需要通过 `c10::Dispatcher` 桥接 SGLang/Torch schema 时，放到 `dispatcher/`。
- 只有 native kernel 时，不需要为了形式统一强行创建空目录。

当前已有示例：

```text
deepseek_v4_mhc_fused_post_pre/
  deepseek_v4_mhc_fused_post_pre.cc
  aten/deepseek_v4_mhc_fused_post_pre_aten.cc
  kernel/deepseek_v4_mhc_fused_post_pre.cc
  kernel/deepseek_v4_mhc_fused_post_pre_kernel.cu
  kernel/deepseek_v4_mhc_fused_post_pre_kernel.hpp

deepseek_v4_rmsnorm_self/
  deepseek_v4_rmsnorm_self.cc
  aten/deepseek_v4_rmsnorm_self_aten.cc
  kernel/deepseek_v4_rmsnorm_self.cc
  kernel/deepseek_v4_rmsnorm_self_kernel.cu
  kernel/deepseek_v4_rmsnorm_self_kernel.hpp

deepseek_v4_silu_and_mul/
  deepseek_v4_silu_and_mul.cc
  dispatcher/deepseek_v4_silu_and_mul_dispatcher.cc
  kernel/deepseek_v4_silu_and_mul.cc
  kernel/deepseek_v4_silu_and_mul_kernel.cu
  kernel/deepseek_v4_silu_and_mul_kernel.hpp
```

## 源码职责

根目录 `<op_name>.cc` 是 public API 入口，职责尽量轻：

- 实现 `deepseek_v4_<op>` 返回 Tensor 版本，如果该算子天然适合返回单个 Tensor。
- 实现 `deepseek_v4_<op>_` out 参数版本。
- 默认转发到 `deepseek_v4_<op>_kernel_`。
- 不在这里写 graph plan/run/cleanup 细节。

`kernel/<op_name>.cc` 是 native kernel 的 C++ 包装层：

- 定义 graph class 的构造函数和 `execute`。
- 定义 `plan`、`run`、`cleanup` 和 `INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE`。
- 做 shape、dtype、device、contiguous 校验。
- 调用 `<op_name>_kernel.cu` 暴露的 `launch_*` 函数。
- native kernel launch 使用 `context::getStream()`。

`kernel/<op_name>_kernel.cu` 放设备端 kernel 和 launch 实现。

`kernel/<op_name>_kernel.hpp` 只声明 C++ 包装层需要调用的 launch 函数，不放 public API。

`aten/<op_name>_aten.cc` 只放 ATen reference 或 naive 实现：

- 命名为 `deepseek_v4_<op>_aten` / `deepseek_v4_<op>_aten_`。
- 可用于精度对比和诊断。
- 不是 InfiniLM 热路径。
- ATen/Torch 路径需要保持当前 stream，对 Hygon/NVIDIA 分别使用合适的 stream guard。

`dispatcher/<op_name>_dispatcher.cc` 只放 dispatcher bridge：

- 命名为 `deepseek_v4_<op>_dispatcher_`。
- 用于桥接 SGLang/Torch schema 或 vendor 扩展。
- 不作为默认热路径，除非明确确认性能、graph 和 stream 行为都安全。

## 函数命名

对外 public API 统一使用 `deepseek_v4_` 前缀：

```cpp
Tensor deepseek_v4_<op>(...);
void deepseek_v4_<op>_(Tensor out, ...);
void deepseek_v4_<op>_kernel_(Tensor out, ...);
void deepseek_v4_<op>_aten_(Tensor out, ...);
void deepseek_v4_<op>_dispatcher_(Tensor out, ...);
```

命名约定：

- 无后缀版本返回新 Tensor，内部申请输出。
- `_` 后缀版本使用外部传入的输出或 workspace。
- `_kernel_` 表示 native kernel / graph-aware 路径。
- `_aten_` 表示 ATen reference 路径。
- `_dispatcher_` 表示 `c10::Dispatcher` 桥接路径。
- 不再使用 `_naive_` 作为新接入算子的推荐命名；如果是 ATen naive/reference，优先命名为 `_aten_`。

多输出或 workspace 型算子可以只提供 out 参数版本。例如 `deepseek_v4_mhc_fused_post_pre_` 需要外部传入 `residual_cur/post_mix_cur/comb_mix_cur/layer_input_cur`，不适合强行提供单 Tensor 返回版本。

## Graph 命名和组织

graph class 放在：

```cpp
namespace infinicore::op {
namespace deepseek_v4 {

INFINICORE_GRAPH_OP_CLASS(OpClass, ...);

} // namespace deepseek_v4
} // namespace infinicore::op
```

实现文件也保持同样结构：

```cpp
namespace infinicore::op {

namespace deepseek_v4 {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(OpClass);

OpClass::OpClass(...) {
    INFINICORE_GRAPH_OP_DISPATCH(...);
}

void OpClass::execute(...) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(OpClass, ...);
}

namespace deepseek_v4_<op>_impl {
struct PlannedMeta { ... };
void *plan(...);
void run(void *planned_meta);
void cleanup(void **planned_meta_ptr);
} // namespace deepseek_v4_<op>_impl

namespace deepseek_v4_<op>_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(OpClass,
                                       &deepseek_v4_<op>_impl::plan,
                                       &deepseek_v4_<op>_impl::run,
                                       &deepseek_v4_<op>_impl::cleanup);
} // namespace deepseek_v4_<op>_register

} // namespace deepseek_v4

void deepseek_v4_<op>_kernel_(...) {
    deepseek_v4::OpClass::execute(...);
}

} // namespace infinicore::op
```

风格建议：

- graph class 名使用 PascalCase。
- 新算子优先使用语义化短名，例如 `SiluAndMul`、`RmsnormSelf`。
- 已有较长类名如 `DeepseekV4MhcFusedPostPre` 可以保留，但也应放入 `infinicore::op::deepseek_v4` 命名空间。
- `plan` 中保存 graph replay 需要的 tensor、shape、dtype、常量参数和 workspace。
- `run` 中只做底层 launch，避免执行期临时申请 tensor。
- 不在 graph hot path 中保留调试环境变量、`fprintf`、同步或 CPU round trip。

## 参数和校验

kernel wrapper 应在进入 graph class 前后保持清晰校验：

- device：Hygon build 检查 HYGON，NVIDIA build 检查 NVIDIA。
- dtype：显式检查 BF16/F16/F32 等支持类型。
- shape：错误信息包含算子名和参数含义。
- contiguous：当前 native kernel 如果只支持连续 tensor，要明确报错。
- 多 tensor 输入使用 `INFINICORE_ASSERT_TENSORS_SAME_DEVICE` 保证设备一致。

对于 DeepSeek V4 固定 shape 特化，可以在 public kernel 入口处做限制，例如：

```cpp
if (!use_dsv4_fixed_shape(...)) {
    throw std::runtime_error("deepseek_v4_<op>_kernel_ expects standard DeepSeek V4 shape.");
}
```

如果未来需要通用 fallback，优先在 InfiniCore 内部选择，不要把特化判断泄漏到 InfiniLM 模型层。

## Include 和 pybind

每个 public 算子需要在 `include/infinicore/ops/<op_name>.hpp` 声明对外函数和 graph class。

Python 绑定放在：

```text
src/infinicore/pybind11/ops/<op_name>.hpp
```

并在：

```text
src/infinicore/pybind11/ops.hpp
```

中包含对应 pybind 文件。

测试脚本优先直接使用：

```python
from infinicore.lib import _infinicore
```

然后调用 `_infinicore.deepseek_v4_<op>_...`。不再为每个 DeepSeek V4 专用算子额外新增 `python/infinicore/ops/<op_name>.py` 包装文件，除非确实需要 Python 层 API。

## xmake 接入

当前 `xmake.lua` 已收录：

```lua
add_files("src/infinicore/deepseek_v4_ops/*/*.cc")
add_files("src/infinicore/deepseek_v4_ops/*/*/*.cc")
```

因此新增 `.cc` 文件通常不需要单独加规则。

但 `.cu` 文件需要在 Hygon/NVIDIA 对应分支显式加入，例如：

```lua
add_files("src/infinicore/deepseek_v4_ops/<op_name>/kernel/*.cu")
```

新增 native kernel 算子时，不要忘记同步检查 Hygon 和 NVIDIA 两个构建分支。

## 测试组织

DeepSeek V4 专用测试放在：

```text
test/infinicore/deepseek_v4_ops/
```

推荐每个算子一个测试文件：

```text
test/infinicore/deepseek_v4_ops/deepseek_v4_<op>.py
```

测试文件建议同时覆盖：

- 精度：kernel/public 与 aten/dispatcher/torch reference 对比。
- 性能：覆盖 DeepSeek V4 推理常见 tokens 范围。
- 默认参数：尽量贴近 DeepSeek V4 config 和 InfiniLM 实际调用 shape。
- 输出：打印 backend、tokens、avg/median/total、speedup 和精度误差。

如果测试调用 Hygon/CoreX 运行时，执行前需要：

```bash
source ~/.bashrc
source /.myenv.sh
```

## 整体风格

新增算子时遵循以下原则：

- public API 简洁，复杂实现下沉到 `kernel/aten/dispatcher`。
- native kernel 是默认热路径；ATen/dispatcher 是 reference、兼容或诊断路径。
- graph 支持在 InfiniCore 算子内部完成，不在 InfiniLM 模型层写 graph 判断。
- device kernel 使用当前 InfiniCore stream，不创建默认流或外部流。
- 避免在 hot path 中使用 Python、GIL、ATen 临时 tensor、CPU 同步和调试打印。
- workspace 尽量外部传入或在 graph `plan` 阶段规划，避免 replay/run 阶段反复申请。
- 文件名、函数名、测试名保持和算子名一致，便于 `rg deepseek_v4_<op>` 直接定位完整链路。


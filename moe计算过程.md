
> 分析下 infinicore::op::deepseek_v4_shared_experts_impl_int8_marlin_prepare_metadata_和  infinicore::op::deepseek_v4_shared_experts_impl_int8_marlin_workspace_函数内部调用过程。关系函数的输入参数，输出参数，数据类型。
>
> 请分析。

<details><summary>6 previous messages</summary>

> 我会用 Hygon/InfiniLM DeepSeek V4 这套 skill 来梳理这两个 shared experts int8 marlin 接口。先定位声明、pybind 和实现，再按“输入/输出/workspace/数据类型/调用链”拆开说明。
>
> <details><summary>Explored 2 files, 2 searches</summary>
>
> - Read `/root/.codex/skills/hygon-bw1000-infinilm-deepseek-v4-0727/SKILL.md`
> - Read `/root/.codex/skills/hygon-bw1000-infinilm-deepseek-v4-0727/references/infinicore-deepseek-v4-ops.md`
> - Searched for `deepseek_v4_shared_experts_impl_int8_marlin_(prepare_metadata|workspace)|shared_experts_impl_int8_marlin` in `include`
> - Searched for `shared_experts_impl_int8_marlin_(prepare_metadata|workspace)|forward_packed|DeepseekV4PackedMLP|shared_experts` in `models`
>
> </details>
>
> 我找到了 InfiniLM 里 `DeepseekV4PackedMLP::forward_packed` 的调用点：它会先 prepare metadata，再把这些 metadata 和 compute workspace 传给 workspace 版本。接下来打开实现细节，把参数类型和内部调用链逐项对应。
>
> <details><summary>Explored 4 files</summary>
>
> - Read `./include/infinicore/ops/deepseek_v4_shared_experts_impl_int8_marlin.hpp`
> - Read `./src/infinicore/ops/deepseek_v4_shared_experts_impl_int8_marlin/deepseek_v4_shared_experts_impl_int8_marlin.cc`
> - Read `./src/infinicore/ops/deepseek_v4_shared_experts_impl_int8_marlin/deepseek_v4_shared_experts_impl_int8_marlin.cc`
> - Read `./csrc/models/deepseek_v4/deepseek_v4_mlp.cpp`
>
> </details>
>
> 我已经有了主流程，还需要后半段 graph/workspace wrapper 和底层 metadata kernel，这样能把 prepare 与 workspace 的关系讲完整。
>
> <details><summary>Explored 4 files</summary>
>
> - Read `./src/infinicore/ops/deepseek_v4_shared_experts_impl_int8_marlin/deepseek_v4_shared_experts_impl_int8_marlin.cc`
> - Read `./src/infinicore/ops/deepseek_v4_shared_experts_impl_int8_marlin/deepseek_v4_shared_experts_impl_int8_marlin_kernel.cu`
> - Read `./src/infinicore/pybind11/ops/deepseek_v4_shared_experts_impl_int8_marlin.hpp`
> - Read `./test/infinicore/ops/deepseek_v4_shared_experts_int8_marlin.py`
>
> </details>
</details>

这两个函数是一组配套接口：

- `prepare_metadata_`：只负责预生成 shared expert 的 MoE metadata。
- `workspace_`：使用外部传入的 metadata 和 compute workspace，执行完整 shared MLP 计算。

**1. prepare_metadata_**
签名在 [deepseek_v4_shared_experts_impl_int8_marlin.hpp](./InfiniCore/include/infinicore/ops/deepseek_v4_shared_experts_impl_int8_marlin.hpp:52)：

```cpp
void deepseek_v4_shared_experts_impl_int8_marlin_prepare_metadata_(
    Tensor sorted_token_ids,
    Tensor expert_ids,
    Tensor num_tokens_post_pad,
    Tensor topk_weights,
    size_t tokens);
```

输入/输出关系：

| 参数 | 角色 | shape | dtype | 说明 |
|---|---|---:|---|---|
| `sorted_token_ids` | 输出 workspace | `[padded_tokens]` | `I32` | token 排序结果。真实 token 写 `0..tokens-1`，padding 写 sentinel `tokens` |
| `expert_ids` | 输出 workspace | `[padded_tokens / 16]` | `I32` | 每个 block 对应 expert id；shared expert 只有 1 个，所以全部写 `0` |
| `num_tokens_post_pad` | 输出 workspace | `[1]` | `I32` | 写入 padding 后 token 数 |
| `topk_weights` | 输出 workspace | `[tokens, 1]` | `F32` | shared expert 权重，真实 token 全部写 `1.0` |
| `tokens` | 输入标量 | - | `size_t` | 当前 token 数 |

内部调用过程在 [deepseek_v4_shared_experts_impl_int8_marlin.cc](./InfiniCore/src/infinicore/ops/deepseek_v4_shared_experts_impl_int8_marlin/deepseek_v4_shared_experts_impl_int8_marlin.cc:650)：

1. 检查 tensor shape/dtype/contiguous。
2. 计算：
   ```cpp
   flat_topk = tokens * 1
   padded_tokens = ceil((flat_topk + 15) / 16) * 16
   ```
3. 调用 `fill_single_expert_metadata_`。
4. 最终 launch kernel：
   [deepseek_v4_shared_experts_impl_int8_marlin_kernel.cu](./InfiniCore/src/infinicore/ops/deepseek_v4_shared_experts_impl_int8_marlin/deepseek_v4_shared_experts_impl_int8_marlin_kernel.cu:11)

这个函数本质上是在构造“单 shared expert、topk=1”的固定 metadata，给后面的 lightop MoE Marlin GEMM 复用。

**2. workspace_**
签名在 [deepseek_v4_shared_experts_impl_int8_marlin.hpp](./InfiniCore/include/infinicore/ops/deepseek_v4_shared_experts_impl_int8_marlin.hpp:58)：

```cpp
void deepseek_v4_shared_experts_impl_int8_marlin_workspace_(
    Tensor output,
    const Tensor &hidden_states,
    const Tensor &w1,
    const Tensor &w2,
    const Tensor &w1_scale,
    const Tensor &w2_scale,
    Tensor sorted_token_ids,
    Tensor expert_ids,
    Tensor num_tokens_post_pad,
    Tensor topk_weights,
    Tensor q_hidden,
    Tensor hidden_scale,
    Tensor gate_up,
    Tensor q_activated,
    Tensor activated_scale,
    int gemm1_mode = -1,
    int gemm2_mode = -1,
    int delta = 1);
```

主要输入：

| 参数 | shape | dtype | 说明 |
|---|---:|---|---|
| `hidden_states` | `[tokens, hidden]` | `BF16` | shared MLP 输入 |
| `w1` | `[1, hidden/64, 2*intermediate*64]` | `I8` | Marlin layout 的 gate/up packed 权重 |
| `w2` | `[1, intermediate/64, hidden*64]` | `I8` | Marlin layout 的 down 权重 |
| `w1_scale` | `[1, 2*intermediate, 1]` | 通常 `F32` | w1 per-channel scale |
| `w2_scale` | `[1, hidden, 1]` | 通常 `F32` | w2 per-channel scale |

主要输出：

| 参数 | shape | dtype | 说明 |
|---|---:|---|---|
| `output` | `[tokens, hidden]` | `BF16` | shared MLP 输出 |

metadata workspace：

| 参数 | shape | dtype |
|---|---:|---|
| `sorted_token_ids` | `[padded_tokens]` | `I32` |
| `expert_ids` | `[padded_tokens / 16]` | `I32` |
| `num_tokens_post_pad` | `[1]` | `I32` |
| `topk_weights` | `[tokens, 1]` | `F32` |

compute workspace：

| 参数 | shape | dtype | 说明 |
|---|---:|---|---|
| `q_hidden` | `[tokens, hidden]` | `I8` | hidden_states 动态量化结果 |
| `hidden_scale` | `[tokens, 1]` | `F32` | hidden_states per-token scale |
| `gate_up` | `[tokens, 1, 2*intermediate]` | `BF16` | 第一段 GEMM 输出 |
| `q_activated` | `[tokens, intermediate]` | `I8` | `silu(gate) * up` 后量化结果 |
| `activated_scale` | `[tokens, 1]` | `F32` | activation per-token scale |

内部主流程在 [deepseek_v4_shared_experts_impl_int8_marlin.cc](./InfiniCore/src/infinicore/ops/deepseek_v4_shared_experts_impl_int8_marlin/deepseek_v4_shared_experts_impl_int8_marlin.cc:239)：

1. 检查 shape/dtype/contiguous。
2. 根据 shape 选择 `gemm1_mode/gemm2_mode`。自动模式目前只支持 `hidden=4096, intermediate=256`。
3. 因为 workspace 版本传入 `metadata_ready=true`，所以不会重新生成 metadata。
4. 调用 `launch_per_token_quant_int8_bf16`：
   ```text
   hidden_states BF16 -> q_hidden I8 + hidden_scale F32
   ```
5. 调用第一次 `deepseek_v4_lightop_moe_gemm_marlin_w8a8_`：
   ```text
   q_hidden I8 × w1 I8 -> gate_up BF16
   ```
6. 调用 `deepseek_v4_lightop_fuse_silu_mul_quant_`：
   ```text
   gate_up BF16 -> silu(gate) * up -> q_activated I8 + activated_scale F32
   ```
7. 调用第二次 `deepseek_v4_lightop_moe_gemm_marlin_w8a8_`：
   ```text
   q_activated I8 × w2 I8 -> output BF16
   ```

**两者关系**
在 InfiniLM 里调用点是 [deepseek_v4_mlp.cpp](./InfiniLM/csrc/models/deepseek_v4/deepseek_v4_mlp.cpp:136)：

1. `forward_packed` 先申请 scratch。
2. 如果 `metadata_ready_ == false` 或 token 数变化，调用：
   ```cpp
   deepseek_v4_shared_experts_impl_int8_marlin_prepare_metadata_(...)
   ```
3. 每次 forward 都调用：
   ```cpp
   deepseek_v4_shared_experts_impl_int8_marlin_workspace_(...)
   ```

所以设计意图是：

```text
prepare_metadata_：低频执行，只在 tokens 变化时刷新 metadata
workspace_：高频执行，每次 forward 复用外部 workspace 和 metadata
```

这比原始 `deepseek_v4_shared_experts_impl_int8_marlin_` 更适合推理热路径，因为避免了内部反复申请 workspace，也避免了每次重复构造固定 topk=1 metadata。

> 分析下 infinicore::op::deepseek_v4_fused_experts_impl_int8_marlin_函数内部调用过程。关键函数的输入参数，输出参数，数据类型。
>
> 请分析。

<details><summary>6 previous messages</summary>

> 我会继续用 Hygon/InfiniLM DeepSeek V4 skill 来看这个 fused routed experts 算子。先定位 public 签名、pybind、主实现和 kernel/SO bridge，然后按调用链把关键输入输出和 dtype 拆开。
>
> <details><summary>Explored 2 files, 2 searches</summary>
>
> - Read `/root/.codex/skills/hygon-bw1000-infinilm-deepseek-v4-0727/SKILL.md`
> - Read `/root/.codex/skills/hygon-bw1000-infinilm-deepseek-v4-0727/references/infinicore-deepseek-v4-ops.md`
> - Searched for `deepseek_v4_fused_experts_impl_int8_marlin_|fused_experts_impl_int8_marlin` in `include`
> - Searched for `fused_experts_impl_int8_marlin|routed_expert|RoutedExpert|experts_impl` in `deepseek_v4`
>
> </details>
>
> 我现在打开主实现，关注 `deepseek_v4_fused_experts_impl_int8_marlin_impl_` 内部如何串起 quant、align、两次 GEMM、activation 和 sum/add。
>
> <details><summary>Explored 4 files</summary>
>
> - Read `./include/infinicore/ops/deepseek_v4_fused_experts_impl_int8_marlin.hpp`
> - Read `./src/infinicore/ops/deepseek_v4_fused_experts_impl_int8_marlin/deepseek_v4_fused_experts_impl_int8_marlin.cc`
> - Read `./src/infinicore/ops/deepseek_v4_fused_experts_impl_int8_marlin/deepseek_v4_fused_experts_impl_int8_marlin.cc`
> - Read `./src/infinicore/pybind11/ops/deepseek_v4_fused_experts_impl_int8_marlin.hpp`
>
> </details>
>
> 主链路已经明确了。还需要看自定义 kernel 里量化和 shared_output add 的 dtype/shape 细节，以及 InfiniLM 调用时传的参数来源。
>
> <details><summary>Explored 4 files</summary>
>
> - Read `./src/infinicore/ops/deepseek_v4_fused_experts_impl_int8_marlin/deepseek_v4_fused_experts_impl_int8_marlin.cc`
> - Read `./src/infinicore/ops/deepseek_v4_fused_experts_impl_int8_marlin/deepseek_v4_fused_experts_impl_int8_marlin_kernel.cu`
> - Read `./src/infinicore/ops/deepseek_v4_fused_experts_impl_int8_marlin/deepseek_v4_fused_experts_impl_int8_marlin_kernel.hpp`
> - Read `./csrc/models/deepseek_v4/moe_backends/routed_expert_backend.cpp`
>
> </details>
</details>

`infinicore::op::deepseek_v4_fused_experts_impl_int8_marlin_` 是 routed experts 的 fused INT8 Marlin MoE 路径。它内部不是一个单 kernel，而是串起了：

```text
topk metadata align
hidden BF16 -> INT8 per-token quant
expert gate/up GEMM
silu(gate) * up + INT8 quant
expert down GEMM
topk sum / scale / optional shared add
```

**Public 签名**
在 [deepseek_v4_fused_experts_impl_int8_marlin.hpp](./InfiniCore/include/infinicore/ops/deepseek_v4_fused_experts_impl_int8_marlin.hpp:28)：

```cpp
void deepseek_v4_fused_experts_impl_int8_marlin_(
    Tensor output,
    const Tensor &hidden_states,
    const Tensor &w1,
    const Tensor &w2,
    const Tensor &topk_weights,
    const Tensor &topk_ids,
    const Tensor &w1_scale,
    const Tensor &w2_scale,
    int64_t global_num_experts,
    double routed_scaling_factor = 1.0,
    bool inplace = false,
    const std::optional<Tensor> &shared_output = std::nullopt);
```

**输入/输出参数**
| 参数 | 角色 | shape | dtype | 说明 |
|---|---|---:|---|---|
| `output` | 输出 | `[tokens, hidden]` | 通常 `BF16` | 最终 routed experts 输出；如果 `inplace=true`，最后可能从 `hidden_states` copy 回 output |
| `hidden_states` | 输入 | `[tokens, hidden]` | `BF16` | MoE routed experts 输入 |
| `w1` | 输入 | `[experts, hidden/64, 2*intermediate*64]` | `I8` | Marlin layout 的 gate/up packed 权重 |
| `w2` | 输入 | `[experts, intermediate/64, hidden*64]` | `I8` | Marlin layout 的 down 权重 |
| `topk_weights` | 输入 | `[tokens, topk]` | `F32` | router topk 权重 |
| `topk_ids` | 输入 | `[tokens, topk]` | `I32` 常见 | 每个 token 选中的 expert id |
| `w1_scale` | 输入 | `[experts, 2*intermediate, 1]` | 通常 `F32` | w1 per-channel scale |
| `w2_scale` | 输入 | `[experts, hidden, 1]` | 通常 `F32` | w2 per-channel scale |
| `global_num_experts` | 输入标量 | - | `int64_t` | expert 总数，DeepSeek V4 通常 256 |
| `routed_scaling_factor` | 输入标量 | - | `double` | 最后 routed output scale，DeepSeek V4 常见 1.5 |
| `inplace` | 输入标量 | - | `bool` | `true` 时 target output 使用 `hidden_states` |
| `shared_output` | 可选输入 | `[tokens, hidden]` | 同 `hidden_states` | 如果传入，在 fused sum kernel 中直接加 shared expert 输出 |

支持的固定形状在 [deepseek_v4_fused_experts_impl_int8_marlin.cc](./InfiniCore/src/infinicore/ops/deepseek_v4_fused_experts_impl_int8_marlin/deepseek_v4_fused_experts_impl_int8_marlin.cc:68)：
- `hidden=4096, intermediate=256, topk=6`
- `hidden=7168, intermediate=256, topk=8`

**内部 workspace**
如果是 graph plan，会提前创建 workspace；eager 时函数内部临时创建。定义在 [deepseek_v4_fused_experts_impl_int8_marlin.cc](./InfiniCore/src/infinicore/ops/deepseek_v4_fused_experts_impl_int8_marlin/deepseek_v4_fused_experts_impl_int8_marlin.cc:186)：

| workspace | shape | dtype | 说明 |
|---|---:|---|---|
| `sorted_token_ids` | `[max_num_tokens_padded]` | `I32` | topk token 按 expert/block 对齐后的 token id |
| `expert_ids` | `[ceil(max_num_tokens_padded / 16)]` | `I32` | 每个 block 对应 expert id |
| `num_tokens_post_pad` | `[1]` | `I32` | padding 后 token 数 |
| `q_hidden` | `[tokens, hidden]` | `I8` | hidden_states 动态量化结果 |
| `hidden_scale` | `[tokens, 1]` | `F32` | hidden per-token scale |
| `gate_up` | `[tokens, topk, 2*intermediate]` | `BF16` | 第一段 expert GEMM 输出 |
| `q_activated` | `[tokens*topk, intermediate]` | `I8` | activation 后动态量化结果 |
| `activated_scale` | `[tokens*topk, 1]` | `F32` | activation per-token scale |
| `down` | `[tokens, topk, hidden]` | `BF16` | 第二段 expert GEMM 输出 |

**内部调用过程**
主实现从 [deepseek_v4_fused_experts_impl_int8_marlin.cc](./InfiniCore/src/infinicore/ops/deepseek_v4_fused_experts_impl_int8_marlin/deepseek_v4_fused_experts_impl_int8_marlin.cc:237) 开始。

1. **shape/config 推导**
   ```cpp
   infer_fused_experts_shape(...)
   ```
   推出：
   ```text
   tokens, hidden, topk, num_experts, intermediate, gate_up_size, flat_topk, max_num_tokens_padded
   ```
   并选择 `gemm1_mode/gemm2_mode/delta`。

2. **topk metadata align**
   ```cpp
   deepseek_v4_lightop_moe_align_block_size_(
       topk_ids,
       num_experts,
       block_size,
       sorted_token_ids,
       expert_ids,
       num_tokens_post_pad,
       true);
   ```
   输入：`topk_ids [tokens, topk]`
   输出：`sorted_token_ids / expert_ids / num_tokens_post_pad`

3. **hidden per-token quant**
   ```cpp
   lmslim_per_token_quant_int8_bf16_(q_hidden, hidden_scale, hidden_states);
   ```
   底层 kernel 是 [launch_per_token_quant_int8_bf16](./InfiniCore/src/infinicore/ops/deepseek_v4_fused_experts_impl_int8_marlin/deepseek_v4_fused_experts_impl_int8_marlin_kernel.cu:86)。

   输入：
   ```text
   hidden_states: [tokens, hidden], BF16
   ```
   输出：
   ```text
   q_hidden:      [tokens, hidden], I8
   hidden_scale:  [tokens, 1], F32
   ```

4. **第一次 Marlin GEMM：gate/up**
   ```cpp
   deepseek_v4_lightop_moe_gemm_marlin_w8a8_(
       q_hidden,
       w1,
       gate_up,
       hidden_scale,
       w1_scale,
       nullopt,
       sorted_token_ids,
       expert_ids,
       num_tokens_post_pad,
       topk,
       gemm1_mode,
       delta);
   ```

   语义：
   ```text
   q_hidden I8 × w1 I8 -> gate_up BF16
   ```
   输出：
   ```text
   gate_up: [tokens, topk, 2*intermediate], BF16
   ```

5. **activation + quant**
   ```cpp
   deepseek_v4_lightop_fuse_silu_mul_quant_(
       q_activated,
       activated_scale,
       gate_up.view({tokens * topk, 2*intermediate}),
       nullopt,
       1,
       -1,
       nullopt);
   ```

   语义：
   ```text
   gate = gate_up[..., :intermediate]
   up   = gate_up[..., intermediate:]
   activated = silu(gate) * up
   activated BF16 -> q_activated I8 + activated_scale F32
   ```

   输出：
   ```text
   q_activated:     [tokens*topk, intermediate], I8
   activated_scale: [tokens*topk, 1], F32
   ```

6. **第二次 Marlin GEMM：down**
   ```cpp
   deepseek_v4_lightop_moe_gemm_marlin_w8a8_(
       q_activated,
       w2,
       down,
       activated_scale,
       w2_scale,
       topk_weights,
       sorted_token_ids,
       expert_ids,
       num_tokens_post_pad,
       1,
       gemm2_mode,
       delta);
   ```

   语义：
   ```text
   q_activated I8 × w2 I8 -> down BF16
   ```
   输出：
   ```text
   down: [tokens, topk, hidden], BF16
   ```

7. **topk reduce / scale / optional shared add**
   如果有 `shared_output`：
   ```cpp
   launch_moe_sum_scale_add_bf16(
       target_output,
       down,
       shared_output,
       tokens,
       topk,
       hidden,
       routed_scaling_factor);
   ```
   语义：
   ```text
   output[token, hidden] =
       sum_k down[token, k, hidden] * routed_scaling_factor
       + shared_output[token, hidden]
   ```

   如果没有 `shared_output`：
   ```cpp
   deepseek_v4_lightop_moe_sum_(
       target_output,
       down,
       nullopt,
       nullopt,
       nullopt,
       routed_scaling_factor,
       -1);
   ```

**Graph 路径**
`deepseek_v4_fused_experts_impl_int8_marlin_` 是 graph-aware。graph plan 会在 [FusedExpertsGraphMeta](./InfiniCore/src/infinicore/ops/deepseek_v4_fused_experts_impl_int8_marlin/deepseek_v4_fused_experts_impl_int8_marlin.cc:419) 中保存输入 tensor 和 workspace；run 阶段调用同一个 `deepseek_v4_fused_experts_impl_int8_marlin_impl_`，但复用 graph plan 中的 workspace。

**InfiniLM 调用**
InfiniLM routed expert 默认 fused backend 在 [routed_expert_backend.cpp](./InfiniLM/csrc/models/deepseek_v4/moe_backends/routed_expert_backend.cpp:182) 调用：

```cpp
deepseek_v4_fused_experts_impl_int8_marlin_(
    fused_output,
    hidden_states,
    ctx.w13_weight_marlin,
    ctx.w2_weight_marlin,
    topk_weights,
    topk_indices,
    ctx.w13_weight_scale,
    ctx.w2_weight_scale,
    ctx.num_experts,
    ctx.routed_scaling_factor,
    true,
    shared_output);
```

这里 `inplace=true`，并且如果 shared experts 已经提前算出 `shared_output`，会在 fused sum 阶段一起加进去，避免额外一次 `add_`。

> /workspace/sglang推理框架的offline_dsv4_tp_bench.py脚本执行了deepseek_v4模型的推理。请分析deepseek_v4中的linear的量化类型是什么，量化计算时调用的是哪些算子。

<details><summary>27 previous messages</summary>

> 我会使用 `hygon-bw1000-infinilm-deepseek-v4` 这个 skill，因为问题涉及 `/workspace_codex`、SGLang 的 DeepSeek-V4 推理路径、量化 linear 和算子调用链。先按 skill 读说明和相关环境文档，然后顺着脚本到模型/quantization 实现去定位。
>
> <details><summary>Explored a file, a search, a list, ran a command</summary>
>
> <details><summary>Explored 2 files</summary>
>
> - Read `./.codex/skills/hygon-bw1000-infinilm-deepseek-v4-0727/SKILL.md`
> - Read `./.codex/skills/hygon-bw1000-infinilm-deepseek-v4-0727/references/environment.md`
>
> </details>
>
> - Ran `git -C /workspace_codex/InfiniLM status --short`
>
> <details><summary>Explored a search, a list</summary>
>
> - Listed files in `workspace_codex`
> - Searched for `offline_dsv4_tp_bench.py|deepseek_v4|slimquant|quantization`
>
> </details>
>
> </details>
>
> 我已经看到这次 `InfiniLM` 仓库里有用户已有改动：`scripts/test_perf.py` 是 modified，我不会碰它。接下来我主要读 `/workspace/sglang` 的 SGLang 模型与量化实现，不做代码改动。
>
> <details><summary>Explored 2 files, 2 searches</summary>
>
> - Read `./.codex/skills/hygon-bw1000-infinilm-deepseek-v4-0727/references/architecture.md`
> - Searched for `offline_dsv4_tp_bench.py` in `workspace_codex`
> - Read `./.codex/skills/hygon-bw1000-infinilm-deepseek-v4-0727/references/infinicore-deepseek-v4-ops.md`
> - Searched for `slimquant|quantization|Linear|MergedColumnParallelLinear|RowParallelLinear|ReplicatedLinear|QKVParallelLinear|DeepseekV4|fused_experts|moe|marlin|lmslim` in `deepseek_v4.py`
>
> </details>
>
> `offline_dsv4_tp_bench.py` 在刚才的快速查找里没有直接命中，我会再缩小查找脚本名和启动参数；同时重点看 SGLang 的 `deepseek_v4.py`、linear 层、`slimquant_marlin` 量化实现和 fused MoE 路径。
>
> <details><summary>Explored 2 files, 2 searches</summary>
>
> - Searched for `*dsv4*bench*.py` in `workspace_codex`
> - Searched for `quantization=|--quantization|slimquant|Engine\(|model_path|DeepSeek-V4|DeepseekV4` in `workspace_codex`
> - Read `/workspace/sglang/python/sglang/srt/layers/quantization/slimquant_w4a8_marlin.py`
> - Read `/workspace/sglang/python/sglang/srt/layers/linear.py`
>
> </details>
>
> 现在能确认脚本族实际是 `offline_dsv4_tp_infer_v3.py`，它默认 `quantization=slimquant_marlin`；你说的 bench 脚本大概率是同一条 SGLang Engine 路径。我继续看 `slimquant_marlin` 这个名字如何映射到 linear/MoE 方法，以及 DeepSeek-V4 哪些 Linear 带 quant_config。
>
> <details><summary>Explored 4 files</summary>
>
> - Read `/workspace/sglang/python/sglang/srt/layers/quantization/slimquant_w4a8_marlin.py`
> - Read `/workspace/sglang/python/sglang/srt/layers/linear.py`
> - Read `/workspace/sglang/python/sglang/srt/layers/quantization/slimquant_w4a8.py`
> - Read `/workspace/sglang/python/sglang/srt/layers/quantization/__init__.py`
>
> </details>
>
> 有一个关键点：命令行里的 `slimquant_marlin` 不是直接等于 `slimquant_w4a8_marlin`，而是先走 `SlimQuantCompressedTensorsMarlinConfig`，再根据 HF checkpoint 的量化配置 override 到真正的 W4A8 Marlin。我要把这个 override 链条也确认清楚，避免把名字和实际量化类型混在一起。
>
> <details><summary>Explored 3 files, a search</summary>
>
> - Read `/workspace/sglang/python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_marlin.py`
> - Searched for `override_quantization_method|quant_method|slimquant|target_scheme_map|Auto-detected routed-expert|SGLANG_DSV4_FP4_EXPERTS|safetensors dtype` in `model_config.py`
> - Read `/workspace/sglang/python/sglang/srt/models/deepseek_v4.py`
> - Read `/workspace/sglang/python/sglang/srt/models/deepseek_v4.py`
>
> </details>
>
> 目前链路更清楚了：普通 Linear 走 compressed-tensors 的 LinearMethod，MoE expert 走 compressed-tensors 的 Marlin MoE method；另一个 `slimquant_w4a8_marlin.py` 是专门的 W4A8 Marlin MoE/Linear实现。我再读 compressed-tensors 里实际创建权重和 apply 的部分。
>
> <details><summary>Explored 3 files, a search</summary>
>
> - Read `/workspace/sglang/python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`
> - Read `/workspace/sglang/python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe_marlin.py`
> - Read `/workspace/sglang/python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe_marlin.py`
> - Searched for `def apply\(|scaled_mm|cutlass_scaled_mm|triton_scaled_mm|weight_scale|input_scale|scheme.strategy|QuantizationStrategy|process_weights_after_loading|create_weights` in `compressed_tensors.py`
>
> </details>
>
> MoE 的实际算子已经定位到 `torch.ops.sglang.fused_experts_impl_int8_marlin`，它包装 lmslim 的 `fused_experts_impl_int8_marlin`。普通 Linear 的 apply 还差 scheme 细节，我继续读 `CompressedTensorsLinearMethod.apply` 和 scheme 类。
>
> <details><summary>Explored 2 files, 2 searches</summary>
>
> - Read `/workspace/sglang/python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`
> - Searched for `class CompressedTensors.*Int8|def apply\(|triton_scaled_mm|scaled_mm|per_token_quant_int8|cutlass_scaled_mm|rocblas_scaled_mm|process_weights_after_loading|create_weights` in `compressed_tensors`
> - Read `/workspace/sglang/python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`
> - Searched for `class .*W8A8.*Int8|per_token_quant_int8|triton_scaled_mm|cutlass_scaled_mm|rocblas_scaled_mm|strategy.*TOKEN|strategy.*CHANNEL` in `schemes`
>
> </details>
>
> 普通 Linear 的 scheme 是 `CompressedTensorsW8A8Int8`，也就是权重 int8、激活动态 token int8。我再确认模型配置文件里的 compressed-tensors 参数，和 `offline` 脚本传入的量化参数是否一致。
>
> <details><summary>Explored 4 files</summary>
>
> - Read `/workspace/sglang/python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py`
> - Read `/workspace_codex/offline_dsv4_tp_infer_v3.py`
> - Read `/data/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8/config.json`
> - Read `/workspace/sglang/python/sglang/srt/layers/quantization/compressed_tensors/quant_ops.py`
>
> </details>
>
> 配置文件也确认了：HF `compression_config.quant_method=compressed-tensors`，weights 是 `int8/channel/static/symmetric`，input activations 是 `int8/token/dynamic/symmetric`。这就是普通 Linear 的 W8A8 动态 token 量化。现在我把 DeepSeek-V4 里哪些线性层使用这些方法和 MoE/shared 路径串起来。
>
> <details><summary>Explored 3 files, a search</summary>
>
> - Read `/workspace/sglang/python/sglang/srt/models/deepseek_v4.py`
> - Searched for `class DeepseekV2MoE|class DeepseekV2MLP|FusedMoE|gate_proj|up_proj|down_proj|shared_experts|ColumnParallelLinear|RowParallelLinear|ReplicatedLinear|MergedColumnParallelLinear` in `deepseek_v2.py`
> - Read `/workspace/sglang/python/sglang/srt/models/deepseek_v2.py`
> - Read `/workspace/sglang/python/sglang/srt/models/deepseek_v2.py`
>
> </details>
>
> DeepSeek-V4 继承/复用 DeepSeek-V2 的 MLP/MoE 结构：dense/shared MLP 是 `gate_up_proj + down_proj` 两个量化 Linear；routed experts 是 `FusedMoE`。我继续看 forward 里 shared_output 和 fused MoE 的调用，确认是否走 `apply_with_shared_output`。
>
> <details><summary>Explored 4 files</summary>
>
> - Read `/workspace/sglang/python/sglang/srt/models/deepseek_v2.py`
> - Read `/workspace/sglang/python/sglang/srt/models/deepseek_v2.py`
> - Read `/workspace/sglang/python/sglang/srt/models/deepseek_v2.py`
> - Read `/workspace/sglang/python/sglang/srt/models/deepseek_v2.py`
>
> </details>
>
> 补充一点：MoE gate 的 router linear 不是普通量化 Linear 类，而是直接 `F.linear`/专门 gate 路径，权重是 `config.n_routed_experts x hidden_size`。我读一下 gate forward，确认它不属于这次 linear quant 的 W8A8 GEMM。
>
> <details><summary>Explored 2 files, 2 searches</summary>
>
> - Read `/workspace/sglang/python/sglang/srt/models/deepseek_v2.py`
> - Searched for `def forward\(|linear\(|dsv3_router_gemm|gate` in `deepseek_v2.py`
> - Searched for `linear_bf16_fp32|def linear_bf16_fp32|torch.ops|sgl_kernel|quant_ops|deepseek_v4` in `deepseek_v4.py`
> - Read `/workspace/sglang/python/sglang/srt/models/deepseek_v2.py`
>
> </details>
>
> 我注意到之前运行日志里有 `Auto-detected routed-expert safetensors dtype=I8; SGLANG_DSV4_FP4_EXPERTS=True` 这种容易误导的日志。这个只影响 DeepSeek-V4 2604 的 FP4 expert 分支判断/wo_a 处理；就当前模型 `compression_config` 和实际 quant method 来看，linear/MoE 主计算仍是 INT8 W8A8 Marlin 路径。最后我再核对一下这段自动检测代码。
>
> <details><summary>Explored 2 files, 2 searches</summary>
>
> - Read `/workspace/sglang/python/sglang/srt/configs/model_config.py`
> - Read `/workspace/sglang/python/sglang/srt/configs/model_config.py`
> - Searched for `SGLANG_DSV4_FP4_EXPERTS|use_int4_w4a16|use_int4_w4a8|use_int8_w8a8|weight.dtype|uint8|int8` in `quantization`
> - Searched for `SGLANG_DSV4_FP4_EXPERTS|AUTO|I8|F4|FP4` in `environ.py`
>
> </details>
>
> 我已经拿到主要结论了。最后取一下关键代码行号，方便你后面直接跳到实现处看。
>
> <details><summary>Explored 4 files</summary>
>
> - Read `/workspace_codex/offline_dsv4_tp_infer_v3.py`
> - Read `/data/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8/config.json`
> - Read `/workspace/sglang/python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py`
> - Read `/workspace/sglang/python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe_marlin.py`
>
> </details>
>
> Context compaction
>
> Source: automatic
> Status: completed
</details>

结论：这个 DeepSeek-V4 跑的不是 FP8/FP4 Linear，而是 **compressed-tensors W8A8 int8 Linear**，也就是：

- **weight**：int8，静态量化，per-channel/channel scale
- **activation**：int8，动态量化，per-token/token scale
- **对称量化**：symmetric int8
- SGLang 侧通过 `--quantization slimquant_marlin` 进入对应实现

依据是脚本默认量化参数在 [offline_dsv4_tp_infer_v3.py](/workspace_codex/offline_dsv4_tp_infer_v3.py:594)，Engine 创建时传入 `quantization=args.quantization`，见 [offline_dsv4_tp_infer_v3.py](/workspace_codex/offline_dsv4_tp_infer_v3.py:666)。模型配置里 `compression_config.quant_method = compressed-tensors`，目标是 `Linear`，activation 是 `token int8 dynamic`，weight 是 `channel int8 static`，见 [/data/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8/config.json](/data/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8/config.json:55)。

**普通 Linear 调用路径**

普通的 `ColumnParallelLinear`、`RowParallelLinear`、`MergedColumnParallelLinear` 会走：

`CompressedTensorsLinearMethod.apply`
→ `CompressedTensorsW8A8Int8.apply_weights`
→ 先动态量化 activation：
`lmslim.layers.gemm.int8_utils.per_token_quant_int8`
→ 再做 int8 GEMM。

默认算子是：

```text
lmslim.quant_ops.triton_scaled_mm
```

代码位置在 [compressed_tensors_w8a8_int8.py](/workspace/sglang/python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py:179)。

同时这里受 `W8A8_SUPPORT_METHODS` 控制：

```text
1: ops.triton_scaled_mm -> lmslim.quant_ops.triton_scaled_mm   默认
2: ops.cutlass_scaled_mm -> lmslim.quant_ops.rocblas_scaled_mm_nn
3: ops.blaslt_scaled_mm -> lmslim.quant_ops.hipblaslt_w8a8_gemm
else: ops.rocblas_scaled_mm -> lmslim.quant_ops.rocblas_scaled_mm_nn
```

封装在 [quant_ops.py](/workspace/sglang/python/sglang/srt/layers/quantization/compressed_tensors/quant_ops.py:9)。

**MoE Expert Linear 调用路径**

DeepSeek-V4 的 routed experts 不是逐个普通 Linear GEMM，而是走 fused MoE int8 Marlin 路径。量化类型同样是 **W8A8 int8：weight per-channel，activation per-token dynamic**。

核心调用是：

```text
torch.ops.sglang.fused_experts_impl_int8_marlin
```

这个 custom op 注册后实际调用：

```text
lmslim.layers.fused_moe.fuse_moe_int8_marlin.fused_experts_impl_int8_marlin
```

相关实现见 [compressed_tensors_moe_marlin.py](/workspace/sglang/python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe_marlin.py:220) 和调用点 [compressed_tensors_moe_marlin.py](/workspace/sglang/python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe_marlin.py:395)。

**几个特例**

- `attn.wo_a` 在模型量化配置里被 `ignore: re:.*attn.wo_a.*` 排除了，所以默认不按这个 W8A8 compressed-tensors Linear 路径走。
- MoE gate/router 的打分不是普通 quantized Linear，而是走 `linear_bf16_fp32(hidden_states, self.weight)` 这一类 bf16/fp32 router 路径。
- `slimquant_w4a8_marlin` 代码虽然存在，但当前这个 checkpoint/config 是 `compressed-tensors` W8A8，不是 W4A8。只有显式切到 `slimquant_w4a8_marlin` 或 checkpoint 本身是对应格式时才会走 W4A8 Marlin。





> 查看/workspace_codex/InfiniCore/sglang的量化.md文件，这个文档中说hygon平台的/worksapce/sglang仓库里用到lmslim库做量化。这个lmslim和lightop二者是什么关系。

<details><summary>6 previous messages</summary>

> 我继续按 Hygon/DeepSeek V4 这条技能路径来查。这个问题的关键是把文档里说的 “lmslim 库” 和当前我们实际接入的 `lightop.op`/`op.so` 调用链对应起来。
>
> <details><summary>Explored a file</summary>
>
> - Read `/root/.codex/skills/hygon-bw1000-infinilm-deepseek-v4-0727/SKILL.md`
>
> </details>
>
> 我会先读你提到的量化文档，再在 `/workspace/sglang` 和当前 `lightop` 安装包里找 `lmslim`、`lightop`、`fused_experts_impl_int8_marlin` 的引用关系。
>
> <details><summary>Explored a file, 2 searches, ran a command</summary>
>
> <details><summary>Explored a file</summary>
>
> - Read `./InfiniCore/sglang的量化.md`
>
> </details>
>
> - Ran `rg -n "lmslim|lightop" /usr/local/lib/python3.10/dist-packages/lightop /usr/local/lib/python3.10/dist-packages/lmslim 2>/dev/null`
>
> <details><summary>Explored 2 searches</summary>
>
> - Searched for `lmslim|lightop|fused_experts_impl_int8_marlin|moe_gemm_marlin_w8a8|per_token_dynamic_quant_int8|dynamic_scaled_int8_quant` in `sglang`
> - Searched for `*lmslim*` in `dist-packages`
>
> </details>
>
> </details>
>
> 文档里的描述和当前安装包是一致的：SGLang 的量化层大量 import `lmslim`，但 `lmslim` 内部又会按环境变量决定是否调用 `lightop` 的高性能 kernel。接下来我看 `LMSLIM_USE_LIGHTOP` 默认值和 fused MoE/普通 W8A8 GEMM 的实际分支。
>
> <details><summary>Explored 6 files</summary>
>
> - Read `/usr/local/lib/python3.10/dist-packages/lmslim/envs.py`
> - Read `/usr/local/lib/python3.10/dist-packages/lmslim/layers/gemm/int8_utils.py`
> - Read `/usr/local/lib/python3.10/dist-packages/lmslim/layers/fused_moe/fuse_moe_int8_marlin.py`
> - Read `/usr/local/lib/python3.10/dist-packages/lmslim/layers/fused_moe/fuse_moe_int8_marlin.py`
> - Read `/usr/local/lib/python3.10/dist-packages/lmslim/quantize/quant_ops.py`
> - Read `/usr/local/lib/python3.10/dist-packages/lmslim/quantize/quant_ops.py`
>
> </details>
</details>

`lmslim` 和 `lightop` 是两个独立安装包，但在 Hygon 平台上经常是 **上下层配合关系**：

```text
SGLang
  -> lmslim: 量化方法/调度/包装层
      -> lightop: Hygon 高性能算子库/so kernel
```

更具体地说：

`lmslim` 更像“量化框架层”：
- 提供 W8A8/W4A8/FP8 等量化 linear、MoE 的 Python 逻辑。
- 提供 `per_token_quant_int8`、`triton_scaled_mm`、`fused_experts_impl_int8_marlin` 这类高层接口。
- 管理 MoE 的流程：activation quant、moe align、第一次 GEMM、activation、第二次 GEMM、moe sum。

`lightop` 更像“Hygon 优化算子库”：
- 提供 `lightop/op.cpython-310-x86_64-linux-gnu.so`。
- 里面有成熟 kernel：`moe_gemm_marlin_w8a8`、`gemm_w8a8_smooth_asm`、`fuse_silu_mul_quant`、`moe_sum`、RoPE/norm/topk 等。
- 对 SGLang/lmslim 来说，它是后端加速库。

关键开关在：

```python
# /usr/local/lib/python3.10/dist-packages/lmslim/envs.py
if "gfx936" in LMSLIM_GPU_NAME or "gfx938" in LMSLIM_GPU_NAME:
    LMSLIM_USE_LIGHTOP = int(os.getenv("LMSLIM_USE_LIGHTOP", 1))
else:
    LMSLIM_USE_LIGHTOP = 0
```

也就是说，在 Hygon `gfx936/gfx938` 上，`lmslim` 默认会启用 `lightop`。

DeepSeek V4 里二者关系可以这样理解：

普通 W8A8 Linear：
```text
SGLang compressed-tensors Linear
  -> lmslim.layers.gemm.int8_utils.per_token_quant_int8
  -> lmslim.quant_ops.triton_scaled_mm
      -> 如果 LMSLIM_USE_LIGHTOP=1:
             torch.ops.lmslim.lightop_channel_int8_mm
               -> lightop.gemm_w8a8_smooth
         否则:
             lmslim 自己的 Triton/rocBLAS/lmslimquant 路径
```

MoE INT8 Marlin：
```text
torch.ops.sglang.fused_experts_impl_int8_marlin
  -> lmslim.layers.fused_moe.fuse_moe_int8_marlin.fused_experts_impl_int8_marlin
      -> 如果 LMSLIM_USE_LIGHTOP=1:
             lightop.moe_gemm_marlin_w8a8
             lightop.fuse_silu_mul_quant
             lightop.op.moe_sum
```

所以文档里说 “Hygon 平台用 lmslim 做量化” 是对的，但更精确地说是：

```text
lmslim 负责量化计算流程和接口封装；
lightop 负责 Hygon 上很多实际高性能 kernel 的执行。
```

对 InfiniCore 来说，我们现在很多地方是**绕过 lmslim，直接桥接 lightop 的 so 符号**。这更适合 C++ 热路径，因为可以少经过 Python/lmslim 包装层，但语义上要对齐 lmslim/SGLang 的调用流程。

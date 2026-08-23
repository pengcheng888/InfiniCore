
> 请查看开源仓库sglang的代码，分析sglang-0.5.16\python\sglang\srt\models\deepseek_v4.py中的if self.compressor is not None分支里的attn_backend.forward_core_compressor函数中调用的过程。请分析

<details><summary>8 previous messages</summary>

> 我继续用 `hygon-bw1000-infinilm-deepseek-v4` 技能来梳理这条 SGLang DeepSeek V4 attention/compressor 调用链。先定位 `if self.compressor is not None` 分支，再顺着 `attn_backend.forward_core_compressor` 看 backend、compressor、kernel 的层层调用。
>
> `forward_core_compressor` 实现主要在 `srt/layers/attention/dsv4/compressor.py` 的 `CompressorBackendMixin`，另外 `compressor_v2.py` 有 unified 版本 alias。我先看模型调用上下文和 mixin 实现。
>
> 这里有两个 compressor mixin 版本：`compressor.py` 的旧分步路径，以及 `compressor_v2.py` 的 all-in-one 路径。需要看 backend 类实际继承的是哪个 mixin，才能判断当前 `forward_core_compressor` 最终走哪条。
>
> 实际 backend 继承的是 `compressor_v2.CompressorBackendMixin`，所以当前 `forward_core_compressor = forward_unified`。下一步看 `Compressor.__call__/forward` 和 `compute_kv_score`，因为 `forward_unified` 的第一步就是 `compressor.compute_kv_score(x, forward_batch)`。
>
> `Compressor` 本身只负责生成 `kv_score_input`，当前 backend 的 v2 路径再从 state pool 读历史 `kv_score_buffer`，做 compress + norm/rope + store 合并。现在补看 `compress.py` 里的两个 JIT 函数入口。
>
> `compress_norm_rope_store` 会 JIT 编译 `fused_norm_rope_v2.cuh`，根据是否 indexer/use_fp4 和 head_dim 走不同 kernel。对 core compressor 来说通常是 FlashMLA extra KV cache，走 `fused_norm_rope_flashmla`。
>
> 为了补上“什么时候 self.compressor 不为 None”，我再看一下 `DeepseekV4Attention` 初始化里 compressor 的创建条件。
>
> 我还看到一处 `elif self.compressor is not None`，可能是另外一种 prepare 分支；我补看上下文，避免漏掉一条 compressor 调用入口。
</details>

结论：`deepseek_v4.py` 中 `if self.compressor is not None` 分支里的

```python
attn_backend.forward_core_compressor(x, forward_batch, self.layer_id, self.compressor)
```

当前实际走的是 **`compressor_v2.CompressorBackendMixin.forward_unified`**，不是旧版 `compressor.py` 里的分步 `forward_core_compressor`。

原因是两个 DeepSeek V4 backend 都这样继承：

```python
from sglang.srt.layers.attention.dsv4.compressor_v2 import CompressorBackendMixin
```

见：
- [deepseek_v4_backend.py](/workspace_codex/InfiniCore/sglang-0.5.16/python/sglang/srt/layers/attention/deepseek_v4_backend.py:38)
- [deepseek_v4_backend_hip_radix.py](/workspace_codex/InfiniCore/sglang-0.5.16/python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py:29)

并且 `compressor_v2.py` 里有：

```python
forward_core_compressor = forward_unified
```

见 [compressor_v2.py](/workspace_codex/InfiniCore/sglang-0.5.16/python/sglang/srt/layers/attention/dsv4/compressor_v2.py:394)。

**整体调用链**

```text
deepseek_v4.py
  if self.compressor is not None:
      attn_backend.forward_core_compressor(x, forward_batch, layer_id, compressor)

=> compressor_v2.CompressorBackendMixin.forward_unified
  1. compressor.compute_kv_score(x, forward_batch)
  2. state_pool = compressor.get_state_pool(attn_backend)
  3. 选择 kv_cache / page_size / out_loc
  4. _forward_compress_all_in_one(...)
      4.1 compress_forward(...)
      4.2 compress_norm_rope_store(...)
```

**第 1 步：compute_kv_score**

`Compressor.compute_kv_score` 做：

```python
kv_score = linear_bf16_fp32(x, self.wkv_gate.weight)
```

也就是对输入 `x` 做一个 packed 的 `wkv_gate` linear，输出包含 kv 和 gate score。见 [compressor.py](/workspace_codex/InfiniCore/sglang-0.5.16/python/sglang/srt/layers/attention/dsv4/compressor.py:421)。

如果是 prefill CP，还会做 `cp_all_gather_rerange_output`。

**第 2 步：取 state pool**

```python
state_pool = compressor.get_state_pool(self)
```

core compressor 时取的是：

```python
token_to_kv_pool.get_attention_compress_states(layer_id)
```

这里面有历史 `kv_score_buffer`，用于 C4/C128 压缩时读取历史窗口。

**第 3 步：选择输出 cache**

在 `forward_unified` 中，如果不是 indexer：

```python
_, _, compress_kv_pool = token_to_kv_pool.layer_mapping[layer_id]
kv_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)
out_loc = self._get_out_loc(compressor.ratio)
```

见 [compressor_v2.py](/workspace_codex/InfiniCore/sglang-0.5.16/python/sglang/srt/layers/attention/dsv4/compressor_v2.py:230)。

也就是说 core compressor 写的是 **extra key buffer**，不是 SWA cache。

**第 4 步：all-in-one 压缩写 cache**

`_forward_compress_all_in_one` 做两件事：

1. `compress_forward(...)`

   根据 ratio 选择 JIT kernel：

   ```text
   ratio=4   -> deepseek_v4/c4_v2.cuh -> FlashCompress4Kernel
   ratio=128 -> deepseek_v4/c128_v2.cuh -> FlashCompress128Kernel
   ```

   它从历史 `kv_score_buffer` 和当前 `kv_score_input` 里计算新的 compressed KV。

2. `compress_norm_rope_store(...)`

   继续调用 JIT：

   ```text
   deepseek_v4/fused_norm_rope_v2.cuh
     -> FusedNormRopeKernel::forward
     -> fused_norm_rope_flashmla
   ```

   这一步完成：
   - RMSNorm
   - RoPE
   - FP8/BF16 FlashMLA cache layout 写入

所以当前 SGLang 的 core compressor 路径是：

```text
linear_bf16_fp32
  -> compress_forward
  -> fused norm + rope + store extra FlashMLA cache
```

不是先返回 BF16 compressed KV 再单独 `store_cache`，而是 v2 路径里把 **norm/rope/store** 合到 `compress_norm_rope_store` 里了。

**补充：什么时候 self.compressor 不为 None**

在 `MQALayer.__init__` 中：

```python
if self.compress_ratio in (4, 128):
    self.compressor = Compressor(...)
```

见 [deepseek_v4.py](/workspace_codex/InfiniCore/sglang-0.5.16/python/sglang/srt/models/deepseek_v4.py:602)。

所以：
- `compress_ratio=0`：没有 compressor，只写 SWA cache。
- `compress_ratio=4`：有 C4 compressor，同时还有 indexer。
- `compress_ratio=128`：有 C128 compressor，没有 C4 indexer。


> InfiniCore中，包装的是哪个flash mla算子。

<details><summary>9 previous messages</summary>

> 我会继续使用 `hygon-bw1000-infinilm-deepseek-v4` skill，因为这是 InfiniCore 对 FlashMLA 底层 SO 的包装关系分析。先按约定读 skill，然后定位 InfiniCore 里 `flashmla`/`flash_mla` 相关算子和桥接代码。
>
> <details><summary>Ran a command</summary>
>
> <details><summary>Explored a file</summary>
>
> - Read `/root/.codex/skills/hygon-bw1000-infinilm-deepseek-v4-0727/SKILL.md`
>
> </details>
>
> - Ran `rg -n "flash_mla|flashmla|FlashMLA|sparse_decode|dense_decode|flash_mla_cuda|fwd_kvcache" /workspace_codex/InfiniCore/src /workspace_codex/InfiniCore/include /workspace_codex/InfiniLM/csrc/models/deepseek_v4 2>/dev/null`
>
> </details>
>
> 检索结果显示 InfiniCore 有两层包装：一层是 Python pybind 里对 `flash_mla.cuda` 各个入口的直通绑定；另一层是 C++ 模型热路径用的 `deepseek_v4_flashmla_sparse_attention_*`。我现在重点看 C++ 热路径到底调用哪个底层 FlashMLA 函数。
>
> <details><summary>Explored 2 files, a search</summary>
>
> - Read `/workspace_codex/InfiniCore/include/infinicore/ops/deepseek_v4_flashmla_compute.hpp`
> - Searched for `*deepseek_v4_flashmla_compute*` in `infinicore`
> - Read `/workspace_codex/InfiniCore/src/infinicore/pybind11/ops/deepseek_v4_flashmla_compute.hpp`
>
> </details>
>
> `pybind` 里还额外暴露了一组 `deepseek_v4_dense_decode_fwd`、`deepseek_v4_sparse_decode_fwd` 等直通 `flash_mla.cuda` 的调试/测试入口；但 InfiniLM C++ attention 里主要不是走这些 Python 直通入口，而是走 InfiniCore 自己的 sparse attention wrapper。继续看实现细节。
>
> <details><summary>Explored 3 files</summary>
>
> - Read `/workspace_codex/InfiniCore/src/infinicore/ops/deepseek_v4_flashmla_compute/deepseek_v4_flashmla_compute.cc`
> - Read `/workspace_codex/InfiniCore/src/infinicore/ops/deepseek_v4_flashmla_compute/deepseek_v4_flashmla_compute_naive.cc`
> - Read `/workspace_codex/InfiniCore/src/infinicore/ops/deepseek_v4_flashmla_compute/deepseek_v4_flashmla_compute.cc`
>
> </details>
>
> 目前可以确定：InfiniCore 的核心包装目标是 `flash_mla.cuda` 里的 **sparse decode FP8** 路径；普通 with_metadata 版本直接调接口函数，out_workspace 版本绕过 Python/pybind，直接解析 `.so` 里的局部 kernel 符号。继续看 out_workspace 的具体分发。
>
> <details><summary>Explored 3 files</summary>
>
> - Read `/workspace_codex/InfiniCore/src/infinicore/ops/deepseek_v4_flashmla_compute/deepseek_v4_flashmla_compute.cc`
> - Read `/workspace_codex/InfiniCore/src/infinicore/ops/deepseek_v4_flashmla_compute/deepseek_v4_flashmla_compute.cc`
> - Read `./csrc/models/deepseek_v4/deepseek_v4_attention.cpp`
>
> </details>
>
> Context compaction
>
> Source: automatic
> Status: completed
</details>

InfiniCore 里主路径包装的是 **FlashMLA 的 FP8 sparse decode 算子**，也就是 Python 侧 `flash_mla.cuda.sparse_decode_fwd` 对应的底层实现。

具体分两层：

1. **InfiniLM 实际调用的 C++ 热路径**

在 InfiniCore 里主要是这些接口：

```cpp
infinicore::op::deepseek_v4_flashmla_sparse_attention_with_metadata_
infinicore::op::deepseek_v4_flashmla_sparse_attention_out_workspace_
infinicore::op::deepseek_v4_flashmla_sparse_attention_metadata_
```

其中：

- `deepseek_v4_flashmla_sparse_attention_with_metadata_`
  - 包装 FlashMLA 的 `sparse_attn_decode_interface`
  - 等价于 Python 侧的 `flash_mla.cuda.sparse_decode_fwd`

- `deepseek_v4_flashmla_sparse_attention_out_workspace_`
  - 不走 Python interface
  - 直接解析并调用 FlashMLA `.so` 里的本地 kernel symbol：
    ```cpp
    gfx936::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType, H>
    gfx96::decode::run_flash_mla_combine_kernel<cutlass::bfloat16_t>
    ```
  - 也就是 sparse decode 的 split-KV kernel + combine kernel

- `deepseek_v4_flashmla_sparse_attention_metadata_`
  - 包装 metadata kernel：
    ```cpp
    gfx96::decode::run_get_decoding_sched_meta_kernel
    ```

2. **Python pybind 里还额外暴露了一些 FlashMLA 函数**

InfiniCore 的 pybind 层还直接转发了这些 `flash_mla.cuda` 函数：

```cpp
dense_decode_fwd
dense_decode_fwd_kvfp8
dense_decode_fwd_qkvfp8
fwd_kvcache_mla_fp8
fwd_kvcache_mla_fp8_with_cat
fwd_kvcache_mla_nope_pe
fwd_kvcache_quantization_mla
fwd_kvcache_quantization_q_nope_pe_mla
get_mla_decoding_metadata_dense_fp8
sparse_decode_fwd
sparse_prefill_fwd
```

但这些更像是测试/调试用的 Python 直通包装，不是 InfiniLM 当前 DeepSeek V4 attention 的主要 C++ 热路径。

所以一句话总结：

**InfiniCore 在 InfiniLM 主流程中包装的是 FlashMLA 的 `sparse_decode_fwd`，更具体地说是 FP8 sparse decode 的 split-KV FlashMLA kernel，加 metadata kernel 和 BF16 combine kernel。不是 dense decode，也不是 sparse prefill。**




> /workspace_codex/InfiniCore/src/infinicore/ops/mha_kvcache/mha_kvcache.cc文件中，mha_kvcache算子是何如连接到flash attn库的so的？

<details><summary>8 previous messages</summary>

> 我继续使用 `hygon-bw1000-infinilm-deepseek-v4` skill；这个问题是在看 InfiniCore 里 vendor SO bridge 的接入方式。先读相关约定，然后直接看 `mha_kvcache.cc` 和它周边的 backend 文件。
>
> <details><summary>Explored 2 files, a search</summary>
>
> - Read `./src/infinicore/ops/mha_kvcache/mha_kvcache.cc`
> - Read `/root/.codex/skills/hygon-bw1000-infinilm-deepseek-v4-0727/SKILL.md`
> - Searched for `mha_kvcache|flash|flash_attn|flashattn|dlopen|dlsym|Dispatcher|findSchemaOrThrow|REGISTER|plan|run|cleanup|handle` in `mha_kvcache`
>
> </details>
>
> `mha_kvcache.cc` 本身只是 public API 和 graph 外壳，真正连 FlashAttention 的逻辑在后端文件里。我再展开 Hygon 后端和 adaptor，看它是直接链接、dispatcher 还是 Python wrapper。
>
> <details><summary>Explored 3 files, a search</summary>
>
> - Read `./src/infinicore/ops/mha_kvcache/hygon/mha_kvcache_flashattn_hygon.cc`
> - Read `./src/infinicore/ops/mha_kvcache/mha_kvcache_flashattn.cc`
> - Read `./src/infinicore/adaptor/flash_attn/hygon/flash_attn_hygon.hpp`
> - Searched for `mha_fwd_kvcache|flash_attn_hygon|dlopen|dlsym|py::module|TORCH_LIBRARY|findSchema|Dispatcher` in `adaptor`
>
> </details>
>
> 这里能看出 Hygon 路径不是通过 `torch.ops` 注册名，而是 `dlopen/dlsym` 找 FlashAttention extension 里的 C++ 符号。我再看一下 resolver 的候选 `.so` 路径和符号名。
>
> <details><summary>Explored a file</summary>
>
> - Read `./src/infinicore/adaptor/flash_attn/hygon/flash_attn_hygon.cc`
>
> </details>
>
> 我已经确认了调用链：`mha_kvcache.cc` 不直接碰 `.so`，Hygon 后端通过一个 adaptor 动态打开 `flash_attn_2_cuda*.so`，再找 `mha_fwd_kvcache` 符号。下面我按调用链说明。
>
> <details><summary>Explored 3 files</summary>
>
> - Read `./src/infinicore/ops/mha_kvcache/mha_kvcache.cc`
> - Read `./src/infinicore/ops/mha_kvcache/hygon/mha_kvcache_flashattn_hygon.cc`
> - Read `./src/infinicore/adaptor/flash_attn/hygon/flash_attn_hygon.cc`
>
> </details>
</details>

`mha_kvcache.cc` 本身没有直接连接 FlashAttention `.so`。它只是 public API + graph 外壳，真正连接 `.so` 的逻辑在 Hygon 后端和 adaptor 里。

调用链是：

```text
mha_kvcache_()
  -> MhaKVCache::execute()
    -> graph record 或直接 run
      -> HYGON backend: mha_kvcache_flashattn_hygon.cc::run()
        -> flash::mha_fwd_kvcache(...)
          -> dlopen flash_attn_2_cuda*.so
          -> dlsym("mha_fwd_kvcache")
          -> 调用 FlashAttention extension 的 C++ 符号
```

关键位置：

- [mha_kvcache.cc](./InfiniCore/src/infinicore/ops/mha_kvcache/mha_kvcache.cc:21)
  `MhaKVCache::execute` 通过 `INFINICORE_GRAPH_OP_RECORD_OR_RUN` 进入 graph op 的 plan/run 机制。

- [mha_kvcache_flashattn_hygon.cc](./InfiniCore/src/infinicore/ops/mha_kvcache/hygon/mha_kvcache_flashattn_hygon.cc:159)
  Hygon 后端注册：

```cpp
MhaKVCache::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
MhaKVCache::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
MhaKVCache::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
```

- [mha_kvcache_flashattn_hygon.cc](./InfiniCore/src/infinicore/ops/mha_kvcache/hygon/mha_kvcache_flashattn_hygon.cc:124)
  `run()` 里把 InfiniCore `Tensor` 转成 `at::Tensor`，然后调用：

```cpp
auto result = flash::mha_fwd_kvcache(...);
```

- [flash_attn_hygon.cc](./InfiniCore/src/infinicore/adaptor/flash_attn/hygon/flash_attn_hygon.cc:73)
  这里才是真正连接 `.so` 的地方：

```cpp
dlopen("flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so", RTLD_NOW | RTLD_GLOBAL)
dlsym(handle, "mha_fwd_kvcache")
```

所以它不是通过 `torch.ops` / `c10::Dispatcher` 找 schema，也不是 Python import 调用；而是 **C++ 里 `dlopen + dlsym` 动态解析 FlashAttention extension 的导出 C++ 符号 `mha_fwd_kvcache`**。

另外 Hygon 后端还用了：

```cpp
c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
```

这保证 FlashAttention SO 调用走当前 InfiniCore/HIP stream。

> C++ 通过 dlopen + dlsym 动态解析flash_mla/cuda*.so，是找不到sparse_attn_decode_interface符号。
> 请问 dlopen + dlsym 动态解析flash_mla/cuda*.so找到的符号有哪些？

<details><summary>4 previous messages</summary>

> 我继续用 `hygon-bw1000-infinilm-deepseek-v4` skill 来看这个 SO bridge 问题。你这个判断是对的：普通 `dlopen + dlsym` 只能找动态导出符号，找不到 local 的 `sparse_attn_decode_interface`。我直接列一下当前 `flash_mla/cuda*.so` 的动态导出符号。
>
> <details><summary>Ran 2 commands</summary>
>
> <details><summary>Explored a file</summary>
>
> - Read `/root/.codex/skills/hygon-bw1000-infinilm-deepseek-v4-0727/SKILL.md`
>
> </details>
>
> - Ran `nm -D --defined-only /usr/local/lib/python3.10/dist-packages/flash_mla/cuda.cpython-310-x86_64-linux-gnu.so | c++filt | rg ' T | W | V | B | D '`
>
> - Ran `nm -D --defined-only /usr/local/lib/python3.10/dist-packages/flash_mla/cuda.cpython-310-x86_64-linux-gnu.so | c++filt | rg 'mha_fwd|decode|PyInit|flash|run_'`
>
> </details>
>
> 我补一句很容易踩坑的细节：`nm -D | c++filt` 显示的是 demangle 后的人类可读名，但 `dlsym` 实际要用 `.so` 里的原始 mangled 符号名，除非符号是 `extern "C"`。我再抓一下 sparse/mla 相关的原始名字，方便对照。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `nm -D --defined-only /usr/local/lib/python3.10/dist-packages/flash_mla/cuda.cpython-310-x86_64-linux-gnu.so | rg 'mha_fwd|metadata|Decode|Sparse|sparse|flash_fwd_mla_combine|splitkv_mla_fp8' | head -120`
> - Ran `nm -D --defined-only /usr/local/lib/python3.10/dist-packages/flash_mla/cuda.cpython-310-x86_64-linux-gnu.so | c++filt | rg 'sparse|Sparse|Decode|mha_fwd|metadata|combine' | head -120`
>
> </details>
</details>

对，普通 `dlopen + dlsym` 解析 `flash_mla/cuda*.so` 时，只能找到 **动态符号表 `.dynsym` 里导出的符号**。`sparse_attn_decode_interface` / `dense_attn_decode_interface` 是 local symbol，不在这个范围内。

当前这个 SO 里，比较有用、能被 `dlsym` 找到的主要有这些类别：

**1. Python 模块入口**

```text
PyInit_cuda
```

这个是 Python import `flash_mla.cuda` 时用的入口。

**2. MLA C++ wrapper 入口**

这些是比较接近可直接接入的 C++ wrapper：

```cpp
mha_fwd_kvcache_mla_fp8(...)
mha_fwd_kvcache_mla_nope_pe(...)
mha_fwd_kvcache_mla_fp8_with_cat(...)
mha_fwd_kvcache_quantization_mla(...)
mha_fwd_kvcache_quantization_q_nope_pe_mla(...)
```

但 `dlsym` 不能用 demangle 后的名字，要用原始 mangled name，例如：

```cpp
"_Z23mha_fwd_kvcache_mla_fp8RN2at6TensorERKS0_RSt8optionalIS2_EiS3_S3_fbS3_S3_RKS4_IS0_ES9_"
"_Z27mha_fwd_kvcache_mla_nope_peRN2at6TensorES1_RKS0_RSt8optionalIS2_EiS3_S3_fbS3_S3_"
"_Z32mha_fwd_kvcache_mla_fp8_with_catRN2at6TensorES1_RKS0_RSt8optionalIS2_EiS3_S3_fbS3_S3_RKS4_IS0_ES9_"
```

**3. metadata 相关入口**

```cpp
get_mla_metadata_func(...)
get_mla_decoding_metadata_dense_fp8(...)
run_get_mla_metadata_kernel(...)
gfx9::decode::run_get_decoding_sched_meta_kernel(...)
gfx9::decode::run_get_decoding_sched_meta_no_split_kernel(...)
```

**4. 更底层 dense decode kernel / runner**

例如：

```cpp
gfx93::run_flash_splitkv_mla_kernel<cutlass::bfloat16_t>(...)
gfx93::run_flash_splitkv_mla_kernel<cutlass::half_t>(...)
gfx93::run_flash_splitkv_mla_kvfp8_kernel<cutlass::bfloat16_t>(...)
gfx93::run_flash_splitkv_mla_qkvfp8_kernel<cutlass::float_e4m3_t>(...)
```

**5. sparse 相关可见符号**

可以看到一些 sparse class/runner 级别符号，例如：

```cpp
Decode_Sm90_Impl::run_(...)
Decode_Sm90_Impl::get_meta(...)
Fwd_Sm90_Impl::run_(...)
gfx93::run_fwd_kernel(...)
```

但这类不是 `flash_mla.cuda.sparse_decode_fwd` 的直接 wrapper，直接接入风险更高，需要还原 struct 参数布局。

**6. combine kernel 相关**

例如：

```cpp
gfx9::decode::run_flash_mla_combine_kernel<cutlass::bfloat16_t>(...)
gfx9::decode::run_flash_mla_combine_kernel<cutlass::half_t>(...)
flash::flash_fwd_splitkv_mla_combine_kernel<...>
```

所以简洁结论是：

- `dlsym` **找不到** `sparse_attn_decode_interface` / `dense_attn_decode_interface`。
- `dlsym` **可以找到** `mha_fwd_kvcache_mla_fp8`、`mha_fwd_kvcache_mla_nope_pe`、`mha_fwd_kvcache_mla_fp8_with_cat`、metadata、部分 dense/sparse runner、combine kernel 等导出符号。
- 如果要用普通 `dlopen + dlsym` 接入，优先考虑 `mha_fwd_kvcache_mla_fp8` / `mha_fwd_kvcache_mla_nope_pe` 这种 wrapper；如果要接 `sparse_attn_decode_interface`，就必须像已有 `deepseek_v4_flashmla_compute.cc` 那样解析 ELF local symbol。

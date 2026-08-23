# flash_mla cuda.so 可 dlsym 符号记录

本文记录当前环境中 `/usr/local/lib/python3.10/dist-packages/flash_mla/cuda.cpython-310-x86_64-linux-gnu.so` 里，普通 `dlopen + dlsym` 能直接找到的主要 FlashMLA 相关动态导出符号。

检查命令：

```bash
nm -D --defined-only /usr/local/lib/python3.10/dist-packages/flash_mla/cuda.cpython-310-x86_64-linux-gnu.so | c++filt
```

注意：`dlsym` 需要传入原始 mangled symbol 名，不能传入 `c++filt` 之后的人类可读签名，除非该符号是 C ABI。

## Python 模块入口

| 原始符号 | demangle 后含义 | 说明 |
| --- | --- | --- |
| `PyInit_cuda` | `PyInit_cuda` | Python import `flash_mla.cuda` 的模块初始化入口。 |

## 高层 MLA wrapper

这些符号是当前更适合作为 C++ bridge 尝试入口的动态导出函数。

| 原始符号 | demangle 后签名 | 备注 |
| --- | --- | --- |
| `_Z23mha_fwd_kvcache_mla_fp8RN2at6TensorERKS0_RSt8optionalIS2_EiS3_S3_fbS3_S3_RKS4_IS0_ES9_` | `mha_fwd_kvcache_mla_fp8(at::Tensor&, at::Tensor const&, std::optional<at::Tensor const>&, int, at::Tensor const&, at::Tensor const&, float, bool, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, std::optional<at::Tensor> const&)` | FP8 KV cache MLA wrapper。 |
| `_Z27mha_fwd_kvcache_mla_nope_peRN2at6TensorES1_RKS0_RSt8optionalIS2_EiS3_S3_fbS3_S3_` | `mha_fwd_kvcache_mla_nope_pe(at::Tensor&, at::Tensor&, at::Tensor const&, std::optional<at::Tensor const>&, int, at::Tensor const&, at::Tensor const&, float, bool, at::Tensor const&, at::Tensor const&)` | no-pe / pe 分离形式 wrapper。 |
| `_Z32mha_fwd_kvcache_mla_fp8_with_catRN2at6TensorES1_RKS0_RSt8optionalIS2_EiS3_S3_fbS3_S3_RKS4_IS0_ES9_` | `mha_fwd_kvcache_mla_fp8_with_cat(at::Tensor&, at::Tensor&, at::Tensor const&, std::optional<at::Tensor const>&, int, at::Tensor const&, at::Tensor const&, float, bool, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, std::optional<at::Tensor> const&)` | FP8 MLA wrapper，带 cat/拼接路径。 |
| `_Z32mha_fwd_kvcache_quantization_mlaRN2at6TensorERKS0_RSt8optionalIS2_EiS3_S3_fbS3_S3_S3_RKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE` | `mha_fwd_kvcache_quantization_mla(at::Tensor&, at::Tensor const&, std::optional<at::Tensor const>&, int, at::Tensor const&, at::Tensor const&, float, bool, at::Tensor const&, at::Tensor const&, at::Tensor const&, std::string const&)` | 带 quantization 的 MLA wrapper。 |
| `_Z42mha_fwd_kvcache_quantization_q_nope_pe_mlaRN2at6TensorES1_RKS0_RSt8optionalIS2_EiS3_S3_fbS3_S3_S3_RKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE` | `mha_fwd_kvcache_quantization_q_nope_pe_mla(at::Tensor&, at::Tensor&, at::Tensor const&, std::optional<at::Tensor const>&, int, at::Tensor const&, at::Tensor const&, float, bool, at::Tensor const&, at::Tensor const&, at::Tensor const&, std::string const&)` | q/nope/pe 分离并带 quantization 的 wrapper。 |

## Metadata 相关

| 原始符号 | demangle 后签名 | 说明 |
| --- | --- | --- |
| `_Z21get_mla_metadata_funcR19Mla_metadata_paramsP12ihipStream_t` | `get_mla_metadata_func(Mla_metadata_params&, ihipStream_t*)` | MLA metadata 生成入口。 |
| `_Z27run_get_mla_metadata_kernelR25GetDecodingMetadataParamsP12ihipStream_t` | `run_get_mla_metadata_kernel(GetDecodingMetadataParams&, ihipStream_t*)` | decode metadata kernel launcher。 |
| `_Z35get_mla_decoding_metadata_dense_fp8RN2at6TensorEii` | `get_mla_decoding_metadata_dense_fp8(at::Tensor&, int, int)` | dense FP8 decoding metadata helper。 |
| `_ZN4gfx96decode43run_get_decoding_sched_meta_no_split_kernelER24GetDecodeSchedMetaParams` | `gfx9::decode::run_get_decoding_sched_meta_no_split_kernel(GetDecodeSchedMetaParams&)` | gfx9 decode schedule metadata no-split launcher。 |

下面这些也是动态符号，但属于 kernel function object / data symbol，普通 `dlsym` 能取地址，直接调用时需要准确匹配底层参数布局：

```text
_Z23get_mla_metadata_kernel19Mla_metadata_params
_Z32get_mla_decoding_metadata_kernel25GetDecodingMetadataParams
_ZN4gfx96decode23get_mla_metadata_kernelE24GetDecodeSchedMetaParams
_ZN4gfx96decode32get_mla_metadata_no_split_kernelE24GetDecodeSchedMetaParams
```

## Dense decode runner

| 原始符号 | demangle 后签名 | 说明 |
| --- | --- | --- |
| `_ZN5gfx9328run_flash_splitkv_mla_kernelIN7cutlass10bfloat16_tEEEvR21DenseAttnDecodeParams` | `gfx93::run_flash_splitkv_mla_kernel<cutlass::bfloat16_t>(DenseAttnDecodeParams&)` | dense BF16 runner。 |
| `_ZN5gfx9328run_flash_splitkv_mla_kernelIN7cutlass6half_tEEEvR21DenseAttnDecodeParams` | `gfx93::run_flash_splitkv_mla_kernel<cutlass::half_t>(DenseAttnDecodeParams&)` | dense FP16 runner。 |
| `_ZN5gfx9334run_flash_splitkv_mla_kvfp8_kernelIN7cutlass10bfloat16_tEEEvR25DenseAttnDecodeParams_fp8` | `gfx93::run_flash_splitkv_mla_kvfp8_kernel<cutlass::bfloat16_t>(DenseAttnDecodeParams_fp8&)` | dense KV-FP8 runner。 |
| `_ZN5gfx9335run_flash_splitkv_mla_qkvfp8_kernelIN7cutlass12float_e4m3_tEEEvR25DenseAttnDecodeParams_fp8` | `gfx93::run_flash_splitkv_mla_qkvfp8_kernel<cutlass::float_e4m3_t>(DenseAttnDecodeParams_fp8&)` | dense QKV-FP8 runner。 |

## Sparse 相关 runner

这些符号可以通过 `dlsym` 找到，但不是 `flash_mla.cuda.sparse_decode_fwd` 的直接 wrapper。直接使用它们需要还原 `SparseAttnDecodeParams` / `DecodeFeatures` 等结构体布局。

| 原始符号 | demangle 后签名 | 说明 |
| --- | --- | --- |
| `_ZN16Decode_Sm90_Impl4run_ERK22SparseAttnDecodeParamsRKSt6vectorI14DecodeFeaturesSaIS4_EE` | `Decode_Sm90_Impl::run_(SparseAttnDecodeParams const&, std::vector<DecodeFeatures> const&)` | sparse decode 实现类 runner。 |
| `_ZN16Decode_Sm90_Impl8get_metaEii` | `Decode_Sm90_Impl::get_meta(int, int)` | sparse decode metadata 查询。 |
| `_ZN13Fwd_Sm90_Impl4run_ERK19SparseAttnFwdParamsRKSt6vectorI11FwdFeaturesSaIS4_EE` | `Fwd_Sm90_Impl::run_(SparseAttnFwdParams const&, std::vector<FwdFeatures> const&)` | sparse forward 实现类 runner。 |
| `_ZN5gfx9314run_fwd_kernelERK19SparseAttnFwdParams` | `gfx93::run_fwd_kernel(SparseAttnFwdParams const&)` | gfx93 sparse forward kernel runner。 |
| `_ZN5gfx936decode10sparse_fp839run_flash_splitkv_mla_fp8_sparse_kernelIL9ModelType0ELi16EEEvRK22SparseAttnDecodeParams` | `gfx93::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType0, 16>(SparseAttnDecodeParams const&)` | sparse FP8 decode, ModelType0, top/block variant 16。 |
| `_ZN5gfx936decode10sparse_fp839run_flash_splitkv_mla_fp8_sparse_kernelIL9ModelType0ELi64EEEvRK22SparseAttnDecodeParams` | `gfx93::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType0, 64>(SparseAttnDecodeParams const&)` | sparse FP8 decode, ModelType0, variant 64。 |
| `_ZN5gfx936decode10sparse_fp839run_flash_splitkv_mla_fp8_sparse_kernelIL9ModelType0ELi128EEEvRK22SparseAttnDecodeParams` | `gfx93::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType0, 128>(SparseAttnDecodeParams const&)` | sparse FP8 decode, ModelType0, variant 128。 |
| `_ZN5gfx936decode10sparse_fp839run_flash_splitkv_mla_fp8_sparse_kernelIL9ModelType1ELi16EEEvRK22SparseAttnDecodeParams` | `gfx93::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType1, 16>(SparseAttnDecodeParams const&)` | sparse FP8 decode, ModelType1, variant 16。 |
| `_ZN5gfx936decode10sparse_fp839run_flash_splitkv_mla_fp8_sparse_kernelIL9ModelType1ELi64EEEvRK22SparseAttnDecodeParams` | `gfx93::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType1, 64>(SparseAttnDecodeParams const&)` | sparse FP8 decode, ModelType1, variant 64。 |
| `_ZN5gfx936decode10sparse_fp839run_flash_splitkv_mla_fp8_sparse_kernelIL9ModelType1ELi128EEEvRK22SparseAttnDecodeParams` | `gfx93::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType1, 128>(SparseAttnDecodeParams const&)` | sparse FP8 decode, ModelType1, variant 128。 |

## Combine 相关

| 原始符号 | demangle 后签名 | 说明 |
| --- | --- | --- |
| `_ZN4gfx96decode28run_flash_mla_combine_kernelIN7cutlass10bfloat16_tEEEvR13CombineParams` | `gfx9::decode::run_flash_mla_combine_kernel<cutlass::bfloat16_t>(CombineParams&)` | BF16 combine runner。 |
| `_ZN4gfx96decode28run_flash_mla_combine_kernelIN7cutlass6half_tEEEvR13CombineParams` | `gfx9::decode::run_flash_mla_combine_kernel<cutlass::half_t>(CombineParams&)` | FP16 combine runner。 |

另外有多组 `gfx9::decode::flash_fwd_mla_combine_kernel<...>` 和 `flash::flash_fwd_splitkv_mla_combine_kernel<...>` kernel function object 符号。它们也在动态符号表中，但属于更底层 kernel 符号，直接调用需要匹配 kernel stub/参数结构。

## 不可普通 dlsym 的 local wrapper

下面这些函数存在于完整符号表中，但符号类型是小写 `t`，表示 local text symbol。普通 `dlsym` 只能查动态符号表，因此找不到它们。

| local wrapper | 对应 Python API | 说明 |
| --- | --- | --- |
| `sparse_attn_decode_interface(...)` | `flash_mla.cuda.sparse_decode_fwd` | sparse decode Python API 对应的 C++ pybind wrapper。 |
| `dense_attn_decode_interface(...)` | `flash_mla.cuda.dense_decode_fwd` | dense decode Python API 对应的 C++ pybind wrapper。 |
| `dense_attn_decode_kvfp8_interface(...)` | `flash_mla.cuda.dense_decode_fwd_kvfp8` | dense KV-FP8 Python API 对应 wrapper。 |
| `dense_attn_decode_qkvfp8_interface(...)` | `flash_mla.cuda.dense_decode_fwd_qkvfp8` | dense QKV-FP8 Python API 对应 wrapper。 |

如果一定要调用这些 local wrapper，需要像 `deepseek_v4_flashmla_compute.cc` 那样解析 ELF section symbol table，计算 SO load base 后再取地址；这不是普通 `dlopen + dlsym` 路线，且 ABI 稳定性更弱。

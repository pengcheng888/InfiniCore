#include "infinicore/ops/deepseek_v4_flashmla_cache.hpp"

#include "deepseek_v4_flashmla_cache_kernel.hpp"

#include "infinicore/ops/concat_and_cache_mla.hpp"
#include "infinicore/ops/deepseek_v4_create_flashmla_kv_indices.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#include <cmath>
#include <stdexcept>

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#elif defined(ENABLE_NVIDIA_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4StoreFlashMlaRawCacheKernel);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4IndexerRotate128Kernel);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4StoreIndexerRawCacheKernel);

namespace {

constexpr int64_t kDsv4FlashMlaInputDim = 512;
constexpr int64_t kDsv4FlashMlaNopeDim = 448;
constexpr int64_t kDsv4FlashMlaRopeDim = 64;
constexpr int64_t kDsv4FlashMlaValueBytesPerToken = 576;
constexpr int64_t kDsv4FlashMlaScaleBytesPerToken = 8;
constexpr int64_t kDsv4IndexerInputDim = 128;
constexpr int64_t kDsv4IndexerScaleBytesPerToken = 4;
constexpr double kDsv4Fp8E4M3Max = 448.0;

int64_t div_ceil_i64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

int64_t dsv4_flashmla_page_bytes(int page_size) {
    const auto bytes = (kDsv4FlashMlaValueBytesPerToken + kDsv4FlashMlaScaleBytesPerToken) *
                       static_cast<int64_t>(page_size);
    return div_ceil_i64(bytes, kDsv4FlashMlaValueBytesPerToken) * kDsv4FlashMlaValueBytesPerToken;
}

int64_t dsv4_indexer_page_bytes(int page_size) {
    return (kDsv4IndexerInputDim + kDsv4IndexerScaleBytesPerToken) * static_cast<int64_t>(page_size);
}

void check_raw_store_shapes(const Tensor &input, const Tensor &cache, const Tensor &indices, int page_size) {
    if (input->ndim() != 2 || input->size(1) != static_cast<size_t>(kDsv4FlashMlaInputDim)) {
        throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_ expects input [tokens, 512].");
    }
    if (cache->ndim() != 2) {
        throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_ expects raw cache [blocks, page_bytes].");
    }
    if (indices->ndim() != 1 || indices->size(0) != input->size(0)) {
        throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_ expects indices [tokens].");
    }
    if (input->dtype() != DataType::BF16 && input->dtype() != DataType::F16 && input->dtype() != DataType::F32) {
        throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_ supports bf16/fp16/fp32 input.");
    }
    if (cache->dtype() != DataType::U8) {
        throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_ raw cache dtype must be uint8.");
    }
    if (indices->dtype() != DataType::I32 && indices->dtype() != DataType::I64) {
        throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_ indices dtype must be int32 or int64.");
    }
    if (page_size <= 0 || (page_size & (page_size - 1)) != 0) {
        throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_ page_size must be a positive power of two.");
    }
    const auto expected_page_bytes = static_cast<size_t>(dsv4_flashmla_page_bytes(page_size));
    if (cache->size(1) != expected_page_bytes) {
        throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_ raw cache page_bytes mismatch.");
    }
}

void check_indexer_rotate_shapes(const Tensor &input) {
    if (input->ndim() < 1 || input->size(input->ndim() - 1) == 0) {
        throw std::runtime_error("deepseek_v4_indexer_rotate_ expects a non-empty last dimension.");
    }
    const auto dim = input->size(input->ndim() - 1);
    if ((dim & (dim - 1)) != 0) {
        throw std::runtime_error("deepseek_v4_indexer_rotate_ expects the last dimension to be a power of two.");
    }
    if (input->dtype() != DataType::BF16 && input->dtype() != DataType::F16 && input->dtype() != DataType::F32) {
        throw std::runtime_error("deepseek_v4_indexer_rotate_ supports bf16/fp16/fp32 input.");
    }
}

void check_indexer_raw_store_shapes(const Tensor &input, const Tensor &cache, const Tensor &indices, int page_size) {
    if (input->ndim() != 2 || input->size(1) != static_cast<size_t>(kDsv4IndexerInputDim)) {
        throw std::runtime_error("deepseek_v4_store_indexer_raw_cache_ expects input [tokens, 128].");
    }
    if (cache->ndim() != 2) {
        throw std::runtime_error("deepseek_v4_store_indexer_raw_cache_ expects raw cache [blocks, page_bytes].");
    }
    if (indices->ndim() != 1 || indices->size(0) != input->size(0)) {
        throw std::runtime_error("deepseek_v4_store_indexer_raw_cache_ expects indices [tokens].");
    }
    if (input->dtype() != DataType::BF16 && input->dtype() != DataType::F16 && input->dtype() != DataType::F32) {
        throw std::runtime_error("deepseek_v4_store_indexer_raw_cache_ supports bf16/fp16/fp32 input.");
    }
    if (cache->dtype() != DataType::U8) {
        throw std::runtime_error("deepseek_v4_store_indexer_raw_cache_ raw cache dtype must be uint8.");
    }
    if (indices->dtype() != DataType::I32 && indices->dtype() != DataType::I64) {
        throw std::runtime_error("deepseek_v4_store_indexer_raw_cache_ indices dtype must be int32 or int64.");
    }
    if (page_size <= 0 || (page_size & (page_size - 1)) != 0) {
        throw std::runtime_error("deepseek_v4_store_indexer_raw_cache_ page_size must be a positive power of two.");
    }
    const auto expected_page_bytes = static_cast<size_t>(dsv4_indexer_page_bytes(page_size));
    if (cache->size(1) != expected_page_bytes) {
        throw std::runtime_error("deepseek_v4_store_indexer_raw_cache_ raw cache page_bytes mismatch.");
    }
}

#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API))
at::Tensor arange_like_cols(int64_t cols, const at::Tensor &ref) {
    return at::arange(cols, ref.options().dtype(at::kLong));
}

at::ScalarType dsv4_indexer_fp8_dtype() {
    return at::ScalarType::Float8_e4m3fn;
}

double dsv4_indexer_fp8_max() {
    return kDsv4Fp8E4M3Max;
}

int dsv4_scalar_type_for_kernel(const Tensor &tensor, const char *op_name) {
    if (tensor->dtype() == DataType::BF16) {
        return deepseek_v4_flashmla_cache::kDsv4BF16;
    }
    if (tensor->dtype() == DataType::F16) {
        return deepseek_v4_flashmla_cache::kDsv4F16;
    }
    if (tensor->dtype() == DataType::F32) {
        return deepseek_v4_flashmla_cache::kDsv4F32;
    }
    throw std::runtime_error(std::string(op_name) + " supports bf16/fp16/fp32 input.");
}

#endif

} // namespace

void deepseek_v4_fused_store_flashmla_cache_(const Tensor &kv_c,
                                             const Tensor &k_pe,
                                             Tensor kv_cache,
                                             const Tensor &slot_mapping,
                                             const std::string &kv_cache_dtype,
                                             const Tensor &scale) {
    concat_and_cache_mla_(kv_c,
                          k_pe,
                          kv_cache,
                          slot_mapping,
                          kv_cache_dtype,
                          scale);
}



void deepseek_v4_store_flashmla_raw_cache_(const Tensor &input,
                                           Tensor cache,
                                           const Tensor &indices,
                                           int page_size) {
    deepseek_v4_store_flashmla_raw_cache_kernel_(input, cache, indices, page_size);
}

void deepseek_v4_indexer_rotate_(Tensor input, bool apply_scale) {
    deepseek_v4_indexer_rotate_128_kernel_(input, apply_scale);
}

void deepseek_v4_store_indexer_raw_cache_(const Tensor &input,
                                          Tensor cache,
                                          const Tensor &indices,
                                          int page_size) {
    deepseek_v4_store_indexer_raw_cache_kernel_(input, cache, indices, page_size);
}

DeepseekV4StoreFlashMlaRawCacheKernel::DeepseekV4StoreFlashMlaRawCacheKernel(const Tensor &input,
                                                                             Tensor cache,
                                                                             const Tensor &indices,
                                                                             int page_size) {
    INFINICORE_GRAPH_OP_DISPATCH(input->device().getType(), input, cache, indices, page_size);
}

void DeepseekV4StoreFlashMlaRawCacheKernel::execute(const Tensor &input,
                                                    Tensor cache,
                                                    const Tensor &indices,
                                                    int page_size) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4StoreFlashMlaRawCacheKernel, input, cache, indices, page_size);
}

namespace deepseek_v4_store_flashmla_raw_cache_graph_impl {

struct PlannedMeta {
    graph::GraphTensor input;
    graph::GraphTensor cache;
    graph::GraphTensor indices;
    int input_dtype;
    bool indices_i64;
    int64_t tokens;
    int page_size;
    int64_t page_bytes;
};

void *plan(const Tensor &input, Tensor cache, const Tensor &indices, int page_size) {
    check_raw_store_shapes(input, cache, indices, page_size);
    if (!input->is_contiguous() || !cache->is_contiguous() || !indices->is_contiguous()) {
        throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_kernel_ expects contiguous tensors.");
    }
    if (input->dtype() == DataType::F32) {
        throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_kernel_ expects bf16/fp16 input because RoPE bytes are copied verbatim.");
    }
    return new PlannedMeta{graph::GraphTensor(input),
                           graph::GraphTensor(cache),
                           graph::GraphTensor(indices),
                           dsv4_scalar_type_for_kernel(input, "deepseek_v4_store_flashmla_raw_cache_kernel_"),
                           indices->dtype() == DataType::I64,
                           static_cast<int64_t>(input->size(0)),
                           page_size,
                           dsv4_flashmla_page_bytes(page_size)};
}

void run(void *planned_meta) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    deepseek_v4_flashmla_cache::launch_store_flashmla_raw_cache(planned->input->data(),
                                                                planned->input_dtype,
                                                                reinterpret_cast<uint8_t *>(planned->cache->data()),
                                                                planned->indices->data(),
                                                                planned->indices_i64,
                                                                planned->tokens,
                                                                planned->page_size,
                                                                planned->page_bytes,
                                                                context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_store_flashmla_raw_cache_graph_impl

namespace deepseek_v4_store_flashmla_raw_cache_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4StoreFlashMlaRawCacheKernel,
                                       &deepseek_v4_store_flashmla_raw_cache_graph_impl::plan,
                                       &deepseek_v4_store_flashmla_raw_cache_graph_impl::run,
                                       &deepseek_v4_store_flashmla_raw_cache_graph_impl::cleanup);
} // namespace deepseek_v4_store_flashmla_raw_cache_register


DeepseekV4IndexerRotate128Kernel::DeepseekV4IndexerRotate128Kernel(Tensor input, bool apply_scale) {
    INFINICORE_GRAPH_OP_DISPATCH(input->device().getType(), input, apply_scale);
}

void DeepseekV4IndexerRotate128Kernel::execute(Tensor input, bool apply_scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4IndexerRotate128Kernel, input, apply_scale);
}

namespace deepseek_v4_indexer_rotate_128_graph_impl {

struct PlannedMeta {
    graph::GraphTensor input;
    int input_dtype;
    int64_t rows;
    bool apply_scale;
};

void *plan(Tensor input, bool apply_scale) {
    check_indexer_rotate_shapes(input);
    if (input->size(input->ndim() - 1) != static_cast<size_t>(kDsv4IndexerInputDim)) {
        throw std::runtime_error("DeepseekV4IndexerRotate128Kernel only supports last dimension 128.");
    }
    if (!input->is_contiguous()) {
        throw std::runtime_error("DeepseekV4IndexerRotate128Kernel expects contiguous tensors.");
    }
    return new PlannedMeta{graph::GraphTensor(input),
                           dsv4_scalar_type_for_kernel(input, "DeepseekV4IndexerRotate128Kernel"),
                           static_cast<int64_t>(input->numel() / kDsv4IndexerInputDim),
                           apply_scale};
}

void run(void *planned_meta) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    deepseek_v4_flashmla_cache::launch_indexer_rotate_128(
        planned->input->data(),
        planned->input_dtype,
        planned->rows,
        planned->apply_scale,
        context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("DeepseekV4IndexerRotate128Kernel requires a HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_indexer_rotate_128_graph_impl

namespace deepseek_v4_indexer_rotate_128_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4IndexerRotate128Kernel,
                                       &deepseek_v4_indexer_rotate_128_graph_impl::plan,
                                       &deepseek_v4_indexer_rotate_128_graph_impl::run,
                                       &deepseek_v4_indexer_rotate_128_graph_impl::cleanup);
} // namespace deepseek_v4_indexer_rotate_128_register

DeepseekV4StoreIndexerRawCacheKernel::DeepseekV4StoreIndexerRawCacheKernel(const Tensor &input,
                                                                           Tensor cache,
                                                                           const Tensor &indices,
                                                                           int page_size) {
    INFINICORE_GRAPH_OP_DISPATCH(input->device().getType(), input, cache, indices, page_size);
}

void DeepseekV4StoreIndexerRawCacheKernel::execute(const Tensor &input,
                                                   Tensor cache,
                                                   const Tensor &indices,
                                                   int page_size) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4StoreIndexerRawCacheKernel, input, cache, indices, page_size);
}

namespace deepseek_v4_store_indexer_raw_cache_graph_impl {

struct PlannedMeta {
    graph::GraphTensor input;
    graph::GraphTensor cache;
    graph::GraphTensor indices;
    int input_dtype;
    bool indices_i64;
    int64_t tokens;
    int page_size;
    int64_t page_bytes;
};

void *plan(const Tensor &input, Tensor cache, const Tensor &indices, int page_size) {
    check_indexer_raw_store_shapes(input, cache, indices, page_size);
    if (!input->is_contiguous() || !cache->is_contiguous() || !indices->is_contiguous()) {
        throw std::runtime_error("DeepseekV4StoreIndexerRawCacheKernel expects contiguous tensors.");
    }
    return new PlannedMeta{graph::GraphTensor(input),
                           graph::GraphTensor(cache),
                           graph::GraphTensor(indices),
                           dsv4_scalar_type_for_kernel(input, "DeepseekV4StoreIndexerRawCacheKernel"),
                           indices->dtype() == DataType::I64,
                           static_cast<int64_t>(input->size(0)),
                           page_size,
                           dsv4_indexer_page_bytes(page_size)};
}

void run(void *planned_meta) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    deepseek_v4_flashmla_cache::launch_store_indexer_raw_cache(
        planned->input->data(),
        planned->input_dtype,
        reinterpret_cast<uint8_t *>(planned->cache->data()),
        planned->indices->data(),
        planned->indices_i64,
        planned->tokens,
        planned->page_size,
        planned->page_bytes,
        context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("DeepseekV4StoreIndexerRawCacheKernel requires a HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_store_indexer_raw_cache_graph_impl

namespace deepseek_v4_store_indexer_raw_cache_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4StoreIndexerRawCacheKernel,
                                       &deepseek_v4_store_indexer_raw_cache_graph_impl::plan,
                                       &deepseek_v4_store_indexer_raw_cache_graph_impl::run,
                                       &deepseek_v4_store_indexer_raw_cache_graph_impl::cleanup);
} // namespace deepseek_v4_store_indexer_raw_cache_register

void deepseek_v4_store_flashmla_raw_cache_kernel_(const Tensor &input,
                                                  Tensor cache,
                                                  const Tensor &indices,
                                                  int page_size) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (input->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_kernel_ expects HYGON tensors in this build.");
    }
#else
    if (input->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_kernel_ expects NVIDIA tensors in this build.");
    }
#endif
    check_raw_store_shapes(input, cache, indices, page_size);
    if (!input->is_contiguous() || !cache->is_contiguous() || !indices->is_contiguous()) {
        throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_kernel_ expects contiguous tensors.");
    }
    if (input->dtype() == DataType::F32) {
        throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_kernel_ expects bf16/fp16 input because RoPE bytes are copied verbatim.");
    }
    DeepseekV4StoreFlashMlaRawCacheKernel::execute(input, cache, indices, page_size);
#else
    (void)input;
    (void)cache;
    (void)indices;
    (void)page_size;
    throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_indexer_rotate_128_kernel_(Tensor input, bool apply_scale) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (input->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_indexer_rotate_128_kernel_ expects HYGON tensors in this build.");
    }
#else
    if (input->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_indexer_rotate_128_kernel_ expects NVIDIA tensors in this build.");
    }
#endif
    check_indexer_rotate_shapes(input);
    if (input->size(input->ndim() - 1) != static_cast<size_t>(kDsv4IndexerInputDim)) {
        throw std::runtime_error("deepseek_v4_indexer_rotate_128_kernel_ only supports last dimension 128.");
    }
    if (!input->is_contiguous()) {
        throw std::runtime_error("deepseek_v4_indexer_rotate_128_kernel_ expects contiguous tensors.");
    }
    const int64_t rows = static_cast<int64_t>(input->numel() / kDsv4IndexerInputDim);
    DeepseekV4IndexerRotate128Kernel::execute(input, apply_scale);
#else
    (void)input;
    (void)apply_scale;
    throw std::runtime_error("deepseek_v4_indexer_rotate_128_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_store_indexer_raw_cache_kernel_(const Tensor &input,
                                                 Tensor cache,
                                                 const Tensor &indices,
                                                 int page_size) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (input->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_store_indexer_raw_cache_kernel_ expects HYGON tensors in this build.");
    }
#else
    if (input->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_store_indexer_raw_cache_kernel_ expects NVIDIA tensors in this build.");
    }
#endif
    check_indexer_raw_store_shapes(input, cache, indices, page_size);
    if (!input->is_contiguous() || !cache->is_contiguous() || !indices->is_contiguous()) {
        throw std::runtime_error("deepseek_v4_store_indexer_raw_cache_kernel_ expects contiguous tensors.");
    }
    DeepseekV4StoreIndexerRawCacheKernel::execute(input, cache, indices, page_size);
#else
    (void)input;
    (void)cache;
    (void)indices;
    (void)page_size;
    throw std::runtime_error("deepseek_v4_store_indexer_raw_cache_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_flashmla_cache_indexer_(const Tensor &req_to_token,
                                         const Tensor &req_pool_indices,
                                         const Tensor &page_kernel_lens,
                                         std::optional<Tensor> kv_start_idx,
                                         Tensor kv_indices,
                                         int req_to_token_stride,
                                         int kv_indices_stride,
                                         int page_size) {
    deepseek_v4_create_flashmla_kv_indices_(req_to_token,
                                            req_pool_indices,
                                            page_kernel_lens,
                                            kv_start_idx,
                                            kv_indices,
                                            req_to_token_stride,
                                            kv_indices_stride,
                                            page_size);
}

} // namespace infinicore::op

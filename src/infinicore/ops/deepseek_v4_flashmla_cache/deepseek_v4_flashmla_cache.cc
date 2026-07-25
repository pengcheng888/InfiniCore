#include "infinicore/ops/deepseek_v4_flashmla_cache.hpp"

#include "deepseek_v4_flashmla_cache_kernel.hpp"

#include "infinicore/ops/deepseek_v4_concat_and_cache_mla.hpp"
#include "infinicore/ops/deepseek_v4_create_flashmla_kv_indices.hpp"

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

#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
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

void *current_accelerator_stream() {
#if defined(ENABLE_HYGON_API)
    return reinterpret_cast<void *>(infinicore::adaptor::get_hip_stream().stream());
#else
    return reinterpret_cast<void *>(infinicore::adaptor::get_cuda_stream().stream());
#endif
}
#endif

} // namespace

void deepseek_v4_fused_store_flashmla_cache_(const Tensor &kv_c,
                                             const Tensor &k_pe,
                                             Tensor kv_cache,
                                             const Tensor &slot_mapping,
                                             const std::string &kv_cache_dtype,
                                             const Tensor &scale) {
    deepseek_v4_concat_and_cache_mla_(kv_c,
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
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (input->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (input->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    check_raw_store_shapes(input, cache, indices, page_size);
    auto input_at = infinicore::adaptor::to_aten_tensor(input).contiguous();
    auto cache_at = infinicore::adaptor::to_aten_tensor(cache);
    auto indices_at = infinicore::adaptor::to_aten_tensor(indices).to(at::kLong);

    auto valid_mask = indices_at >= 0;
    auto valid_rows = at::nonzero(valid_mask).reshape({-1});
    if (valid_rows.numel() == 0) {
        return;
    }
    if (valid_rows.numel() != indices_at.numel()) {
        input_at = input_at.index_select(0, valid_rows);
        indices_at = indices_at.index_select(0, valid_rows);
    }

    const int64_t num_tokens = input_at.size(0);
    if (num_tokens == 0) {
        return;
    }
    const int64_t page_size_i64 = static_cast<int64_t>(page_size);
    const int64_t page_bytes = dsv4_flashmla_page_bytes(page_size);
    const auto page = at::floor_divide(indices_at, page_size_i64);
    const auto offset = at::remainder(indices_at, page_size_i64);

    auto no_pe = input_at.slice(1, 0, kDsv4FlashMlaNopeDim)
                     .reshape({num_tokens, 7, 64})
                     .to(at::kFloat);
    auto scale_raw = at::clamp_min(at::amax(at::abs(no_pe), {-1}, true), 1.0e-4) / kDsv4Fp8E4M3Max;
    auto scale_exp = at::clamp(at::ceil(at::log2(scale_raw)).to(at::kInt) + 127, 0, 255).to(at::kByte);
    auto scale = at::pow(at::scalar_tensor(2.0, scale_raw.options()), scale_exp.to(at::kFloat) - 127.0);
    auto quant_fp8 = at::clamp(no_pe / scale, -kDsv4Fp8E4M3Max, kDsv4Fp8E4M3Max)
                         .to(at::ScalarType::Float8_e4m3fn)
                         .view(at::kByte)
                         .reshape({num_tokens, kDsv4FlashMlaNopeDim});

    auto rope_bytes = input_at.slice(1, kDsv4FlashMlaNopeDim, kDsv4FlashMlaInputDim)
                          .contiguous()
                          .view(at::kByte)
                          .reshape({num_tokens, kDsv4FlashMlaRopeDim * 2});

    auto flat_cache = cache_at.reshape({cache_at.size(0) * cache_at.size(1)});
    auto token_base = page * page_bytes + offset * kDsv4FlashMlaValueBytesPerToken;
    auto nope_cols = arange_like_cols(kDsv4FlashMlaNopeDim, indices_at);
    auto rope_cols = arange_like_cols(kDsv4FlashMlaRopeDim * 2, indices_at);
    auto scale_cols = arange_like_cols(7, indices_at);

    auto nope_pos = (token_base.unsqueeze(1) + nope_cols.unsqueeze(0)).reshape({-1});
    auto rope_pos = (token_base.unsqueeze(1) + kDsv4FlashMlaNopeDim + rope_cols.unsqueeze(0)).reshape({-1});
    auto scale_pos = (page * page_bytes + kDsv4FlashMlaValueBytesPerToken * page_size_i64 +
                      offset * kDsv4FlashMlaScaleBytesPerToken)
                         .unsqueeze(1) +
                     scale_cols.unsqueeze(0);

    flat_cache.index_put_({nope_pos}, quant_fp8.reshape({-1}));
    flat_cache.index_put_({rope_pos}, rope_bytes.reshape({-1}));
    flat_cache.index_put_({scale_pos.reshape({-1})}, scale_exp.reshape({num_tokens, 7}).reshape({-1}));
#else
    (void)input;
    (void)cache;
    (void)indices;
    (void)page_size;
    throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}


void deepseek_v4_indexer_rotate_(Tensor input, bool apply_scale) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (input->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_indexer_rotate_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (input->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_indexer_rotate_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    check_indexer_rotate_shapes(input);
    auto input_at = infinicore::adaptor::to_aten_tensor(input);
    const int64_t dim = input_at.size(input_at.dim() - 1);
    const int64_t rows = input_at.numel() / dim;
    auto work = input_at.reshape({rows, dim}).to(at::kFloat).contiguous();
    for (int64_t span = 1; span < dim; span <<= 1) {
        auto view = work.reshape({rows, dim / (2 * span), 2, span});
        auto even = view.select(2, 0).clone();
        auto odd = view.select(2, 1).clone();
        view.select(2, 0).copy_(even + odd);
        view.select(2, 1).copy_(even - odd);
    }
    if (apply_scale) {
        work.mul_(1.0 / std::sqrt(static_cast<double>(dim)));
    }
    input_at.copy_(work.reshape(input_at.sizes()).to(input_at.scalar_type()));
#else
    (void)input;
    (void)apply_scale;
    throw std::runtime_error("deepseek_v4_indexer_rotate_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_store_indexer_raw_cache_(const Tensor &input,
                                          Tensor cache,
                                          const Tensor &indices,
                                          int page_size) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (input->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_store_indexer_raw_cache_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (input->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_store_indexer_raw_cache_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    check_indexer_raw_store_shapes(input, cache, indices, page_size);
    auto input_at = infinicore::adaptor::to_aten_tensor(input).contiguous();
    auto cache_at = infinicore::adaptor::to_aten_tensor(cache);
    auto indices_at = infinicore::adaptor::to_aten_tensor(indices).to(at::kLong);

    auto valid_mask = indices_at >= 0;
    auto valid_rows = at::nonzero(valid_mask).reshape({-1});
    if (valid_rows.numel() == 0) {
        return;
    }
    if (valid_rows.numel() != indices_at.numel()) {
        input_at = input_at.index_select(0, valid_rows);
        indices_at = indices_at.index_select(0, valid_rows);
    }

    const int64_t num_tokens = input_at.size(0);
    if (num_tokens == 0) {
        return;
    }
    const int64_t page_size_i64 = static_cast<int64_t>(page_size);
    const int64_t page_bytes = dsv4_indexer_page_bytes(page_size);
    const auto page = at::floor_divide(indices_at, page_size_i64);
    const auto offset = at::remainder(indices_at, page_size_i64);

    auto input_float = input_at.to(at::kFloat);
    const double fp8_max = dsv4_indexer_fp8_max();
    auto scale = at::clamp_min(at::amax(at::abs(input_float), {-1}, true), 1.0e-4) / fp8_max;
    auto quant_fp8 = at::clamp(input_float / scale, -fp8_max, fp8_max)
                         .to(dsv4_indexer_fp8_dtype())
                         .view(at::kByte)
                         .reshape({num_tokens, kDsv4IndexerInputDim});
    auto scale_bytes = scale.reshape({num_tokens}).contiguous().view(at::kByte).reshape({num_tokens, kDsv4IndexerScaleBytesPerToken});

    auto flat_cache = cache_at.reshape({cache_at.size(0) * cache_at.size(1)});
    auto token_base = page * page_bytes + offset * kDsv4IndexerInputDim;
    auto value_cols = arange_like_cols(kDsv4IndexerInputDim, indices_at);
    auto scale_cols = arange_like_cols(kDsv4IndexerScaleBytesPerToken, indices_at);
    auto value_pos = (token_base.unsqueeze(1) + value_cols.unsqueeze(0)).reshape({-1});
    auto scale_pos = (page * page_bytes + kDsv4IndexerInputDim * page_size_i64 +
                      offset * kDsv4IndexerScaleBytesPerToken)
                         .unsqueeze(1) +
                     scale_cols.unsqueeze(0);

    flat_cache.index_put_({value_pos}, quant_fp8.reshape({-1}));
    flat_cache.index_put_({scale_pos.reshape({-1})}, scale_bytes.reshape({-1}));
#else
    (void)input;
    (void)cache;
    (void)indices;
    (void)page_size;
    throw std::runtime_error("deepseek_v4_store_indexer_raw_cache_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}


void deepseek_v4_store_flashmla_raw_cache_kernel_(const Tensor &input,
                                                  Tensor cache,
                                                  const Tensor &indices,
                                                  int page_size) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (input->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_kernel_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (input->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_kernel_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    check_raw_store_shapes(input, cache, indices, page_size);
    if (!input->is_contiguous() || !cache->is_contiguous() || !indices->is_contiguous()) {
        throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_kernel_ expects contiguous tensors.");
    }
    if (input->dtype() == DataType::F32) {
        throw std::runtime_error("deepseek_v4_store_flashmla_raw_cache_kernel_ expects bf16/fp16 input because RoPE bytes are copied verbatim.");
    }
    deepseek_v4_flashmla_cache::launch_store_flashmla_raw_cache(
        input->data(),
        dsv4_scalar_type_for_kernel(input, "deepseek_v4_store_flashmla_raw_cache_kernel_"),
        reinterpret_cast<uint8_t *>(cache->data()),
        indices->data(),
        indices->dtype() == DataType::I64,
        static_cast<int64_t>(input->size(0)),
        page_size,
        dsv4_flashmla_page_bytes(page_size),
        current_accelerator_stream());
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
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (input->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_indexer_rotate_128_kernel_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    check_indexer_rotate_shapes(input);
    if (input->size(input->ndim() - 1) != static_cast<size_t>(kDsv4IndexerInputDim)) {
        throw std::runtime_error("deepseek_v4_indexer_rotate_128_kernel_ only supports last dimension 128.");
    }
    if (!input->is_contiguous()) {
        throw std::runtime_error("deepseek_v4_indexer_rotate_128_kernel_ expects contiguous tensors.");
    }
    const int64_t rows = static_cast<int64_t>(input->numel() / kDsv4IndexerInputDim);
    deepseek_v4_flashmla_cache::launch_indexer_rotate_128(
        input->data(),
        dsv4_scalar_type_for_kernel(input, "deepseek_v4_indexer_rotate_128_kernel_"),
        rows,
        apply_scale,
        current_accelerator_stream());
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
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (input->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_store_indexer_raw_cache_kernel_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    check_indexer_raw_store_shapes(input, cache, indices, page_size);
    if (!input->is_contiguous() || !cache->is_contiguous() || !indices->is_contiguous()) {
        throw std::runtime_error("deepseek_v4_store_indexer_raw_cache_kernel_ expects contiguous tensors.");
    }
    deepseek_v4_flashmla_cache::launch_store_indexer_raw_cache(
        input->data(),
        dsv4_scalar_type_for_kernel(input, "deepseek_v4_store_indexer_raw_cache_kernel_"),
        reinterpret_cast<uint8_t *>(cache->data()),
        indices->data(),
        indices->dtype() == DataType::I64,
        static_cast<int64_t>(input->size(0)),
        page_size,
        dsv4_indexer_page_bytes(page_size),
        current_accelerator_stream());
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

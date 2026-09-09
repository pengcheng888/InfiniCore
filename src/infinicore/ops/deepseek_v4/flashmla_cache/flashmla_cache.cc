#include "infinicore/ops/deepseek_v4/flashmla_cache.hpp"

#include "../graph_deferred.hpp"

#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#include "../../../utils.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#endif
#endif

#include <cmath>
#include <stdexcept>
#include <string>

namespace infinicore::op::deepseek_v4 {
namespace {

constexpr int64_t kDsv4FlashMlaNopeDim = 448;
constexpr int64_t kDsv4FlashMlaRopeDim = 64;
constexpr int64_t kDsv4FlashMlaInputDim = kDsv4FlashMlaNopeDim + kDsv4FlashMlaRopeDim;
constexpr int64_t kDsv4FlashMlaValueBytesPerToken = 576;
constexpr int64_t kDsv4FlashMlaScaleBytesPerToken = 8;
constexpr int64_t kDsv4FlashMlaBytesPerToken = kDsv4FlashMlaValueBytesPerToken + kDsv4FlashMlaScaleBytesPerToken;
constexpr double kDsv4Fp8E4M3Max = 448.0;

int64_t div_ceil_i64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

int64_t dsv4_flashmla_page_bytes(int page_size) {
    const auto bytes = kDsv4FlashMlaBytesPerToken * static_cast<int64_t>(page_size);
    return div_ceil_i64(bytes, kDsv4FlashMlaValueBytesPerToken) * kDsv4FlashMlaValueBytesPerToken;
}

void check_raw_store_shapes(const Tensor &input, const Tensor &cache, const Tensor &indices, int page_size) {
    if (!input || !cache || !indices) {
        throw std::runtime_error("store_flashmla_raw_cache_ expects non-empty input/cache/indices.");
    }
    if (input->ndim() != 2 || input->size(1) != static_cast<size_t>(kDsv4FlashMlaInputDim)) {
        throw std::runtime_error("store_flashmla_raw_cache_ expects input [tokens, 512].");
    }
    if (cache->ndim() != 2 || cache->dtype() != DataType::U8) {
        throw std::runtime_error("store_flashmla_raw_cache_ expects uint8 raw cache [blocks, page_bytes].");
    }
    if (indices->ndim() != 1 || indices->numel() != input->size(0)) {
        throw std::runtime_error("store_flashmla_raw_cache_ expects indices [tokens].");
    }
    if (indices->dtype() != DataType::I32 && indices->dtype() != DataType::I64) {
        throw std::runtime_error("store_flashmla_raw_cache_ indices must be int32/int64.");
    }
    if (page_size <= 0 || (page_size & (page_size - 1)) != 0) {
        throw std::runtime_error("store_flashmla_raw_cache_ page_size must be a positive power of two.");
    }
    const auto expected_page_bytes = static_cast<size_t>(dsv4_flashmla_page_bytes(page_size));
    if (cache->size(1) != expected_page_bytes) {
        throw std::runtime_error("store_flashmla_raw_cache_ raw cache page_bytes mismatch.");
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(input, cache, indices);
    if (!input->is_contiguous() || !cache->is_contiguous() || !indices->is_contiguous()) {
        throw std::runtime_error("store_flashmla_raw_cache_ expects contiguous tensors.");
    }
}

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
at::Tensor arange_like_cols(int64_t cols, const at::Tensor &ref) {
    return at::arange(cols, ref.options().dtype(at::kLong));
}
#endif

} // namespace

void store_flashmla_raw_cache_aten_(const Tensor &input,
                                    Tensor cache,
                                    const Tensor &indices,
                                    int page_size);

void store_flashmla_raw_cache_(const Tensor &input,
                               Tensor cache,
                               const Tensor &indices,
                               int page_size) {
    auto input_graph = graph::GraphTensor(input);
    auto cache_graph = graph::GraphTensor(cache);
    auto indices_graph = graph::GraphTensor(indices);
    detail::record_or_run_host_graph_op([input_graph, cache_graph, indices_graph, page_size]() mutable {
        store_flashmla_raw_cache_aten_(input_graph, cache_graph, indices_graph, page_size);
    });
}

void store_flashmla_raw_cache_aten_(const Tensor &input,
                                    Tensor cache,
                                    const Tensor &indices,
                                    int page_size) {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    if (input->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("store_flashmla_raw_cache_ expects HYGON tensors in this build.");
    }
    if (input->dtype() == DataType::F32) {
        throw std::runtime_error("store_flashmla_raw_cache_ expects bf16/fp16 input because RoPE bytes are copied verbatim.");
    }
    check_raw_store_shapes(input, cache, indices, page_size);
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());

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
    auto scale_pos = (page * page_bytes + kDsv4FlashMlaValueBytesPerToken * page_size_i64
                      + offset * kDsv4FlashMlaScaleBytesPerToken)
                         .unsqueeze(1)
                   + scale_cols.unsqueeze(0);

    flat_cache.index_put_({nope_pos}, quant_fp8.reshape({-1}));
    flat_cache.index_put_({rope_pos}, rope_bytes.reshape({-1}));
    flat_cache.index_put_({scale_pos.reshape({-1})}, scale_exp.reshape({num_tokens, 7}).reshape({-1}));
#else
    (void)input;
    (void)cache;
    (void)indices;
    (void)page_size;
    throw std::runtime_error("store_flashmla_raw_cache_ requires an ATen-enabled HYGON build.");
#endif
}

} // namespace infinicore::op::deepseek_v4

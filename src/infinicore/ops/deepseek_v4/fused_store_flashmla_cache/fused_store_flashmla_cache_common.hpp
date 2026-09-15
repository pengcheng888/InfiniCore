#pragma once

#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/ops/deepseek_v4/fused_store_flashmla_cache.hpp"

#include "../../../utils.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace infinicore::op::deepseek_v4::fused_store_flashmla_cache_detail {

constexpr int64_t kNopeDim = 448;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kInputDim = kNopeDim + kRopeDim;
constexpr int64_t kValueBytesPerToken = 576;
constexpr int64_t kScaleBytesPerToken = 8;
constexpr int64_t kBytesPerToken = kValueBytesPerToken + kScaleBytesPerToken;

static inline int64_t page_bytes(int page_size) {
    const auto bytes = kBytesPerToken * static_cast<int64_t>(page_size);
    return ((bytes + kValueBytesPerToken - 1) / kValueBytesPerToken) * kValueBytesPerToken;
}

static inline void check_shapes(const Tensor &input,
                                const Tensor &cache,
                                const Tensor &indices,
                                int page_size) {
    if (!input || !cache || !indices) {
        throw std::runtime_error("fused_store_flashmla_cache_ expects non-empty input/cache/indices.");
    }
    if (input->ndim() != 2 || input->size(1) != static_cast<size_t>(kInputDim)) {
        throw std::runtime_error("fused_store_flashmla_cache_ expects input [tokens, 512].");
    }
    if (input->dtype() != DataType::BF16 && input->dtype() != DataType::F16) {
        throw std::runtime_error("fused_store_flashmla_cache_ expects bf16/fp16 input.");
    }
    if (cache->ndim() != 2 || cache->dtype() != DataType::U8) {
        throw std::runtime_error("fused_store_flashmla_cache_ expects uint8 raw cache [blocks, page_bytes].");
    }
    if (indices->ndim() != 1 || indices->numel() != input->size(0)) {
        throw std::runtime_error("fused_store_flashmla_cache_ expects indices [tokens].");
    }
    if (indices->dtype() != DataType::I32 && indices->dtype() != DataType::I64) {
        throw std::runtime_error("fused_store_flashmla_cache_ indices must be int32/int64.");
    }
    if (page_size <= 0 || (page_size & (page_size - 1)) != 0) {
        throw std::runtime_error("fused_store_flashmla_cache_ page_size must be a positive power of two.");
    }
    if (cache->size(1) != static_cast<size_t>(page_bytes(page_size))) {
        throw std::runtime_error("fused_store_flashmla_cache_ raw cache page_bytes mismatch.");
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(input, cache, indices);
    if (!input->is_contiguous() || !cache->is_contiguous() || !indices->is_contiguous()) {
        throw std::runtime_error("fused_store_flashmla_cache_ expects contiguous tensors.");
    }
}

} // namespace infinicore::op::deepseek_v4::fused_store_flashmla_cache_detail

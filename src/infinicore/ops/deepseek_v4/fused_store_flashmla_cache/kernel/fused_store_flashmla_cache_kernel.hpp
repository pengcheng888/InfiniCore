#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4::fused_store_flashmla_cache {

enum ScalarType : int {
    BF16 = 0,
    F16 = 1,
};

void launch_fused_store_flashmla_cache(const void *input,
                                       int input_dtype,
                                       uint8_t *cache,
                                       const void *indices,
                                       bool indices_i64,
                                       int64_t num_tokens,
                                       int page_size,
                                       int64_t page_bytes,
                                       void *stream);

} // namespace infinicore::op::deepseek_v4::fused_store_flashmla_cache

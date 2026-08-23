#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_flashmla_cache {

enum Dsv4ScalarType : int {
    kDsv4BF16 = 0,
    kDsv4F16 = 1,
    kDsv4F32 = 2,
};

void launch_indexer_rotate_128(void *input,
                               int dtype,
                               int64_t rows,
                               bool apply_scale,
                               void *stream);

void launch_store_indexer_raw_cache(const void *input,
                                    int input_dtype,
                                    uint8_t *cache,
                                    const void *indices,
                                    bool indices_i64,
                                    int64_t num_tokens,
                                    int page_size,
                                    int64_t page_bytes,
                                    void *stream);

void launch_store_flashmla_raw_cache(const void *input,
                                     int input_dtype,
                                     uint8_t *cache,
                                     const void *indices,
                                     bool indices_i64,
                                     int64_t num_tokens,
                                     int page_size,
                                     int64_t page_bytes,
                                     void *stream);

} // namespace infinicore::op::deepseek_v4_flashmla_cache

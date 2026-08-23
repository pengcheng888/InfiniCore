#pragma once

#include "infinicore/dtype.hpp"

#include <cstdint>

namespace infinicore::op::deepseek_v4_embedding_and_hc_expand_kernel_impl {

void launch_embedding(void *out,
                      const void *input,
                      const void *weight,
                      int64_t tokens,
                      int64_t hc_mult,
                      int64_t hidden,
                      int64_t vocab,
                      DataType out_dtype,
                      DataType input_dtype,
                      void *stream);

} // namespace infinicore::op::deepseek_v4_embedding_and_hc_expand_kernel_impl

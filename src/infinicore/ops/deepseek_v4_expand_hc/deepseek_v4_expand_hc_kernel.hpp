#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_expand_hc_impl {

void launch_expand_hc(void *output,
                      const void *input,
                      int64_t tokens,
                      int64_t hc,
                      int64_t hidden,
                      int element_size,
                      void *stream);

} // namespace infinicore::op::deepseek_v4_expand_hc_impl

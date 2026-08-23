#pragma once

#include "infinicore/ops/deepseek_v4_fused_rope.hpp"

#include <cstdint>
#include <optional>

namespace infinicore::op {

namespace deepseek_v4_fused_rope_kernel {

enum Dsv4ScalarType : int {
    kDsv4BF16 = 0,
    kDsv4F16 = 1,
    kDsv4F32 = 2,
};

void launch_fused_rope(void *tensor,
                       int dtype,
                       const void *freqs_cis,
                       const void *positions,
                       bool positions_i64,
                       int64_t tokens,
                       int64_t heads,
                       int64_t stride_token,
                       int64_t stride_head,
                       bool inverse,
                       void *stream);

} // namespace deepseek_v4_fused_rope_kernel

} // namespace infinicore::op

#pragma once

#include "infinicore/ops/deepseek_v4_fused_rope.hpp"

#include <cstdint>
#include <optional>

namespace infinicore::op {

void deepseek_v4_fused_rope_kernel_(Tensor query,
                                    std::optional<Tensor> key,
                                    const Tensor &freqs_cis,
                                    const Tensor &positions,
                                    bool inverse);

namespace deepseek_v4_fused_rope_kernel_native {

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

} // namespace deepseek_v4_fused_rope_kernel_native

} // namespace infinicore::op

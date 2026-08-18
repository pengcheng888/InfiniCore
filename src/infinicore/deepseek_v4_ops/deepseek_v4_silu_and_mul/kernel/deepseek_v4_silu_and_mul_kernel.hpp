#pragma once

#include "infinicore/dtype.hpp"

#include <cstdint>

namespace infinicore::op::deepseek_v4_silu_and_mul_impl {

void launch_silu_and_mul(void *out,
                         const void *x,
                         int64_t tokens,
                         int64_t hidden,
                         DataType dtype,
                         void *stream);

} // namespace infinicore::op::deepseek_v4_silu_and_mul_impl

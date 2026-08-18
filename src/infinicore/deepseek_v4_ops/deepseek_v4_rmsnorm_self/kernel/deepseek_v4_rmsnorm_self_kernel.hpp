#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_rmsnorm_self_native {

enum Dsv4ScalarType : int {
    kDsv4BF16 = 0,
    kDsv4F16 = 1,
    kDsv4F32 = 2,
};

void launch_rmsnorm_self(void *out,
                         const void *input,
                         int dtype,
                         int64_t rows,
                         int64_t dim,
                         float epsilon,
                         void *stream);

} // namespace infinicore::op::deepseek_v4_rmsnorm_self_native

#pragma once

#include <cstdint>

#include "infinicore/dtype.hpp"

namespace infinicore::op::deepseek_v4_lmslim_rocblas_linear_w8a8_impl {

void launch_apply_scales(void *output,
                         const int32_t *accum,
                         const float *input_scale,
                         const float *weight_scale,
                         const void *bias,
                         int64_t m,
                         int64_t n,
                         infinicore::DataType output_dtype,
                         void *stream);

} // namespace infinicore::op::deepseek_v4_lmslim_rocblas_linear_w8a8_impl

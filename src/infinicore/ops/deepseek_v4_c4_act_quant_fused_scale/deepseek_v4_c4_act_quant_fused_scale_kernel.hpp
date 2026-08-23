#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_c4_act_quant_fused_scale {

enum Dsv4ScalarType : int {
    kDsv4BF16 = 0,
    kDsv4F16 = 1,
    kDsv4F32 = 2,
};

void launch_c4_act_quant_fused_scale(const void *q,
                                     int q_dtype,
                                     const void *weights,
                                     int weights_dtype,
                                     uint8_t *q_fp8,
                                     float *q_scale,
                                     float *fused_weights,
                                     int64_t rows,
                                     float weight_scale,
                                     void *stream);

} // namespace infinicore::op::deepseek_v4_c4_act_quant_fused_scale

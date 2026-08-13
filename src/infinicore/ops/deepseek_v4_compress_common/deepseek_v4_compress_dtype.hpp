#pragma once

#include <cstdint>

namespace infinicore::op::deepseek_v4_compress_common {

enum Dsv4ScalarType : int {
    kDsv4BF16 = 0,
    kDsv4F16 = 1,
    kDsv4F32 = 2,
};

} // namespace infinicore::op::deepseek_v4_compress_common

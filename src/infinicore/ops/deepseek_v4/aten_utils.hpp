#pragma once

#include "platform.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"

namespace infinicore::op::deepseek_v4::detail {

inline void prepare_aten_call(const Tensor &tensor, const char *op_name) {
    check_build_device(tensor, op_name);
    // Vendor ATen runtimes must execute on the current InfiniCore stream.
    adaptor::set_aten_stream_to_infinicore();
}

} // namespace infinicore::op::deepseek_v4::detail
#endif

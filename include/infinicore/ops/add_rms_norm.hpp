#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class AddRMSNorm {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, float);
    static void execute(Tensor y, Tensor a, Tensor b, Tensor weight, float epsilon = 1e-5f);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor add_rms_norm(Tensor a, Tensor b, Tensor weight, float epsilon = 1e-5f);
void add_rms_norm_(Tensor y, Tensor a, Tensor b, Tensor weight, float epsilon = 1e-5f);
} // namespace infinicore::op

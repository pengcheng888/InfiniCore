#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include "infinicore/common/utils.hpp"

namespace infinicore::op {
class Ones {

public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor y, Tensor x);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor ones(Tensor x);
void ones_(Tensor y, Tensor x);
} // namespace infinicore::op

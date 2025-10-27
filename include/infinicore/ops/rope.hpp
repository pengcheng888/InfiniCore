#pragma once

#include "common/op.hpp"

namespace infinicore::op {
class RoPE {

public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, Tensor);
    static void execute(Tensor y, Tensor x, Tensor pos_ids, Tensor sin_table, Tensor cos_table);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor rope(Tensor x, Tensor pos_ids, Tensor sin_table, Tensor cos_table);
void rope_(Tensor y, Tensor x, Tensor pos_ids, Tensor sin_table, Tensor cos_table);
} // namespace infinicore::op

#include "infinicore/ops/rope.hpp"

namespace infinicore::op {

common::OpDispatcher<RoPE::schema> &RoPE::dispatcher() {
    static common::OpDispatcher<RoPE::schema> dispatcher_;
    return dispatcher_;
};

void RoPE::execute(Tensor y, Tensor x, Tensor pos_ids, Tensor sin_table, Tensor cos_table) {
    dispatcher().lookup(context::getDevice().getType())(y, x, pos_ids, sin_table, cos_table);
}

Tensor rope(Tensor x, Tensor pos_ids, Tensor sin_table, Tensor cos_table) {
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    rope_(y, x, pos_ids, sin_table, cos_table);
    return y;
}

void rope_(Tensor y, Tensor x, Tensor pos_ids, Tensor sin_table, Tensor cos_table) {
    RoPE::execute(y, x, pos_ids, sin_table, cos_table);
}

} // namespace infinicore::op

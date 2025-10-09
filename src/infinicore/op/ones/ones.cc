#include "infinicore/op/ones.hpp"

namespace infinicore::op {

common::OpDispatcher<Ones::schema> &Ones::dispatcher() {
    static common::OpDispatcher<Ones::schema> dispatcher_;
    return dispatcher_;
};

void Ones::execute(Tensor y, Tensor x) {
    dispatcher().lookup(context::getDevice().getType())(y, x);
}

Tensor ones_py(const Shape &shape, const DataType &dtype, const Device &device, bool pin_memory) {
    auto x = infinicore::Tensor::empty(shape, dtype, device, pin_memory);
    Ones::execute(x, x);
    return x;
}

} // namespace infinicore::op

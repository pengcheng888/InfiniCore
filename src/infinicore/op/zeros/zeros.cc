#include "infinicore/op/zeros.hpp"

namespace infinicore::op {

common::OpDispatcher<Zeros::schema> &Zeros::dispatcher() {
    static common::OpDispatcher<Zeros::schema> dispatcher_;
    return dispatcher_;
};

void Zeros::execute(Tensor y, Tensor x) {
    dispatcher().lookup(context::getDevice().getType())(y, x);
}

Tensor zeros_py(const Shape &shape, const DataType &dtype, const Device &device, bool pin_memory) {
    auto x = infinicore::Tensor::empty(shape, dtype, device, pin_memory);
    Zeros::execute(x, x);
    return x;
}

} // namespace infinicore::op

#include "infinicore/op/ones.hpp"

namespace infinicore::op {

common::OpDispatcher<Ones::schema> &Ones::dispatcher() {
    static common::OpDispatcher<Ones::schema> dispatcher_;
    return dispatcher_;
};

void Ones::execute(Tensor y, Tensor x) {
    dispatcher().lookup(context::getDevice().getType())(y, x);
}

Tensor ones(Tensor x) {
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    ones_(y, x);
    return y;
}

void ones_(Tensor y, Tensor x) {
    Ones::execute(y, x);
}

} // namespace infinicore::op

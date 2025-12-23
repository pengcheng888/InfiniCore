#include "infinicore/ops/add_rms_norm.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<AddRMSNorm::schema> &AddRMSNorm::dispatcher() {
    static common::OpDispatcher<AddRMSNorm::schema> dispatcher_;
    return dispatcher_;
};

void AddRMSNorm::execute(Tensor y, Tensor a, Tensor b, Tensor weight, float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, a, b, weight);
    infinicore::context::setDevice(y->device());
    dispatcher().lookup(y->device().getType())(y, a, b, weight, epsilon);
}

Tensor add_rms_norm(Tensor a, Tensor b, Tensor weight, float epsilon) {
    auto y = Tensor::empty(a->shape(), a->dtype(), a->device());
    add_rms_norm_(y, a, b, weight, epsilon);
    return y;
}

void add_rms_norm_(Tensor y, Tensor a, Tensor b, Tensor weight, float epsilon) {
    AddRMSNorm::execute(y, a, b, weight, epsilon);
}

} // namespace infinicore::op

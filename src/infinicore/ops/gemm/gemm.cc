#include "infinicore/ops/gemm.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Gemm::schema> &Gemm::dispatcher() {
    static common::OpDispatcher<Gemm::schema> dispatcher_;
    return dispatcher_;
};

common::OpDispatcher<Gemm::plan_schema> &Gemm::plan_dispatcher() {
    static common::OpDispatcher<Gemm::plan_schema> dispatcher_;
    return dispatcher_;
}
common::OpDispatcher<Gemm::run_schema> &Gemm::run_dispatcher() {
    static common::OpDispatcher<Gemm::run_schema> dispatcher_;
    return dispatcher_;
}
common::OpDispatcher<Gemm::cleanup_schema> &Gemm::cleanup_dispatcher() {
    static common::OpDispatcher<Gemm::cleanup_schema> dispatcher_;
    return dispatcher_;
}

Gemm::Gemm(Tensor c, Tensor a, Tensor b, float alpha, float beta) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    planned_meta_ = plan_dispatcher().lookup(c->device().getType())(c, a, b, alpha, beta);
    runner_ = run_dispatcher().lookup(c->device().getType());
    deleter_ = cleanup_dispatcher().lookup(c->device().getType());
}

void Gemm::execute(Tensor c, Tensor a, Tensor b, float alpha, float beta) {

    auto op = std::make_shared<Gemm>(c, a, b, alpha, beta);
    if (context::isGraphRecording()) {
        context::addGraphOperator(op);
    } else {
        op->run();
    }
}

Tensor gemm(Tensor a, Tensor b, float alpha, float beta) {
    Shape shape = a->shape();
    Size size = a->ndim();
    shape[size - 1] = b->size(size - 1);
    auto c = Tensor::empty(shape, a->dtype(), a->device());
    gemm_(c, a, b, alpha, beta);
    return c;
}

void gemm_(Tensor c, Tensor a, Tensor b, float alpha, float beta) {
    Gemm::execute(c, a, b, alpha, beta);
}

} // namespace infinicore::op

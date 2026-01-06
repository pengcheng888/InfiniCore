#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Gemm : public graph::GraphOperator {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, float, float);
    using plan_schema = void *(*)(Tensor, Tensor, Tensor, float, float);

    Gemm(Tensor c, Tensor a, Tensor b, float alpha, float beta);

    static void execute(Tensor c, Tensor a, Tensor b, float alpha, float beta);

    static common::OpDispatcher<schema> &dispatcher();
    static common::OpDispatcher<plan_schema> &plan_dispatcher();
    static common::OpDispatcher<run_schema> &run_dispatcher();
    static common::OpDispatcher<cleanup_schema> &cleanup_dispatcher();
};

Tensor gemm(Tensor a, Tensor b, float alpha = 1.0f, float beta = 0.0f);
void gemm_(Tensor c, Tensor a, Tensor b, float alpha, float beta);

} // namespace infinicore::op

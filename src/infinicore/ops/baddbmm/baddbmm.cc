#include "infinicore/ops/baddbmm.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/gemm.hpp"

namespace infinicore::op {

Tensor baddbmm(std::optional<Tensor> input, Tensor mat1, Tensor mat2, float beta, float alpha) {
    Shape shape = mat1->shape(); // b×n×m
    Size p = mat2->shape()[2];   // b×m×p
    shape[2] = p;

    auto out = Tensor::empty(shape, mat1->dtype(), mat1->device());
    baddbmm_(out, input, mat1, mat2, beta, alpha);
    return out;
}

void baddbmm_(Tensor out, std::optional<Tensor> input, Tensor mat1, Tensor mat2, float beta, float alpha) {

    if (!input.has_value()) {
        gemm_(out, mat1, mat2, alpha, 0.0f);
    } else {
        out->copy_from(input.value());
        gemm_(out, mat1, mat2, alpha, beta);
    }
}

} // namespace infinicore::op

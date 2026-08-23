#include "infinicore/ops/deepseek_v4_embedding_and_hc_expand.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op {

namespace {

Shape output_shape_for(const Tensor &input, const Tensor &weight, int64_t hc_mult, const char *op_name) {
    if (weight->ndim() != 2) {
        throw std::runtime_error(std::string(op_name) + " expects weight shape [vocab, hidden].");
    }
    if (hc_mult <= 0) {
        throw std::runtime_error(std::string(op_name) + " expects hc_mult > 0.");
    }
    Shape output_shape = input->shape();
    output_shape.push_back(static_cast<size_t>(hc_mult));
    output_shape.push_back(weight->size(1));
    return output_shape;
}

} // namespace

Tensor deepseek_v4_embedding_and_hc_expand(const Tensor &input, const Tensor &weight, int64_t hc_mult) {
    auto out = Tensor::empty(output_shape_for(input, weight, hc_mult, "deepseek_v4_embedding_and_hc_expand"), weight->dtype(), weight->device());
    deepseek_v4_embedding_and_hc_expand_kernel_(out, input, weight, hc_mult);
    return out;
}

void deepseek_v4_embedding_and_hc_expand_(Tensor out, const Tensor &input, const Tensor &weight, int64_t hc_mult) {
    deepseek_v4_embedding_and_hc_expand_kernel_(out, input, weight, hc_mult);
    return;
}

} // namespace infinicore::op

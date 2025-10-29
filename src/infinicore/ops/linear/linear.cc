#include "infinicore/ops/linear.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/matmul.hpp"
#include "infinicore/ops/rearrange.hpp"

namespace infinicore::op {

void print(Shape &shape) {
    printf("\n");
    for (auto &v : shape) {
        printf("\t %ld", v);
    }
    printf("\n");
}

Tensor linear(Tensor input,
              Tensor weight) {

    Size ndim = input->ndim();
    Shape input_shape = input->shape();
    Shape weight_shape = weight->shape();

    Size num = input_shape[0];
    Size in_features = input_shape[ndim - 1];
    Size out_features = weight_shape[1];

    // y 是 (∗,out_features)
    auto y = Tensor::empty({num, out_features}, input->dtype(), input->device());
    matmul_(y, input, rearrange(weight));
    return y;
}

Tensor linear_bias(Tensor input,  //  (∗,in_features)
                   Tensor weight, // (in_features,out_features)
                   Tensor bias    // (out_features)
) {
    Tensor y = linear(input, weight);
    add_(y, y, bias);
    return y;
}

} // namespace infinicore::op

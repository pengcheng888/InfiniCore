#pragma once

#include "common/op.hpp"

namespace infinicore::op {

Tensor linear(Tensor input, Tensor weight);
Tensor linear_bias(Tensor input, Tensor weight, Tensor bias);
} // namespace infinicore::op

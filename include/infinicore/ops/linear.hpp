#pragma once

#include "common/op.hpp"
#include <pybind11/pybind11.h>

namespace infinicore::op {

Tensor linear(Tensor input, Tensor weight);
Tensor linear_bias(Tensor input, Tensor weight, Tensor bias);

Tensor linear2(Tensor input, Tensor weight, pybind11::object bias);

} // namespace infinicore::op

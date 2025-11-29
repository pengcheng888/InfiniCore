#pragma once

#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

Tensor baddbmm(std::optional<Tensor> input, Tensor batch1, Tensor batch2, float beta = 1.0f, float alpha = 1.0f);

void baddbmm_(Tensor out, std::optional<Tensor> input, Tensor batch1, Tensor batch2, float beta = 1.0f, float alpha = 1.0f);

} // namespace infinicore::op

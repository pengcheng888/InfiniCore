#include "rearrange.hpp"
#include <iostream>

namespace infinicore::py {

Tensor rearrange(const Tensor &x) {
    auto result = infinicore::op::rearrange(x.get());
    return Tensor{result};
}

void rearrange_(Tensor &y, const Tensor &x) {
    infinicore::op::rearrange_(y.get(), x.get());
}

} // namespace infinicore::py

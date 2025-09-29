#include "ones.hpp"
#include <iostream>

namespace infinicore::py {

Tensor ones(const Tensor &x) {
    auto result = infinicore::op::ones(x.get());
    return Tensor{result};
}

void ones_(Tensor &y, const Tensor &x) {
    infinicore::op::ones_(y.get(), x.get());
}

} // namespace infinicore::py

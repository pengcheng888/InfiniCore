#pragma once

#include "../tensor.hpp"

namespace infinicore::op {
class Ones {
public:
    static void execute(Tensor output);
};

Tensor ones();
void ones_(Tensor output);
} // namespace infinicore::op

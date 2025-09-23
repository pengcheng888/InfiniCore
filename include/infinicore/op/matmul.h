#pragma once

#include "../device.hpp"
#include "common/cache.hpp"
#include "common/op.hpp"
#include "infinicore/common/utils.hpp"
#include <array>
#include <vector>

namespace infinicore::op {
class Matmul {

public:
    using schema = void (*)(Tensor, Tensor, Tensor);
    static void execute(Tensor c, Tensor a, Tensor b);
    static common::OpDispatcher<schema> dispatcher;

    static common::OpCache<size_t, infiniopGemmDescriptor_t> caches;

    static void destroyMatmul(Matmul &matmul);
};

Tensor matmul(Tensor a, Tensor b);
void matmul_(Tensor c, Tensor a, Tensor b);
} // namespace infinicore::op

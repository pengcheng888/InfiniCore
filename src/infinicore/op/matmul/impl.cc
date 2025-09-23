#include "infinicore/op/matmul.h"
#include <infiniop.h>

#include "infinicore/common/utils.hpp"

namespace infinicore::op {

common::OpCache<size_t, infiniopGemmDescriptor_t> Matmul::caches(
    100, // capacity
    [](infiniopGemmDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyGemmDescriptor(desc));
            desc = nullptr;
        }
    });

common::OpDispatcher<Matmul::schema> Matmul::dispatcher;

void Matmul::execute(Tensor c, Tensor a, Tensor b) {
    dispatcher.lookup(context::getDevice().getType())(c, a, b);
}

void Matmul::destroyMatmul(Matmul &matmul) {
    matmul.caches.clear();
}

} // namespace infinicore::op

namespace infinicore::op::matmul_impl {
void infiniop(Tensor c, Tensor a, Tensor b) {
    size_t seed = hash_combine(c, b, a);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();
    auto &cache = Matmul::caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopGemmDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateGemmDescriptor(context::getInfiniopHandle(), &desc, c->desc(), a->desc(), b->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetGemmWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopGemm(
        desc, workspace->data(), workspace_size,
        c->data(), a->data(), b->data(), 1.f, 0.f, context::getStream()));
}

static bool registered = []() {
    Matmul::dispatcher.registerAll(infiniop);
    return true;
};
} // namespace infinicore::op::matmul_impl

namespace infinicore::op {
Tensor matmul(Tensor a, Tensor b) {
    auto c = Tensor::empty({a->size(0), b->size(1)}, a->dtype(), a->device());
    matmul_(c, a, b);
    return c;
}

void matmul_(Tensor c, Tensor a, Tensor b) {
    Matmul::execute(c, a, b);
}
} // namespace infinicore::op

#include "infinicore/common/utils.hpp"
#include "infinicore/op/common/cache.hpp"
#include "infinicore/op/zeros.hpp"
#include <infiniop.h>

namespace infinicore::op::zeros_impl::infiniop {

thread_local common::OpCache<size_t, infiniopZerosDescriptor_t> caches(
    100, // capacity
    [](infiniopZerosDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyZerosDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor y, Tensor x) {
    size_t seed = hash_combine(y, x);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopZerosDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateZerosDescriptor(context::getInfiniopHandle(), &desc, y->desc(), x->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetZerosWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopZeros(
        desc, workspace->data(), workspace_size,
        y->data(), x->data(), context::getStream()));
}

static bool registered = []() {
    Zeros::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::zeros_impl::infiniop

#include "infinicore/common/utils.hpp"
#include "infinicore/op/common/cache.hpp"
#include "infinicore/op/ones.hpp"
#include <infiniop.h>

namespace infinicore::op::ones_impl::infiniop {

thread_local common::OpCache<size_t, infiniopOnesDescriptor_t> caches(
    100, // capacity
    [](infiniopOnesDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyOnesDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor y, Tensor x) {
    size_t seed = hash_combine(y, x);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopOnesDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateOnesDescriptor(context::getInfiniopHandle(), &desc, y->desc(), x->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetOnesWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopOnes(
        desc, workspace->data(), workspace_size,
        y->data(), x->data(), context::getStream()));
}

static bool registered = []() {
    Ones::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::ones_impl::infiniop

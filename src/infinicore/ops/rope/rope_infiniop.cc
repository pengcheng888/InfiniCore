#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/rope.hpp"
#include <infiniop.h>

namespace infinicore::op::rope_impl::infiniop {

thread_local common::OpCache<size_t, infiniopRoPEDescriptor_t> caches(
    100, // capacity
    [](infiniopRoPEDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyRoPEDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor y, Tensor x, Tensor pos_ids, Tensor sin_table, Tensor cos_table) {
    size_t seed = hash_combine(y, x, pos_ids, sin_table, cos_table);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopRoPEDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateRoPEDescriptor(
            context::getInfiniopHandle(), &desc,
            y->desc(), x->desc(), pos_ids->desc(), sin_table->desc(), cos_table->desc(), INFINIOP_ROPE_ALGO_GPT_NEOX));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetRoPEWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopRoPE(
        desc, workspace->data(), workspace_size,
        y->data(), x->data(), pos_ids->data(), sin_table->data(), cos_table->data(),
        context::getStream()));
}

static bool registered = []() {
    RoPE::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::rope_impl::infiniop

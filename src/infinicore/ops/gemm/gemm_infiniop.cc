#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/gemm.hpp"
#include <infiniop.h>

namespace infinicore::op::gemm_impl::infiniop {
// A desc holder to make it a shared pointer that can auto clean-up
struct Descriptor {
    infiniopGemmDescriptor_t desc;
    Descriptor(infiniopGemmDescriptor_t desc) : desc(desc) {}
    ~Descriptor() {
        if (desc != nullptr) {
            infiniopDestroyGemmDescriptor(desc);
            desc = nullptr;
        }
    }
};

thread_local common::OpCache<size_t, std::shared_ptr<Descriptor>>
    caches(
        // capacity
        100,
        // on evict
        [](std::shared_ptr<Descriptor> &desc) {
            desc = nullptr;
        });

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, c, a, b;
    float alpha, beta;
};

void *plan(Tensor c, Tensor a, Tensor b, float alpha, float beta) {
    size_t seed = hash_combine(c, b, a, alpha, beta);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto descriptor = cache.get(seed).value_or(nullptr);

    if (!descriptor) {
        descriptor = std::make_shared<Descriptor>(nullptr);
        INFINICORE_CHECK_ERROR(infiniopCreateGemmDescriptor(
            context::getInfiniopHandle(device),
            &descriptor->desc,
            c->desc(), a->desc(), b->desc()));
        cache.put(seed, descriptor);
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetGemmWorkspaceSize(descriptor->desc, &workspace_size));
    Tensor workspace = Tensor::empty({workspace_size}, DataType::U8, device);

    auto planned = new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(c),
        graph::GraphTensor(a),
        graph::GraphTensor(b),
        alpha, beta};

    return planned;
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopGemm(
        planned->descriptor->desc, planned->workspace->data(), planned->workspace->numel(),
        planned->c->data(), planned->a->data(), planned->b->data(), planned->alpha, planned->beta, context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

void calculate(Tensor c, Tensor a, Tensor b, float alpha, float beta) {
    auto planned = plan(c, a, b, alpha, beta);
    run(planned);
    cleanup(&planned);
}

static bool registered = []() {
    Gemm::dispatcher().registerAll(&calculate, false);
    Gemm::plan_dispatcher().registerAll(&plan, false);
    Gemm::run_dispatcher().registerAll(&run, false);
    Gemm::cleanup_dispatcher().registerAll(&cleanup, false);
    return true;
}();

} // namespace infinicore::op::gemm_impl::infiniop

#if defined(ENABLE_ATEN) && defined(ENABLE_METAX_API)

#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/device.hpp"
#include "infinicore/ops/concat_and_cache_mla.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <stdexcept>
#include <string>

namespace infinicore::op::concat_and_cache_mla_impl::metax {

void run_concat_and_cache_mla_metax(const Tensor &kv_c,
                                    const Tensor &k_pe,
                                    Tensor kv_cache,
                                    const Tensor &slot_mapping,
                                    const std::string &kv_cache_dtype,
                                    Tensor scale) {
    if (kv_c->device().getType() != Device::Type::METAX) {
        throw std::runtime_error("concat_and_cache_mla expects METAX tensors in this build.");
    }
    if (kv_cache_dtype != "auto") {
        throw std::runtime_error("concat_and_cache_mla MetaX fallback currently supports kv_cache_dtype='auto' only.");
    }
    (void)scale;

    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());

    auto kv_c_at = infinicore::adaptor::to_aten_tensor(kv_c);
    auto k_pe_at = infinicore::adaptor::to_aten_tensor(k_pe);
    auto kv_cache_at = infinicore::adaptor::to_aten_tensor(kv_cache);
    auto slot_mapping_at = infinicore::adaptor::to_aten_tensor(slot_mapping).to(at::kLong);

    auto src = at::cat({kv_c_at, k_pe_at}, -1);
    auto flat_cache = kv_cache_at.view({-1, kv_cache_at.size(-1)});
    flat_cache.index_copy_(0, slot_mapping_at.reshape({-1}), src);
}

struct PlannedMeta {
    graph::GraphTensor kv_c;
    graph::GraphTensor k_pe;
    graph::GraphTensor kv_cache;
    graph::GraphTensor slot_mapping;
    std::string kv_cache_dtype;
    graph::GraphTensor scale;
};

void *plan(const Tensor &kv_c,
           const Tensor &k_pe,
           Tensor kv_cache,
           const Tensor &slot_mapping,
           const std::string &kv_cache_dtype,
           Tensor scale) {
    return new PlannedMeta{
        graph::GraphTensor(kv_c),
        graph::GraphTensor(k_pe),
        graph::GraphTensor(kv_cache),
        graph::GraphTensor(slot_mapping),
        kv_cache_dtype,
        graph::GraphTensor(scale)};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    run_concat_and_cache_mla_metax(planned->kv_c,
                                   planned->k_pe,
                                   planned->kv_cache,
                                   planned->slot_mapping,
                                   planned->kv_cache_dtype,
                                   planned->scale);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    vendor_ops::concat_and_cache_mla_dispatcher().registerDevice(Device::Type::METAX, &run_concat_and_cache_mla_metax);
    ConcatAndCacheMla::plan_dispatcher().registerDevice(Device::Type::METAX, &plan);
    ConcatAndCacheMla::run_dispatcher().registerDevice(Device::Type::METAX, &run);
    ConcatAndCacheMla::cleanup_dispatcher().registerDevice(Device::Type::METAX, &cleanup);
    return true;
}();

} // namespace infinicore::op::concat_and_cache_mla_impl::metax

#endif

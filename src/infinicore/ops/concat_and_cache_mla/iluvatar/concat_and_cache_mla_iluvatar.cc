#if defined(ENABLE_ILUVATAR_VENDOR_OPS)
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/iluvatar_vendor_adaptor.hpp"
#include "infinicore/ops/concat_and_cache_mla.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op::concat_and_cache_mla_impl::iluvatar {

void run_concat_and_cache_mla_iluvatar(const Tensor &kv_c,
                                       const Tensor &k_pe,
                                       Tensor kv_cache,
                                       const Tensor &slot_mapping,
                                       const std::string &kv_cache_dtype,
                                       Tensor scale) {
    if (!adaptor::iluvatar_vendor::concat_and_cache_mla_available()) {
        throw std::runtime_error("concat_and_cache_mla requires the Iluvatar vendor extension");
    }
    auto kv_c_at = adaptor::to_aten_tensor(kv_c);
    auto k_pe_at = adaptor::to_aten_tensor(k_pe);
    auto kv_cache_at = adaptor::to_aten_tensor(kv_cache);
    auto slot_mapping_at = adaptor::to_aten_tensor(slot_mapping);
    auto scale_at = adaptor::to_aten_tensor(scale);
    adaptor::iluvatar_vendor::concat_and_cache_mla(
        kv_c_at, k_pe_at,
        kv_cache_at, slot_mapping_at,
        kv_cache_dtype, scale_at);
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
    run_concat_and_cache_mla_iluvatar(planned->kv_c,
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
    vendor_ops::concat_and_cache_mla_dispatcher().registerDevice(Device::Type::ILUVATAR, &run_concat_and_cache_mla_iluvatar);
    ConcatAndCacheMla::plan_dispatcher().registerDevice(Device::Type::ILUVATAR, &plan);
    ConcatAndCacheMla::run_dispatcher().registerDevice(Device::Type::ILUVATAR, &run);
    ConcatAndCacheMla::cleanup_dispatcher().registerDevice(Device::Type::ILUVATAR, &cleanup);
    return true;
}();

} // namespace infinicore::op::concat_and_cache_mla_impl::iluvatar
#endif

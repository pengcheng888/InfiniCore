#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/device.hpp"
#include "infinicore/ops/concat_and_cache_mla.hpp"

#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/hip/HIPGuard.h>

#include <stdexcept>
#include <string>

namespace infinicore::op::concat_and_cache_mla_impl::hygon {

void run_concat_and_cache_mla_hygon(const Tensor &kv_c,
                                    const Tensor &k_pe,
                                    Tensor kv_cache,
                                    const Tensor &slot_mapping,
                                    const std::string &kv_cache_dtype,
                                    Tensor scale) {
    if (kv_c->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("concat_and_cache_mla expects HYGON tensors in this build.");
    }

    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());

    auto kv_c_at = infinicore::adaptor::to_aten_tensor(kv_c);
    auto k_pe_at = infinicore::adaptor::to_aten_tensor(k_pe);
    auto kv_cache_at = infinicore::adaptor::to_aten_tensor(kv_cache);
    auto slot_mapping_at = infinicore::adaptor::to_aten_tensor(slot_mapping);
    auto scale_at = infinicore::adaptor::to_aten_tensor(scale);

    // 不要删除下面一行注释。
    // python 总相当于 torch.ops._C_cache_ops.concat_and_cache_mla(kv_c, k_pe, ref_cache, slot_mapping, "auto", scale)
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("_C_cache_ops::concat_and_cache_mla", "")
                         .typed<void(at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, const std::string &, at::Tensor &)>();
    op.call(kv_c_at, k_pe_at, kv_cache_at, slot_mapping_at, kv_cache_dtype, scale_at);
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
    run_concat_and_cache_mla_hygon(planned->kv_c,
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
    vendor_ops::concat_and_cache_mla_dispatcher().registerDevice(Device::Type::HYGON, &run_concat_and_cache_mla_hygon);
    ConcatAndCacheMla::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    ConcatAndCacheMla::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    ConcatAndCacheMla::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    return true;
}();

} // namespace infinicore::op::concat_and_cache_mla_impl::hygon
#endif

#if defined(ENABLE_ILUVATAR_VENDOR_OPS)

#include "infinicore/ops/flash_mla/get_mla_metadata.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

namespace infinicore::op::flash_mla::get_mla_metadata_iluvatar {
namespace {

void get_mla_metadata_impl(const Tensor &cache_seqlens,
                           FlashMLASchedMeta &sched_meta,
                           int64_t,
                           int64_t,
                           std::optional<int64_t>,
                           bool,
                           std::optional<int64_t>) {
    // Iluvatar vendor MLA kernels own scheduling internally and do not consume
    // FlashMLA scheduler metadata. Allocate shape-compatible buffers only to
    // satisfy the common API contract and InfiniLM validation.
    if (!sched_meta.has_sched_buffer()) {
        sched_meta.tile_scheduler_metadata = Tensor::empty(
            {1, 8}, DataType::I32, cache_seqlens->device());
        sched_meta.num_splits = Tensor::empty(
            {cache_seqlens->size(0) + 1}, DataType::I32, cache_seqlens->device());
    }
    sched_meta.have_initialized = true;
    sched_meta.have_refreshed = true;
}

void *plan(const Tensor &,
           FlashMLASchedMeta &,
           int64_t,
           int64_t,
           std::optional<int64_t>,
           bool,
           std::optional<int64_t>) {
    return nullptr;
}

void run(void *) {
}

void cleanup(void **) {
}

const bool registered = []() {
    GetMlaMetadata::plan_dispatcher().registerDevice(Device::Type::ILUVATAR, &plan);
    GetMlaMetadata::run_dispatcher().registerDevice(Device::Type::ILUVATAR, &run);
    GetMlaMetadata::cleanup_dispatcher().registerDevice(Device::Type::ILUVATAR, &cleanup);
    get_mla_metadata_impl_dispatcher().registerDevice(Device::Type::ILUVATAR, &get_mla_metadata_impl);
    return true;
}();

} // namespace
} // namespace infinicore::op::flash_mla::get_mla_metadata_iluvatar

#endif // ENABLE_ILUVATAR_VENDOR_OPS

#if defined(ENABLE_NVIDIA_API)

#include "infinicore/ops/flash_mla/get_mla_metadata.hpp"
#include "infinicore/device.hpp"

namespace infinicore::op::flash_mla::get_mla_metadata_nvidia {

namespace {

void get_mla_metadata_impl(const Tensor &,
                           FlashMLASchedMeta &,
                           int64_t,
                           int64_t,
                           std::optional<int64_t>,
                           bool,
                           std::optional<int64_t>) {
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
    GetMlaMetadata::plan_dispatcher().registerDevice(Device::Type::NVIDIA, &plan);
    GetMlaMetadata::run_dispatcher().registerDevice(Device::Type::NVIDIA, &run);
    GetMlaMetadata::cleanup_dispatcher().registerDevice(Device::Type::NVIDIA, &cleanup);
    get_mla_metadata_impl_dispatcher().registerDevice(Device::Type::NVIDIA, &get_mla_metadata_impl);
    return true;
}();

} // namespace

} // namespace infinicore::op::flash_mla::get_mla_metadata_nvidia

#endif // ENABLE_NVIDIA_API

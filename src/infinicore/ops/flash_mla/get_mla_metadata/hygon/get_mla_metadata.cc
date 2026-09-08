#if defined(ENABLE_HYGON_API)

#include "infinicore/ops/flash_mla/get_mla_metadata.hpp"
#include "infinicore/device.hpp"

namespace infinicore::op::flash_mla::get_mla_metadata_hygon {

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
    GetMlaMetadata::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    GetMlaMetadata::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    GetMlaMetadata::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    get_mla_metadata_impl_dispatcher().registerDevice(Device::Type::HYGON, &get_mla_metadata_impl);
    return true;
}();

} // namespace

} // namespace infinicore::op::flash_mla::get_mla_metadata_hygon

#endif // ENABLE_HYGON_API

#if defined(ENABLE_HYGON_API)

#include "infinicore/device.hpp"
#include "infinicore/ops/flash_mla/get_mla_metadata.hpp"

namespace infinicore::op::flash_mla::get_mla_metadata_hygon {

namespace {

FlashMLASchedMeta get_mla_metadata_impl() {
    return FlashMLASchedMeta();
}

const bool registered = []() {
    get_mla_metadata_impl_dispatcher().registerDevice(
        Device::Type::HYGON, &get_mla_metadata_impl);
    return true;
}();

} // namespace

} // namespace infinicore::op::flash_mla::get_mla_metadata_hygon

#endif // ENABLE_HYGON_API

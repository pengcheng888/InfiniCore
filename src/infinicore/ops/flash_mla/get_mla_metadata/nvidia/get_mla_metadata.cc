#if defined(ENABLE_NVIDIA_API)

#include "infinicore/device.hpp"
#include "infinicore/ops/flash_mla/get_mla_metadata.hpp"

namespace infinicore::op::flash_mla::get_mla_metadata_nvidia {

namespace {

FlashMLASchedMeta get_mla_metadata_impl() {
    return FlashMLASchedMeta();
}

const bool registered = []() {
    get_mla_metadata_impl_dispatcher().registerDevice(
        Device::Type::NVIDIA, &get_mla_metadata_impl);
    return true;
}();

} // namespace

} // namespace infinicore::op::flash_mla::get_mla_metadata_nvidia

#endif // ENABLE_NVIDIA_API

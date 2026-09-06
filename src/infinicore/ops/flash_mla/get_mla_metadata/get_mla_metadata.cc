#include "infinicore/ops/flash_mla/get_mla_metadata.hpp"

#include "infinicore/context/context.hpp"

namespace infinicore::op::flash_mla {

common::OpDispatcher<GetMlaMetadataImplSchema> &get_mla_metadata_impl_dispatcher() {
    static common::OpDispatcher<GetMlaMetadataImplSchema> dispatcher;
    return dispatcher;
}

FlashMLASchedMeta get_mla_metadata() {
    return get_mla_metadata_impl_dispatcher().lookup(context::getDevice().getType())();
}

} // namespace infinicore::op::flash_mla

#pragma once

#include "flash_mla_sched_meta/flash_mla_sched_meta.hpp"

#include "../common/op.hpp"

namespace infinicore::op::flash_mla {

using GetMlaMetadataImplSchema = FlashMLASchedMeta (*)();

common::OpDispatcher<GetMlaMetadataImplSchema> &get_mla_metadata_impl_dispatcher();

FlashMLASchedMeta get_mla_metadata();

} // namespace infinicore::op::flash_mla

#pragma once

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
#include <ATen/ATen.h>

#include <string>

namespace infinicore::adaptor::hygon_vendor {

void concat_and_cache_mla(at::Tensor &kv_c,
                          at::Tensor &k_pe,
                          at::Tensor &kv_cache,
                          at::Tensor &slot_mapping,
                          const std::string &kv_cache_dtype,
                          at::Tensor &scale);

} // namespace infinicore::adaptor::hygon_vendor
#endif // ENABLE_ATEN && ENABLE_HYGON_API

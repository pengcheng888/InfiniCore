#pragma once

#ifdef ENABLE_ATEN
#include <ATen/ATen.h>
#endif

#include <optional>
#include <vector>

namespace infinicore::op::flash_mla::dense_decode_fwd_metax {

#if defined(ENABLE_ATEN) && defined(ENABLE_METAX_API)

using FlashMlaDenseDecodeFn = std::vector<at::Tensor> (*)(
    at::Tensor &,
    const at::Tensor &,
    std::optional<const at::Tensor> &,
    int,
    const at::Tensor &,
    const at::Tensor &,
    float,
    bool,
    const at::Tensor &,
    const at::Tensor &);

using FlashMlaMetadataFn = std::vector<at::Tensor> (*)(at::Tensor &, int, int);

FlashMlaDenseDecodeFn flashmla_dense_decode_fn(const char *op_name);
FlashMlaMetadataFn flashmla_metadata_fn(const char *op_name);

#endif

} // namespace infinicore::op::flash_mla::dense_decode_fwd_metax

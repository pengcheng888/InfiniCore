#pragma once

#ifdef ENABLE_ATEN
#include <ATen/ATen.h>
#endif

#include <optional>
#include <tuple>

namespace infinicore::op::flash_mla::sparse_decode_fwd_hygon {

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)

using FlashMlaSparseDecodeFn = std::tuple<at::Tensor,
                                          at::Tensor,
                                          std::optional<at::Tensor>,
                                          std::optional<at::Tensor>> (*)(const at::Tensor &,
                                                                         const at::Tensor &,
                                                                         const at::Tensor &,
                                                                         const std::optional<at::Tensor> &,
                                                                         const std::optional<at::Tensor> &,
                                                                         std::optional<at::Tensor> &,
                                                                         std::optional<at::Tensor> &,
                                                                         const std::optional<at::Tensor> &,
                                                                         const std::optional<at::Tensor> &,
                                                                         const std::optional<at::Tensor> &,
                                                                         int,
                                                                         float);

FlashMlaSparseDecodeFn flashmla_sparse_decode_fn(const char *op_name);

#endif

} // namespace infinicore::op::flash_mla::sparse_decode_fwd_hygon

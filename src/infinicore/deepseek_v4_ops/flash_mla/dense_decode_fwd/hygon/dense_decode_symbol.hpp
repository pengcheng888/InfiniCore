#pragma once

#ifdef ENABLE_ATEN
#include <ATen/ATen.h>
#endif

#include <optional>
#include <tuple>

namespace infinicore::op::flash_mla::dense_decode_fwd_hygon {

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)

using FlashMlaDenseDecodeFn = std::tuple<at::Tensor,
                                         at::Tensor,
                                         std::optional<at::Tensor>,
                                         std::optional<at::Tensor>> (*)(at::Tensor &,
                                                                        const at::Tensor &,
                                                                        int,
                                                                        const at::Tensor &,
                                                                        const at::Tensor &,
                                                                        float,
                                                                        bool,
                                                                        std::optional<at::Tensor> &,
                                                                        std::optional<at::Tensor> &);

FlashMlaDenseDecodeFn flashmla_dense_decode_fn(const char *op_name);

#endif

} // namespace infinicore::op::flash_mla::dense_decode_fwd_hygon

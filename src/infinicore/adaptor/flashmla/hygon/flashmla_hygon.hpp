#ifndef INFINICORE_ADAPTOR_FLASHMLA_HYGON_FLASHMLA_HYGON_HPP
#define INFINICORE_ADAPTOR_FLASHMLA_HYGON_FLASHMLA_HYGON_HPP

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)

#include <ATen/ATen.h>

#include <optional>
#include <tuple>

namespace infinicore::adaptor::flashmla::hygon {

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

FlashMlaDenseDecodeFn flashmla_dense_decode_fn(const char *op_name);
FlashMlaSparseDecodeFn flashmla_sparse_decode_fn(const char *op_name);
bool flashmla_dense_decode_available();
bool flashmla_sparse_decode_available();

} // namespace infinicore::adaptor::flashmla::hygon

#endif

#endif

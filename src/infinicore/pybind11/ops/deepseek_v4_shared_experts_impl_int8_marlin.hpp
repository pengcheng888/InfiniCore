#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_shared_experts_impl_int8_marlin.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void py_deepseek_v4_shared_experts_impl_int8_marlin_(Tensor output,
                                                            const Tensor &hidden_states,
                                                            const Tensor &w1,
                                                            const Tensor &w2,
                                                            const Tensor &w1_scale,
                                                            const Tensor &w2_scale,
                                                            int gemm1_mode,
                                                            int gemm2_mode,
                                                            int delta) {
    op::deepseek_v4_shared_experts_impl_int8_marlin_(output,
                                                     hidden_states,
                                                     w1,
                                                     w2,
                                                     w1_scale,
                                                     w2_scale,
                                                     gemm1_mode,
                                                     gemm2_mode,
                                                     delta);
}

inline void py_deepseek_v4_shared_experts_impl_int8_marlin_with_workspace_(Tensor output,
                                                                           const Tensor &hidden_states,
                                                                           const Tensor &w1,
                                                                           const Tensor &w2,
                                                                           const Tensor &w1_scale,
                                                                           const Tensor &w2_scale,
                                                                           Tensor sorted_token_ids,
                                                                           Tensor expert_ids,
                                                                           Tensor num_tokens_post_pad,
                                                                           Tensor topk_weights,
                                                                           Tensor q_hidden,
                                                                           Tensor hidden_scale,
                                                                           Tensor gate_up,
                                                                           Tensor q_activated,
                                                                           Tensor activated_scale,
                                                                           int gemm1_mode,
                                                                           int gemm2_mode,
                                                                           int delta) {
    op::deepseek_v4_shared_experts_impl_int8_marlin_(output,
                                                     hidden_states,
                                                     w1,
                                                     w2,
                                                     w1_scale,
                                                     w2_scale,
                                                     sorted_token_ids,
                                                     expert_ids,
                                                     num_tokens_post_pad,
                                                     topk_weights,
                                                     q_hidden,
                                                     hidden_scale,
                                                     gate_up,
                                                     q_activated,
                                                     activated_scale,
                                                     gemm1_mode,
                                                     gemm2_mode,
                                                     delta);
}

inline void bind_deepseek_v4_shared_experts_impl_int8_marlin(py::module &m) {
    m.def("deepseek_v4_shared_experts_impl_int8_marlin_",
          &py_deepseek_v4_shared_experts_impl_int8_marlin_,
          py::arg("output"),
          py::arg("hidden_states"),
          py::arg("w1"),
          py::arg("w2"),
          py::arg("w1_scale"),
          py::arg("w2_scale"),
          py::arg("gemm1_mode") = -1,
          py::arg("gemm2_mode") = -1,
          py::arg("delta") = 1,
          R"doc(DeepSeek-V4 shared MLP implemented through the INT8 Marlin expert path with a single shared expert.)doc");
    m.def("deepseek_v4_shared_experts_impl_int8_marlin_prepare_metadata_",
          &op::deepseek_v4_shared_experts_impl_int8_marlin_prepare_metadata_,
          py::arg("sorted_token_ids"),
          py::arg("expert_ids"),
          py::arg("num_tokens_post_pad"),
          py::arg("topk_weights"),
          py::arg("tokens"),
          R"doc(Prepare the fixed topk=1/expert=0 metadata used by DeepSeek-V4 shared experts INT8 Marlin.)doc");
    m.def("deepseek_v4_shared_experts_impl_int8_marlin_",
          &py_deepseek_v4_shared_experts_impl_int8_marlin_with_workspace_,
          py::arg("output"),
          py::arg("hidden_states"),
          py::arg("w1"),
          py::arg("w2"),
          py::arg("w1_scale"),
          py::arg("w2_scale"),
          py::arg("sorted_token_ids"),
          py::arg("expert_ids"),
          py::arg("num_tokens_post_pad"),
          py::arg("topk_weights"),
          py::arg("q_hidden"),
          py::arg("hidden_scale"),
          py::arg("gate_up"),
          py::arg("q_activated"),
          py::arg("activated_scale"),
          py::arg("gemm1_mode") = -1,
          py::arg("gemm2_mode") = -1,
          py::arg("delta") = 1,
          R"doc(Workspace variant of DeepSeek-V4 shared experts INT8 Marlin.)doc");
}

} // namespace infinicore::ops

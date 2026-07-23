#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_fused_experts_impl_int8_marlin.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline std::optional<Tensor> py_optional_tensor_for_fused_experts(py::object obj) {
    if (obj.is_none()) {
        return std::nullopt;
    }
    return obj.cast<Tensor>();
}

inline void py_deepseek_v4_fused_experts_impl_int8_marlin_(Tensor output,
                                                           Tensor hidden_states,
                                                           Tensor w1,
                                                           Tensor w2,
                                                           Tensor topk_weights,
                                                           Tensor topk_ids,
                                                           Tensor w1_scale,
                                                           Tensor w2_scale,
                                                           int64_t global_num_experts,
                                                           double routed_scaling_factor,
                                                           bool inplace,
                                                           py::object shared_output) {
    op::deepseek_v4_fused_experts_impl_int8_marlin_(
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        global_num_experts,
        routed_scaling_factor,
        inplace,
        py_optional_tensor_for_fused_experts(shared_output));
}

inline void bind_deepseek_v4_fused_experts_impl_int8_marlin(py::module &m) {
    m.def("deepseek_v4_fused_experts_impl_int8_marlin_",
          &ops::py_deepseek_v4_fused_experts_impl_int8_marlin_,
          py::arg("output"),
          py::arg("hidden_states"),
          py::arg("w1"),
          py::arg("w2"),
          py::arg("topk_weights"),
          py::arg("topk_ids"),
          py::arg("w1_scale"),
          py::arg("w2_scale"),
          py::arg("global_num_experts"),
          py::arg("routed_scaling_factor") = 1.0,
          py::arg("inplace") = false,
          py::arg("shared_output") = py::none(),
          R"doc(DeepSeek-V4 SGLang fused_experts_impl_int8_marlin bridge. shared_output is added inside moe_sum when provided.)doc");
    m.def("deepseek_v4_python_fused_experts_impl_int8_marlin_",
          &op::deepseek_v4_python_fused_experts_impl_int8_marlin_,
          py::arg("output"),
          py::arg("hidden_states"),
          py::arg("w1"),
          py::arg("w2"),
          py::arg("topk_weights"),
          py::arg("topk_ids"),
          py::arg("w1_scale"),
          py::arg("w2_scale"),
          py::arg("global_num_experts"),
          py::arg("routed_scaling_factor") = 1.0,
          py::arg("inplace") = false,
          R"doc(DeepSeek-V4 compatibility bridge through SGLang torch.library. This path can enter Python/GIL and is not intended for InfiniLM hot paths.)doc");
}

} // namespace infinicore::ops

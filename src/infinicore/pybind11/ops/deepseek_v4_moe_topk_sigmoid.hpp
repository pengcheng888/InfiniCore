#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_moe_topk_sigmoid.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void py_deepseek_v4_moe_topk_sigmoid_(Tensor topk_weights,
                                             Tensor topk_indices,
                                             Tensor gating_output,
                                             bool renormalize,
                                             py::object correction_bias) {
    std::optional<Tensor> correction_bias_tensor = std::nullopt;
    if (!correction_bias.is_none()) {
        correction_bias_tensor = correction_bias.cast<Tensor>();
    }
    op::deepseek_v4_moe_topk_sigmoid_(topk_weights, topk_indices, gating_output, renormalize, correction_bias_tensor);
}

inline void bind_deepseek_v4_moe_topk_sigmoid(py::module &m) {
    m.def("deepseek_v4_moe_topk_sigmoid_",
          &ops::py_deepseek_v4_moe_topk_sigmoid_,
          py::arg("topk_weights"),
          py::arg("topk_indices"),
          py::arg("gating_output"),
          py::arg("renormalize") = false,
          py::arg("correction_bias") = py::none(),
          R"doc(DeepSeek-V4 MoE topk sigmoid backed by SGLang sgl_kernel.)doc");
}

} // namespace infinicore::ops

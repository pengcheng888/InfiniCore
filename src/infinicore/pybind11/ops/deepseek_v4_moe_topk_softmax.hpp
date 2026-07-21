#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_moe_topk_softmax.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void py_deepseek_v4_moe_topk_softmax_(Tensor topk_weights,
                                             Tensor topk_indices,
                                             Tensor gating_output,
                                             bool renormalize,
                                             float moe_softcapping,
                                             py::object correction_bias) {
    std::optional<Tensor> correction_bias_tensor = std::nullopt;
    if (!correction_bias.is_none()) {
        correction_bias_tensor = correction_bias.cast<Tensor>();
    }
    op::deepseek_v4_moe_topk_softmax_(topk_weights, topk_indices, gating_output, renormalize, moe_softcapping, correction_bias_tensor);
}

inline void bind_deepseek_v4_moe_topk_softmax(py::module &m) {
    m.def("deepseek_v4_moe_topk_softmax_",
          &ops::py_deepseek_v4_moe_topk_softmax_,
          py::arg("topk_weights"),
          py::arg("topk_indices"),
          py::arg("gating_output"),
          py::arg("renormalize") = true,
          py::arg("moe_softcapping") = 0.0,
          py::arg("correction_bias") = py::none(),
          R"doc(DeepSeek-V4 MoE topk softmax backed by SGLang sgl_kernel.)doc");
}

} // namespace infinicore::ops

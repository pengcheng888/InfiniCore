#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_lightop_linear_w8a8_smooth.hpp"

namespace py = pybind11;

namespace infinicore::ops {

void py_deepseek_v4_lightop_linear_w8a8_smooth_(Tensor output,
                                                const Tensor &input,
                                                const Tensor &weight,
                                                const Tensor &weight_scale,
                                                py::object bias,
                                                Tensor q_input,
                                                Tensor input_scale,
                                                const Tensor &smooth_scale,
                                                py::object is_tuned_slide_block,
                                                py::object tuned_slide_block) {
    std::optional<Tensor> bias_tensor = std::nullopt;
    if (!bias.is_none()) {
        bias_tensor = bias.cast<Tensor>();
    }
    if (is_tuned_slide_block.is_none() || tuned_slide_block.is_none()) {
        op::deepseek_v4_lightop_linear_w8a8_smooth_(
            output, input, weight, weight_scale, bias_tensor, q_input, input_scale, smooth_scale);
    } else {
        op::deepseek_v4_lightop_linear_w8a8_smooth_(
            output,
            input,
            weight,
            weight_scale,
            bias_tensor,
            q_input,
            input_scale,
            smooth_scale,
            is_tuned_slide_block.cast<bool>(),
            tuned_slide_block.cast<int>());
    }
}

inline void bind_deepseek_v4_lightop_linear_w8a8_smooth(py::module &m) {
    m.def("deepseek_v4_lightop_linear_w8a8_smooth_",
          &ops::py_deepseek_v4_lightop_linear_w8a8_smooth_,
          py::arg("output"),
          py::arg("input"),
          py::arg("weight"),
          py::arg("weight_scale"),
          py::arg("bias"),
          py::arg("q_input"),
          py::arg("input_scale"),
          py::arg("smooth_scale"),
          py::arg("is_tuned_slide_block") = py::none(),
          py::arg("tuned_slide_block") = py::none(),
          R"doc(deepseek_v4_lightop_linear_w8a8_smooth_.)doc");
}

} // namespace infinicore::ops

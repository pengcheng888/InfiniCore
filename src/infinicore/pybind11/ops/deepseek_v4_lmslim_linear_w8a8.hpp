#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_lmslim_linear_w8a8.hpp"

namespace py = pybind11;

namespace infinicore::ops {

void py_deepseek_v4_lmslim_linear_w8a8_(Tensor output,
                                        const Tensor &input,
                                        const Tensor &weight_t,
                                        const Tensor &weight_scale,
                                        py::object bias,
                                        Tensor q_input,
                                        Tensor input_scale,
                                        const Tensor &smooth_scale) {
    std::optional<Tensor> bias_tensor = std::nullopt;
    if (!bias.is_none()) {
        bias_tensor = bias.cast<Tensor>();
    }
    op::deepseek_v4_lmslim_linear_w8a8_(output, input, weight_t, weight_scale, bias_tensor, q_input, input_scale, smooth_scale);
}

inline void bind_deepseek_v4_lmslim_linear_w8a8(py::module &m) {
    m.def("deepseek_v4_lmslim_linear_w8a8_",
          &ops::py_deepseek_v4_lmslim_linear_w8a8_,
          py::arg("output"),
          py::arg("input"),
          py::arg("weight_t"),
          py::arg("weight_scale"),
          py::arg("bias"),
          py::arg("q_input"),
          py::arg("input_scale"),
          py::arg("smooth_scale"),
          R"doc(deepseek_v4_lmslim_linear_w8a8_.)doc");
}

} // namespace infinicore::ops

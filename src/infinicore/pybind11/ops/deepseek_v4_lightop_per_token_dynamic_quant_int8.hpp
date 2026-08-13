#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_lightop_per_token_dynamic_quant_int8.hpp"

namespace py = pybind11;

namespace infinicore::ops {

void py_deepseek_v4_lightop_per_token_dynamic_quant_int8_(Tensor q_input,
                                                          const Tensor &input,
                                                          Tensor input_scale,
                                                          const Tensor &smooth_scale) {
    op::deepseek_v4_lightop_per_token_dynamic_quant_int8_(q_input, input, input_scale, smooth_scale);
}

inline void bind_deepseek_v4_lightop_per_token_dynamic_quant_int8(py::module &m) {
    m.def("deepseek_v4_lightop_per_token_dynamic_quant_int8_",
          &ops::py_deepseek_v4_lightop_per_token_dynamic_quant_int8_,
          py::arg("q_input"),
          py::arg("input"),
          py::arg("input_scale"),
          py::arg("smooth_scale"),
          R"doc(deepseek_v4_lightop_per_token_dynamic_quant_int8_.)doc");
}

} // namespace infinicore::ops

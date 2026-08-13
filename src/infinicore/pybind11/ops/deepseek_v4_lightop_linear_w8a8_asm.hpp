#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_lightop_linear_w8a8_asm.hpp"

namespace py = pybind11;

namespace infinicore::ops {

void py_deepseek_v4_lightop_linear_w8a8_asm_(Tensor output,
                                             const Tensor &q_input,
                                             const Tensor &weight,
                                             const Tensor &input_block_scale,
                                             const Tensor &weight_block_scale) {
    op::deepseek_v4_lightop_linear_w8a8_asm_(output, q_input, weight, input_block_scale, weight_block_scale);
}

void py_deepseek_v4_lightop_linear_w8a8_asm_per_channel_(Tensor output,
                                                         const Tensor &input,
                                                         const Tensor &weight,
                                                         const Tensor &weight_scale,
                                                         Tensor q_input,
                                                         Tensor input_scale,
                                                         Tensor input_block_scale,
                                                         Tensor weight_block_scale,
                                                         const Tensor &smooth_scale) {
    op::deepseek_v4_lightop_linear_w8a8_asm_per_channel_(output,
                                                         input,
                                                         weight,
                                                         weight_scale,
                                                         q_input,
                                                         input_scale,
                                                         input_block_scale,
                                                         weight_block_scale,
                                                         smooth_scale);
}

inline void bind_deepseek_v4_lightop_linear_w8a8_asm(py::module &m) {
    m.def("deepseek_v4_lightop_linear_w8a8_asm_",
          &ops::py_deepseek_v4_lightop_linear_w8a8_asm_,
          py::arg("output"),
          py::arg("q_input"),
          py::arg("weight"),
          py::arg("input_block_scale"),
          py::arg("weight_block_scale"),
          R"doc(deepseek_v4_lightop_linear_w8a8_asm_.)doc");
    m.def("deepseek_v4_lightop_linear_w8a8_asm_per_channel_",
          &ops::py_deepseek_v4_lightop_linear_w8a8_asm_per_channel_,
          py::arg("output"),
          py::arg("input"),
          py::arg("weight"),
          py::arg("weight_scale"),
          py::arg("q_input"),
          py::arg("input_scale"),
          py::arg("input_block_scale"),
          py::arg("weight_block_scale"),
          py::arg("smooth_scale"),
          R"doc(deepseek_v4_lightop_linear_w8a8_asm_per_channel_.)doc");
}

} // namespace infinicore::ops

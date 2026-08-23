#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_c4_act_quant_fused_scale.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_c4_act_quant_fused_scale(py::module &m) {
    m.def("deepseek_v4_c4_act_quant_fused_scale_kernel_",
          &op::deepseek_v4_c4_act_quant_fused_scale_kernel_,
          py::arg("q"),
          py::arg("indexer_weights"),
          py::arg("q_fp8"),
          py::arg("q_scale"),
          py::arg("fused_weights"),
          py::arg("weight_scale") = 1.0f);
}

} // namespace infinicore::ops

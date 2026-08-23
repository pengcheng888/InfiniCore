#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_rms_norm_per_block_quant.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void py_deepseek_v4_rms_norm_per_block_quant_(Tensor result,
                                                     Tensor input,
                                                     Tensor weight,
                                                     Tensor scale,
                                                     float epsilon,
                                                     py::object scale_ub,
                                                     py::object residual,
                                                     int group_size,
                                                     bool is_scale_transposed) {
    std::optional<Tensor> scale_ub_tensor = std::nullopt;
    std::optional<Tensor> residual_tensor = std::nullopt;
    if (!scale_ub.is_none()) {
        scale_ub_tensor = scale_ub.cast<Tensor>();
    }
    if (!residual.is_none()) {
        residual_tensor = residual.cast<Tensor>();
    }
    op::deepseek_v4_rms_norm_per_block_quant_(
        result, input, weight, scale, epsilon, scale_ub_tensor, residual_tensor, group_size, is_scale_transposed);
}

inline void bind_deepseek_v4_rms_norm_per_block_quant(py::module &m) {
    m.def("deepseek_v4_rms_norm_per_block_quant_",
          &ops::py_deepseek_v4_rms_norm_per_block_quant_,
          py::arg("result"),
          py::arg("input"),
          py::arg("weight"),
          py::arg("scale"),
          py::arg("epsilon"),
          py::arg("scale_ub") = py::none(),
          py::arg("residual") = py::none(),
          py::arg("group_size"),
          py::arg("is_scale_transposed") = false,
          R"doc(DeepSeek-V4 VLLM RMSNorm per-block quant bridge.)doc");
}

} // namespace infinicore::ops

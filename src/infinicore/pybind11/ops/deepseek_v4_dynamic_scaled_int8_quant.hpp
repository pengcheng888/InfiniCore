#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_dynamic_scaled_int8_quant.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void py_deepseek_v4_dynamic_scaled_int8_quant_(Tensor result,
                                                      Tensor input,
                                                      Tensor scale,
                                                      py::object azp) {
    std::optional<Tensor> azp_tensor = std::nullopt;
    if (!azp.is_none()) {
        azp_tensor = azp.cast<Tensor>();
    }
    op::deepseek_v4_dynamic_scaled_int8_quant_(result, input, scale, azp_tensor);
}

inline void bind_deepseek_v4_dynamic_scaled_int8_quant(py::module &m) {
    m.def("deepseek_v4_dynamic_scaled_int8_quant_",
          &ops::py_deepseek_v4_dynamic_scaled_int8_quant_,
          py::arg("result"),
          py::arg("input"),
          py::arg("scale"),
          py::arg("azp") = py::none(),
          R"doc(DeepSeek-V4 VLLM dynamic scaled int8 quant bridge.)doc");
}

} // namespace infinicore::ops

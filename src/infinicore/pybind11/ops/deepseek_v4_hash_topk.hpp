#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/deepseek_v4_hash_topk.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_deepseek_v4_hash_topk(py::module &m) {
    m.def("deepseek_v4_hash_topk_",
          &op::deepseek_v4_hash_topk_,
          py::arg("topk_weights"),
          py::arg("topk_indices"),
	          py::arg("router_logits"),
	          py::arg("input_ids"),
	          py::arg("tid2eid"),
	          py::arg("num_fused_shared_experts") = 0,
	          py::arg("routed_scaling_factor") = 1.0f,
	          py::arg("scoring_func") = "sqrtsoftplus");
	    m.def("deepseek_v4_hash_topk_aten_",
	          &op::deepseek_v4_hash_topk_aten_,
	          py::arg("topk_weights"),
          py::arg("topk_indices"),
	          py::arg("router_logits"),
	          py::arg("input_ids"),
	          py::arg("tid2eid"),
	          py::arg("num_fused_shared_experts") = 0,
	          py::arg("routed_scaling_factor") = 1.0f,
	          py::arg("scoring_func") = "sqrtsoftplus");
	    m.def("deepseek_v4_hash_topk_generic_kernel_",
	          &op::deepseek_v4_hash_topk_generic_kernel_,
	          py::arg("topk_weights"),
          py::arg("topk_indices"),
	          py::arg("router_logits"),
	          py::arg("input_ids"),
	          py::arg("tid2eid"),
	          py::arg("num_fused_shared_experts") = 0,
	          py::arg("routed_scaling_factor") = 1.0f,
	          py::arg("scoring_func") = "sqrtsoftplus");
	    m.def("deepseek_v4_hash_topk_sglang_kernel_",
	          &op::deepseek_v4_hash_topk_sglang_kernel_,
	          py::arg("topk_weights"),
          py::arg("topk_indices"),
	          py::arg("router_logits"),
	          py::arg("input_ids"),
	          py::arg("tid2eid"),
	          py::arg("num_fused_shared_experts") = 0,
	          py::arg("routed_scaling_factor") = 1.0f,
	          py::arg("scoring_func") = "sqrtsoftplus");
	    m.def("deepseek_v4_hash_topk_kernel_",
	          &op::deepseek_v4_hash_topk_kernel_,
	          py::arg("topk_weights"),
          py::arg("topk_indices"),
	          py::arg("router_logits"),
	          py::arg("input_ids"),
	          py::arg("tid2eid"),
	          py::arg("num_fused_shared_experts") = 0,
	          py::arg("routed_scaling_factor") = 1.0f,
	          py::arg("scoring_func") = "sqrtsoftplus");
	}

} // namespace infinicore::ops

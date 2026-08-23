#pragma once

#include <pybind11/pybind11.h>

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <torch/csrc/utils/pybind.h>
#endif

#include "infinicore/ops/flash_mla/sparse_decode_fwd.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline std::optional<Tensor> py_optional_tensor_for_sparse_decode_fwd(py::object obj) {
    if (obj.is_none()) {
        return std::nullopt;
    }
    return obj.cast<Tensor>();
}

#ifdef ENABLE_ATEN
namespace {

inline py::object to_py_torch_tensor_for_sparse_decode_fwd(const Tensor &tensor) {
    if (!tensor) {
        return py::none();
    }
    return py::cast(infinicore::adaptor::to_aten_tensor(tensor));
}

inline py::tuple sparse_decode_result_to_py_tuple(const std::tuple<Tensor, Tensor, Tensor, Tensor> &result) {
    return py::make_tuple(to_py_torch_tensor_for_sparse_decode_fwd(std::get<0>(result)),
                          to_py_torch_tensor_for_sparse_decode_fwd(std::get<1>(result)),
                          to_py_torch_tensor_for_sparse_decode_fwd(std::get<2>(result)),
                          to_py_torch_tensor_for_sparse_decode_fwd(std::get<3>(result)));
}

} // namespace
#endif

inline py::object py_flash_mla_sparse_decode_fwd(Tensor q,
                                                   Tensor k_cache,
                                                   Tensor indices,
                                                   py::object topk_length,
                                                   py::object attn_sink,
                                                   py::object tile_scheduler_metadata,
                                                   py::object num_splits,
                                                   py::object extra_k_cache,
                                                   py::object extra_indices_in_kvcache,
                                                   py::object extra_topk_length,
                                                   int64_t head_dim_v,
                                                   double softmax_scale) {
#ifdef ENABLE_ATEN
    auto result = op::flash_mla::sparse_decode_fwd(q,
                                                   k_cache,
                                                   indices,
                                                   py_optional_tensor_for_sparse_decode_fwd(topk_length),
                                                   py_optional_tensor_for_sparse_decode_fwd(attn_sink),
                                                   py_optional_tensor_for_sparse_decode_fwd(tile_scheduler_metadata),
                                                   py_optional_tensor_for_sparse_decode_fwd(num_splits),
                                                   py_optional_tensor_for_sparse_decode_fwd(extra_k_cache),
                                                   py_optional_tensor_for_sparse_decode_fwd(extra_indices_in_kvcache),
                                                   py_optional_tensor_for_sparse_decode_fwd(extra_topk_length),
                                                   head_dim_v,
                                                   softmax_scale);
    return sparse_decode_result_to_py_tuple(result);
#endif
    (void)q;
    (void)k_cache;
    (void)indices;
    (void)topk_length;
    (void)attn_sink;
    (void)tile_scheduler_metadata;
    (void)num_splits;
    (void)extra_k_cache;
    (void)extra_indices_in_kvcache;
    (void)extra_topk_length;
    (void)head_dim_v;
    (void)softmax_scale;
    throw std::runtime_error("sparse_decode_fwd requires an ATen-enabled build.");
}

inline void bind_flash_mla_sparse_decode_fwd(py::module &m) {
    m.def("flash_mla_sparse_decode_fwd",
          &ops::py_flash_mla_sparse_decode_fwd,
          py::arg("q"),
          py::arg("k_cache"),
          py::arg("indices"),
          py::arg("topk_length"),
          py::arg("attn_sink"),
          py::arg("tile_scheduler_metadata") = py::none(),
          py::arg("num_splits") = py::none(),
          py::arg("extra_k_cache") = py::none(),
          py::arg("extra_indices_in_kvcache") = py::none(),
          py::arg("extra_topk_length") = py::none(),
          py::arg("head_dim_v") = 512,
          py::arg("softmax_scale") = 1.0,
          R"doc(FlashMLA bridge for flash_mla.cuda sparse_decode_fwd. Returns out, lse, new_tile_scheduler_metadata, new_num_splits.)doc");
}

} // namespace infinicore::ops

#pragma once

#include <stdexcept>

#include <pybind11/pybind11.h>

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <torch/csrc/utils/pybind.h>
#endif

#include "infinicore/dtype.hpp"
#include "infinicore/ops/flash_mla/dense_decode_fwd.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline std::optional<Tensor> py_optional_tensor_for_dense_decode_fwd(py::object obj) {
    if (obj.is_none()) {
        return std::nullopt;
    }
    return obj.cast<Tensor>();
}

#ifdef ENABLE_ATEN
namespace {

inline py::object to_py_torch_tensor_for_dense_decode_fwd(const Tensor &tensor) {
    if (!tensor) {
        return py::none();
    }
    return py::cast(infinicore::adaptor::to_aten_tensor(tensor));
}

inline py::object to_py_torch_tensor_for_dense_decode_fwd(const std::optional<Tensor> &tensor) {
    if (!tensor.has_value()) {
        return py::none();
    }
    return to_py_torch_tensor_for_dense_decode_fwd(tensor.value());
}

} // namespace
#endif

inline py::object py_flash_mla_dense_decode_fwd(Tensor q,
                                                  Tensor k_cache,
                                                  int64_t head_dim_v,
                                                  Tensor cache_seqlens,
                                                  Tensor block_table,
                                                  double softmax_scale,
                                                  bool causal,
                                                  py::object tile_scheduler_metadata,
                                                  py::object num_splits) {
#ifdef ENABLE_ATEN
    if (q->ndim() != 4) {
        throw std::runtime_error("flash_mla_dense_decode_fwd expects q shape [batch, seq_q, heads, head_dim].");
    }
    if (head_dim_v <= 0) {
        throw std::runtime_error("flash_mla_dense_decode_fwd expects positive head_dim_v.");
    }
    Tensor out = Tensor::empty({q->size(0), q->size(1), q->size(2), static_cast<size_t>(head_dim_v)},
                               q->dtype(),
                               q->device());
    Tensor lse = Tensor::empty({q->size(0), q->size(2), q->size(1)},
                               DataType::F32,
                               q->device());
    std::optional<Tensor> new_tile_scheduler_metadata;
    std::optional<Tensor> new_num_splits;
    op::flash_mla::dense_decode_fwd_(out,
                                     lse,
                                     new_tile_scheduler_metadata,
                                     new_num_splits,
                                     q,
                                     k_cache,
                                     head_dim_v,
                                     cache_seqlens,
                                     block_table,
                                     softmax_scale,
                                     causal,
                                     py_optional_tensor_for_dense_decode_fwd(tile_scheduler_metadata),
                                     py_optional_tensor_for_dense_decode_fwd(num_splits));
    return py::make_tuple(to_py_torch_tensor_for_dense_decode_fwd(out),
                          to_py_torch_tensor_for_dense_decode_fwd(lse),
                          to_py_torch_tensor_for_dense_decode_fwd(new_tile_scheduler_metadata),
                          to_py_torch_tensor_for_dense_decode_fwd(new_num_splits));
#endif
    (void)q;
    (void)k_cache;
    (void)head_dim_v;
    (void)cache_seqlens;
    (void)block_table;
    (void)softmax_scale;
    (void)causal;
    (void)tile_scheduler_metadata;
    (void)num_splits;
    throw std::runtime_error("dense_decode_fwd requires an ATen-enabled build.");
}

inline void bind_flash_mla_dense_decode_fwd(py::module &m) {
    m.def("flash_mla_dense_decode_fwd",
          &ops::py_flash_mla_dense_decode_fwd,
          py::arg("q"),
          py::arg("k_cache"),
          py::arg("head_dim_v"),
          py::arg("cache_seqlens"),
          py::arg("block_table"),
          py::arg("softmax_scale"),
          py::arg("causal"),
          py::arg("tile_scheduler_metadata") = py::none(),
          py::arg("num_splits") = py::none(),
          R"doc(FlashMLA bridge for flash_mla.cuda dense_decode_fwd. Returns out, lse, new_tile_scheduler_metadata, new_num_splits.)doc");
}

} // namespace infinicore::ops

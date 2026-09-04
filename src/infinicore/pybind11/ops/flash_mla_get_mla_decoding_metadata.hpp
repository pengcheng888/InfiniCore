#pragma once

#include <stdexcept>

#include <pybind11/pybind11.h>

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <torch/csrc/utils/pybind.h>
#endif

#include "infinicore/ops/flash_mla/get_mla_decoding_metadata.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline std::optional<Tensor> py_optional_tensor_for_flash_mla_metadata(py::object obj) {
    if (obj.is_none()) {
        return std::nullopt;
    }
    return obj.cast<Tensor>();
}

inline std::optional<int64_t> py_optional_i64_for_flash_mla_metadata(py::object obj) {
    if (obj.is_none()) {
        return std::nullopt;
    }
    return obj.cast<int64_t>();
}

#ifdef ENABLE_ATEN
namespace {

inline py::object to_py_torch_tensor_for_flash_mla_metadata(const Tensor &tensor) {
    if (!tensor) {
        return py::none();
    }
    return py::cast(infinicore::adaptor::to_aten_tensor(tensor));
}

} // namespace
#endif

inline py::object py_flash_mla_get_mla_decoding_metadata(Tensor cache_seqlens,
                                                         int64_t num_q_tokens_per_head_k,
                                                         int64_t num_heads_k,
                                                         py::object num_heads_q,
                                                         bool is_fp8_kvcache,
                                                         py::object topk,
                                                         py::object tile_scheduler_metadata,
                                                         py::object num_splits) {
#ifdef ENABLE_ATEN
    auto tile_scheduler_metadata_opt = py_optional_tensor_for_flash_mla_metadata(tile_scheduler_metadata);
    auto num_splits_opt = py_optional_tensor_for_flash_mla_metadata(num_splits);
    Tensor tile_scheduler_metadata_out = tile_scheduler_metadata_opt.has_value()
                                           ? tile_scheduler_metadata_opt.value()
                                           : Tensor{};
    Tensor num_splits_out = num_splits_opt.has_value()
                              ? num_splits_opt.value()
                              : Tensor{};

    auto [new_tile_scheduler_metadata, new_num_splits] =
        op::flash_mla::get_mla_decoding_metadata(tile_scheduler_metadata_out,
                                                 num_splits_out,
                                                 cache_seqlens,
                                                 num_q_tokens_per_head_k,
                                                 num_heads_k,
                                                 py_optional_i64_for_flash_mla_metadata(num_heads_q),
                                                 is_fp8_kvcache,
                                                 py_optional_i64_for_flash_mla_metadata(topk));

    return py::make_tuple(to_py_torch_tensor_for_flash_mla_metadata(new_tile_scheduler_metadata),
                          to_py_torch_tensor_for_flash_mla_metadata(new_num_splits));
#endif
    (void)cache_seqlens;
    (void)num_q_tokens_per_head_k;
    (void)num_heads_k;
    (void)num_heads_q;
    (void)is_fp8_kvcache;
    (void)topk;
    (void)tile_scheduler_metadata;
    (void)num_splits;
    throw std::runtime_error("flash_mla_get_mla_decoding_metadata requires an ATen-enabled build.");
}

inline void bind_flash_mla_get_mla_decoding_metadata(py::module &m) {
    m.def("flash_mla_get_mla_decoding_metadata",
          &ops::py_flash_mla_get_mla_decoding_metadata,
          py::arg("cache_seqlens"),
          py::arg("num_q_tokens_per_head_k"),
          py::arg("num_heads_k"),
          py::arg("num_heads_q") = py::none(),
          py::arg("is_fp8_kvcache") = false,
          py::arg("topk") = py::none(),
          py::arg("tile_scheduler_metadata") = py::none(),
          py::arg("num_splits") = py::none(),
          R"doc(FlashMLA bridge for flash_mla_cuda.get_mla_metadata. Returns tile_scheduler_metadata and num_splits.)doc");
}

} // namespace infinicore::ops

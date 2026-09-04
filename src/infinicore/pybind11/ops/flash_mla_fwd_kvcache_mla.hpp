#pragma once

#include <stdexcept>

#include <pybind11/pybind11.h>

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <torch/csrc/utils/pybind.h>
#endif

#include "infinicore/dtype.hpp"
#include "infinicore/ops/flash_mla/fwd_kvcache_mla.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline std::optional<Tensor> py_optional_tensor_for_flash_mla_fwd_kvcache_mla(py::object obj) {
    if (obj.is_none()) {
        return std::nullopt;
    }
    return obj.cast<Tensor>();
}

#ifdef ENABLE_ATEN
namespace {

inline py::object to_py_torch_tensor_for_flash_mla_fwd_kvcache_mla(const Tensor &tensor) {
    if (!tensor) {
        return py::none();
    }
    return py::cast(infinicore::adaptor::to_aten_tensor(tensor));
}

} // namespace
#endif

inline py::object py_flash_mla_fwd_kvcache_mla(Tensor q,
                                               Tensor k_cache,
                                               py::object k_cache_scale,
                                               int64_t head_dim_v,
                                               Tensor cache_seqlens,
                                               Tensor block_table,
                                               double softmax_scale,
                                               bool causal,
                                               Tensor tile_scheduler_metadata,
                                               Tensor num_splits,
                                               bool is_fp8_kvcache,
                                               py::object extra_k_cache,
                                               py::object extra_block_table,
                                               int64_t cp_world_size,
                                               int64_t cp_rank,
                                               py::object cp_tot_seqused_k,
                                               py::object out,
                                               py::object lse) {
#ifdef ENABLE_ATEN
    if (q->ndim() != 4) {
        throw std::runtime_error("flash_mla_fwd_kvcache_mla expects q shape [batch, seq_q, heads, head_dim].");
    }
    if (head_dim_v <= 0) {
        throw std::runtime_error("flash_mla_fwd_kvcache_mla expects positive head_dim_v.");
    }

    auto out_opt = py_optional_tensor_for_flash_mla_fwd_kvcache_mla(out);
    auto lse_opt = py_optional_tensor_for_flash_mla_fwd_kvcache_mla(lse);
    Tensor out_tensor = out_opt.has_value()
                          ? out_opt.value()
                          : Tensor::empty({q->size(0), q->size(1), q->size(2), static_cast<size_t>(head_dim_v)},
                                          q->dtype(),
                                          q->device());
    Tensor lse_tensor = lse_opt.has_value()
                          ? lse_opt.value()
                          : Tensor::empty({q->size(0), q->size(2), q->size(1)},
                                          DataType::F32,
                                          q->device());

    op::flash_mla::fwd_kvcache_mla_(out_tensor,
                                    lse_tensor,
                                    q,
                                    k_cache,
                                    py_optional_tensor_for_flash_mla_fwd_kvcache_mla(k_cache_scale),
                                    head_dim_v,
                                    cache_seqlens,
                                    block_table,
                                    softmax_scale,
                                    causal,
                                    tile_scheduler_metadata,
                                    num_splits,
                                    is_fp8_kvcache,
                                    py_optional_tensor_for_flash_mla_fwd_kvcache_mla(extra_k_cache),
                                    py_optional_tensor_for_flash_mla_fwd_kvcache_mla(extra_block_table),
                                    cp_world_size,
                                    cp_rank,
                                    py_optional_tensor_for_flash_mla_fwd_kvcache_mla(cp_tot_seqused_k));

    return py::make_tuple(to_py_torch_tensor_for_flash_mla_fwd_kvcache_mla(out_tensor),
                          to_py_torch_tensor_for_flash_mla_fwd_kvcache_mla(lse_tensor));
#endif
    (void)q;
    (void)k_cache;
    (void)k_cache_scale;
    (void)head_dim_v;
    (void)cache_seqlens;
    (void)block_table;
    (void)softmax_scale;
    (void)causal;
    (void)tile_scheduler_metadata;
    (void)num_splits;
    (void)is_fp8_kvcache;
    (void)extra_k_cache;
    (void)extra_block_table;
    (void)cp_world_size;
    (void)cp_rank;
    (void)cp_tot_seqused_k;
    (void)out;
    (void)lse;
    throw std::runtime_error("flash_mla_fwd_kvcache_mla requires an ATen-enabled build.");
}

inline void bind_flash_mla_fwd_kvcache_mla(py::module &m) {
    m.def("flash_mla_fwd_kvcache_mla",
          &ops::py_flash_mla_fwd_kvcache_mla,
          py::arg("q"),
          py::arg("k_cache"),
          py::arg("k_cache_scale"),
          py::arg("head_dim_v"),
          py::arg("cache_seqlens"),
          py::arg("block_table"),
          py::arg("softmax_scale"),
          py::arg("causal"),
          py::arg("tile_scheduler_metadata"),
          py::arg("num_splits"),
          py::arg("is_fp8_kvcache"),
          py::arg("extra_k_cache"),
          py::arg("extra_block_table"),
          py::arg("cp_world_size"),
          py::arg("cp_rank"),
          py::arg("cp_tot_seqused_k"),
          py::arg("out") = py::none(),
          py::arg("lse") = py::none(),
          R"doc(FlashMLA bridge for flash_mla_cuda.fwd_kvcache_mla. Returns out and softmax_lse.)doc");
}

} // namespace infinicore::ops

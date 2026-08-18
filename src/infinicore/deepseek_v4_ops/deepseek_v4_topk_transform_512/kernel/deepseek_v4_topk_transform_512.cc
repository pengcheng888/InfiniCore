#include "infinicore/ops/deepseek_v4_topk_transform_512.hpp"

#include "deepseek_v4_topk_transform_512_kernel.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#endif
#endif

#include <limits>
#include <stdexcept>
#include <string>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4TopkTransform512Kernel);

namespace {

constexpr int64_t kC4TopK = 512;

void check_hygon_tensor(const Tensor &tensor, const char *op_name) {
#if defined(ENABLE_HYGON_API)
    if (tensor->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error(std::string(op_name) + " expects HYGON tensors in this build.");
    }
#else
    (void)tensor;
    (void)op_name;
#endif
}

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
void *current_accelerator_stream() {
    return context::getStream();
}

void topk_transform_512_at(const at::Tensor &scores,
                           const at::Tensor &seq_lens,
                           const at::Tensor &page_tables,
                           at::Tensor &out_page_indices,
                           int page_size) {
    if (scores.dim() != 2 || page_tables.dim() != 2 || out_page_indices.dim() != 2) {
        throw std::runtime_error("topk_transform_512 expects scores/page_tables/out_page_indices to be 2-D.");
    }
    const int64_t batch = scores.size(0);
    const int64_t max_seq_len = scores.size(1);
    if (seq_lens.dim() != 1 || seq_lens.size(0) != batch || page_tables.size(0) != batch || out_page_indices.size(0) != batch || out_page_indices.size(1) < kC4TopK) {
        throw std::runtime_error("topk_transform_512 shape mismatch.");
    }
    if (page_size <= 0 || (page_size & (page_size - 1)) != 0) {
        throw std::runtime_error("topk_transform_512 page_size must be a positive power of two.");
    }

    auto int_opts = out_page_indices.options().dtype(at::kInt);
    auto seq = seq_lens.to(at::kInt);
    auto sequential = at::arange(kC4TopK, int_opts).unsqueeze(0).expand({batch, kC4TopK});
    auto negative = at::full({batch, kC4TopK}, -1, int_opts);
    auto sequential_valid = sequential < seq.unsqueeze(1);

    at::Tensor raw_indices;
    at::Tensor valid_topk;
    if (max_seq_len <= kC4TopK) {
        raw_indices = at::where(sequential_valid, sequential, negative);
        valid_topk = sequential_valid;
    } else {
        auto positions = at::arange(max_seq_len, scores.options().dtype(at::kLong)).unsqueeze(0);
        auto valid_mask = positions < seq.to(at::kLong).unsqueeze(1);
        auto masked_scores = scores.masked_fill(~valid_mask, -std::numeric_limits<float>::infinity());
        auto topk_result = at::topk(masked_scores, kC4TopK, 1, true, false);
        raw_indices = std::get<1>(topk_result).to(at::kInt);
        auto gathered_scores = scores.gather(1, raw_indices.to(at::kLong));
        valid_topk = gathered_scores.ne(-std::numeric_limits<float>::infinity());
        auto needs_sequential = (seq <= kC4TopK).unsqueeze(1);
        raw_indices = at::where(needs_sequential, at::where(sequential_valid, sequential, negative), raw_indices);
        valid_topk = at::where(needs_sequential, sequential_valid, valid_topk);
    }

    auto raw_long = raw_indices.to(at::kLong);
    auto page_idx = at::floor_divide(raw_long, page_size);
    auto offset_in_page = at::remainder(raw_long, page_size);
    auto page_idx_clamped = at::clamp_min(page_idx, 0);
    auto physical_pages = page_tables.to(at::kLong).gather(1, page_idx_clamped);
    auto page_indices = (physical_pages * page_size + offset_in_page).to(at::kInt);
    auto transformed = at::where(valid_topk, page_indices, negative);
    out_page_indices.slice(1, 0, kC4TopK).copy_(transformed);
}

void topk_transform_512_dispatch_at(const at::Tensor &scores,
                                    const at::Tensor &seq_lens,
                                    const at::Tensor &page_tables,
                                    at::Tensor &out_page_indices,
                                    int page_size,
                                    const char *op_name) {
    if (scores.size(1) > kC4TopK) {
        topk_transform_512_at(scores, seq_lens, page_tables, out_page_indices, page_size);
        return;
    }
    if (!scores.is_contiguous() || !seq_lens.is_contiguous() || !page_tables.is_contiguous() || !out_page_indices.is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " native topk path expects contiguous tensors.");
    }
    deepseek_v4_topk_transform_512::launch_topk_transform_512(
        reinterpret_cast<const float *>(scores.data_ptr()),
        scores.stride(0),
        seq_lens.data_ptr(),
        seq_lens.scalar_type() == at::kLong,
        page_tables.data_ptr(),
        page_tables.scalar_type() == at::kLong,
        page_tables.stride(0),
        reinterpret_cast<int32_t *>(out_page_indices.data_ptr()),
        out_page_indices.stride(0),
        scores.size(0),
        scores.size(1),
        page_size,
        current_accelerator_stream());
}

void deepseek_v4_topk_transform_512_kernel_impl(const Tensor &scores,
                                                const Tensor &seq_lens,
                                                const Tensor &page_table,
                                                Tensor out_page_indices,
                                                int page_size) {
    constexpr const char *op_name = "deepseek_v4_topk_transform_512_kernel_";
    check_hygon_tensor(scores, op_name);
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
    if (scores->ndim() != 2 || scores->dtype() != DataType::F32) {
        throw std::runtime_error("deepseek_v4_topk_transform_512_kernel_ expects scores [batch, max_seq_len] fp32.");
    }
    if (seq_lens->ndim() != 1 || seq_lens->size(0) != scores->size(0) ||
        (seq_lens->dtype() != DataType::I32 && seq_lens->dtype() != DataType::I64)) {
        throw std::runtime_error("deepseek_v4_topk_transform_512_kernel_ expects seq_lens [batch] int32/int64.");
    }
    if (page_table->ndim() != 2 || page_table->size(0) != scores->size(0) ||
        (page_table->dtype() != DataType::I32 && page_table->dtype() != DataType::I64)) {
        throw std::runtime_error("deepseek_v4_topk_transform_512_kernel_ expects page_table [batch, pages] int32/int64.");
    }
    if (out_page_indices->ndim() != 2 || out_page_indices->size(0) != scores->size(0) ||
        out_page_indices->size(1) < static_cast<size_t>(kC4TopK) || out_page_indices->dtype() != DataType::I32) {
        throw std::runtime_error("deepseek_v4_topk_transform_512_kernel_ expects out_page_indices [batch, >=512] int32.");
    }
    if (page_size <= 0 || (page_size & (page_size - 1)) != 0) {
        throw std::runtime_error("deepseek_v4_topk_transform_512_kernel_ page_size must be a positive power of two.");
    }
    if (!scores->is_contiguous() || !seq_lens->is_contiguous() || !page_table->is_contiguous() || !out_page_indices->is_contiguous()) {
        throw std::runtime_error("deepseek_v4_topk_transform_512_kernel_ expects contiguous tensors.");
    }
    auto scores_at = infinicore::adaptor::to_aten_tensor(scores);
    auto seq_lens_at = infinicore::adaptor::to_aten_tensor(seq_lens);
    auto page_table_at = infinicore::adaptor::to_aten_tensor(page_table);
    auto out_page_indices_at = infinicore::adaptor::to_aten_tensor(out_page_indices);
    topk_transform_512_dispatch_at(scores_at, seq_lens_at, page_table_at, out_page_indices_at, page_size, op_name);
}
#endif

} // namespace

DeepseekV4TopkTransform512Kernel::DeepseekV4TopkTransform512Kernel(const Tensor &scores,
                                                                   const Tensor &seq_lens,
                                                                   const Tensor &page_table,
                                                                   Tensor out_page_indices,
                                                                   int page_size) {
    INFINICORE_GRAPH_OP_DISPATCH(scores->device().getType(), scores, seq_lens, page_table, out_page_indices, page_size);
}

void DeepseekV4TopkTransform512Kernel::execute(const Tensor &scores,
                                               const Tensor &seq_lens,
                                               const Tensor &page_table,
                                               Tensor out_page_indices,
                                               int page_size) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4TopkTransform512Kernel, scores, seq_lens, page_table, out_page_indices, page_size);
}

namespace deepseek_v4_topk_transform_512_graph_impl {

struct PlannedMeta {
    graph::GraphTensor scores;
    graph::GraphTensor seq_lens;
    graph::GraphTensor page_table;
    graph::GraphTensor out_page_indices;
    int page_size;
};

void *plan(const Tensor &scores,
           const Tensor &seq_lens,
           const Tensor &page_table,
           Tensor out_page_indices,
           int page_size) {
    return new PlannedMeta{graph::GraphTensor(scores),
                           graph::GraphTensor(seq_lens),
                           graph::GraphTensor(page_table),
                           graph::GraphTensor(out_page_indices),
                           page_size};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    deepseek_v4_topk_transform_512_kernel_impl(planned->scores,
                                               planned->seq_lens,
                                               planned->page_table,
                                               planned->out_page_indices,
                                               planned->page_size);
#else
    (void)planned;
    throw std::runtime_error("deepseek_v4_topk_transform_512_kernel_ requires an ATen-enabled HYGON build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_topk_transform_512_graph_impl

namespace deepseek_v4_topk_transform_512_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4TopkTransform512Kernel,
                                       &deepseek_v4_topk_transform_512_graph_impl::plan,
                                       &deepseek_v4_topk_transform_512_graph_impl::run,
                                       &deepseek_v4_topk_transform_512_graph_impl::cleanup);
} // namespace deepseek_v4_topk_transform_512_register

void deepseek_v4_topk_transform_512_kernel_(const Tensor &scores,
                                            const Tensor &seq_lens,
                                            const Tensor &page_table,
                                            Tensor out_page_indices,
                                            int page_size) {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    DeepseekV4TopkTransform512Kernel::execute(scores, seq_lens, page_table, out_page_indices, page_size);
#else
    (void)scores;
    (void)seq_lens;
    (void)page_table;
    (void)out_page_indices;
    (void)page_size;
    throw std::runtime_error("deepseek_v4_topk_transform_512_kernel_ requires an ATen-enabled HYGON build.");
#endif
}

} // namespace infinicore::op

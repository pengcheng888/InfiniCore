#include "infinicore/ops/deepseek_v4_topk_transform_512.hpp"

#include "deepseek_v4_topk_transform_512_kernel.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4TopkTransform512SglangKernel);

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

void deepseek_v4_topk_transform_512_sglang_kernel_impl(const Tensor &scores,
                                                       const Tensor &seq_lens,
                                                       const Tensor &page_table,
                                                       Tensor out_page_indices,
                                                       int page_size) {
    constexpr const char *op_name = "deepseek_v4_topk_transform_512_sglang_kernel_";
    check_hygon_tensor(scores, op_name);
    if (scores->ndim() != 2 || scores->dtype() != DataType::F32) {
        throw std::runtime_error("deepseek_v4_topk_transform_512_sglang_kernel_ expects scores [batch, max_seq_len] fp32.");
    }
    if (seq_lens->ndim() != 1 || seq_lens->size(0) != scores->size(0) || seq_lens->dtype() != DataType::I32) {
        throw std::runtime_error("deepseek_v4_topk_transform_512_sglang_kernel_ expects seq_lens [batch] int32.");
    }
    if (page_table->ndim() != 2 || page_table->size(0) != scores->size(0) || page_table->dtype() != DataType::I32) {
        throw std::runtime_error("deepseek_v4_topk_transform_512_sglang_kernel_ expects page_table [batch, pages] int32.");
    }
    if (out_page_indices->ndim() != 2 || out_page_indices->size(0) != scores->size(0) ||
        out_page_indices->size(1) < static_cast<size_t>(kC4TopK) || out_page_indices->dtype() != DataType::I32) {
        throw std::runtime_error("deepseek_v4_topk_transform_512_sglang_kernel_ expects out_page_indices [batch, >=512] int32.");
    }
    if (page_size <= 0 || (page_size & (page_size - 1)) != 0) {
        throw std::runtime_error("deepseek_v4_topk_transform_512_sglang_kernel_ page_size must be a positive power of two.");
    }
    if (scores->stride(1) != 1 || !seq_lens->is_contiguous() || page_table->stride(1) != 1 || !out_page_indices->is_contiguous()) {
        throw std::runtime_error("deepseek_v4_topk_transform_512_sglang_kernel_ expects scores/page_table last dim contiguous and contiguous seq_lens/out.");
    }
    if (scores->size(1) > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("deepseek_v4_topk_transform_512_sglang_kernel_ max_seq_len exceeds int32 range.");
    }

    deepseek_v4_topk_transform_512::launch_topk_transform_512_sglang(
        reinterpret_cast<const float *>(scores->data()),
        scores->stride(0),
        reinterpret_cast<const int32_t *>(seq_lens->data()),
        reinterpret_cast<const int32_t *>(page_table->data()),
        page_table->stride(0),
        reinterpret_cast<int32_t *>(out_page_indices->data()),
        out_page_indices->stride(0),
        scores->size(0),
        scores->size(1),
        page_size,
        context::getStream());
}

} // namespace

DeepseekV4TopkTransform512SglangKernel::DeepseekV4TopkTransform512SglangKernel(const Tensor &scores,
                                                                               const Tensor &seq_lens,
                                                                               const Tensor &page_table,
                                                                               Tensor out_page_indices,
                                                                               int page_size) {
    INFINICORE_GRAPH_OP_DISPATCH(scores->device().getType(), scores, seq_lens, page_table, out_page_indices, page_size);
}

void DeepseekV4TopkTransform512SglangKernel::execute(const Tensor &scores,
                                                     const Tensor &seq_lens,
                                                     const Tensor &page_table,
                                                     Tensor out_page_indices,
                                                     int page_size) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4TopkTransform512SglangKernel, scores, seq_lens, page_table, out_page_indices, page_size);
}

namespace deepseek_v4_topk_transform_512_sglang_graph_impl {

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
    deepseek_v4_topk_transform_512_sglang_kernel_impl(planned->scores,
                                                      planned->seq_lens,
                                                      planned->page_table,
                                                      planned->out_page_indices,
                                                      planned->page_size);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_topk_transform_512_sglang_graph_impl

namespace deepseek_v4_topk_transform_512_sglang_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4TopkTransform512SglangKernel,
                                       &deepseek_v4_topk_transform_512_sglang_graph_impl::plan,
                                       &deepseek_v4_topk_transform_512_sglang_graph_impl::run,
                                       &deepseek_v4_topk_transform_512_sglang_graph_impl::cleanup);
} // namespace deepseek_v4_topk_transform_512_sglang_register

void deepseek_v4_topk_transform_512_sglang_kernel_(const Tensor &scores,
                                                   const Tensor &seq_lens,
                                                   const Tensor &page_table,
                                                   Tensor out_page_indices,
                                                   int page_size) {
    DeepseekV4TopkTransform512SglangKernel::execute(scores, seq_lens, page_table, out_page_indices, page_size);
}

} // namespace infinicore::op

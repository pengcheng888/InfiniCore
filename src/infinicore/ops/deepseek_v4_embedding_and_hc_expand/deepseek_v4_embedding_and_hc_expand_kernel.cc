#include "infinicore/ops/deepseek_v4_embedding_and_hc_expand.hpp"

#include "deepseek_v4_embedding_and_hc_expand_kernel.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4EmbeddingAndHcExpandKernel);

namespace {

Shape output_shape_for_kernel(const Tensor &input, const Tensor &weight, int64_t hc_mult, const char *op_name) {
    if (weight->ndim() != 2) {
        throw std::runtime_error(std::string(op_name) + " expects weight shape [vocab, hidden].");
    }
    if (hc_mult <= 0) {
        throw std::runtime_error(std::string(op_name) + " expects hc_mult > 0.");
    }
    Shape output_shape = input->shape();
    output_shape.push_back(static_cast<size_t>(hc_mult));
    output_shape.push_back(weight->size(1));
    return output_shape;
}

void check_accelerator_tensor(const Tensor &tensor, const char *op_name) {
#if defined(ENABLE_HYGON_API)
    if (tensor->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error(std::string(op_name) + " expects HYGON tensors in this build.");
    }
#elif defined(ENABLE_NVIDIA_API)
    if (tensor->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error(std::string(op_name) + " expects NVIDIA tensors in this build.");
    }
#else
    (void)tensor;
    (void)op_name;
#endif
}

void check_kernel_tensors(const Tensor &out, const Tensor &input, const Tensor &weight, int64_t hc_mult, const char *op_name) {
    check_accelerator_tensor(out, op_name);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input, weight);
    auto expected = output_shape_for_kernel(input, weight, hc_mult, op_name);
    if (out->shape() != expected) {
        throw std::runtime_error(std::string(op_name) + " output shape mismatch.");
    }
    if (out->dtype() != weight->dtype()) {
        throw std::runtime_error(std::string(op_name) + " output dtype must match weight dtype.");
    }
    if (input->dtype() != DataType::I32 && input->dtype() != DataType::I64) {
        throw std::runtime_error(std::string(op_name) + " expects int32 or int64 input indices.");
    }
    if (weight->dtype() != DataType::BF16 && weight->dtype() != DataType::F16 && weight->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " supports bf16/fp16/fp32 weight tensors only.");
    }
    if (!out->is_contiguous() || !input->is_contiguous() || !weight->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous tensors.");
    }
}

} // namespace

DeepseekV4EmbeddingAndHcExpandKernel::DeepseekV4EmbeddingAndHcExpandKernel(Tensor out, const Tensor &input, const Tensor &weight, int64_t hc_mult) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input, weight);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, input, weight, hc_mult);
}

void DeepseekV4EmbeddingAndHcExpandKernel::execute(Tensor out, const Tensor &input, const Tensor &weight, int64_t hc_mult) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4EmbeddingAndHcExpandKernel, out, input, weight, hc_mult);
}

namespace deepseek_v4_embedding_and_hc_expand_kernel_graph_impl {

struct PlannedMeta {
    graph::GraphTensor out;
    graph::GraphTensor input;
    graph::GraphTensor weight;
    int64_t tokens;
    int64_t hc_mult;
    int64_t hidden;
    int64_t vocab;
    DataType out_dtype;
    DataType input_dtype;
};

void *plan(Tensor out, const Tensor &input, const Tensor &weight, int64_t hc_mult) {
    check_kernel_tensors(out, input, weight, hc_mult, "deepseek_v4_embedding_and_hc_expand_kernel_");
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(input),
        graph::GraphTensor(weight),
        static_cast<int64_t>(input->numel()),
        hc_mult,
        static_cast<int64_t>(weight->size(1)),
        static_cast<int64_t>(weight->size(0)),
        out->dtype(),
        input->dtype()};
}

void run(void *planned_meta) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    deepseek_v4_embedding_and_hc_expand_kernel_impl::launch_embedding(
        planned->out->data(),
        planned->input->data(),
        planned->weight->data(),
        planned->tokens,
        planned->hc_mult,
        planned->hidden,
        planned->vocab,
        planned->out_dtype,
        planned->input_dtype,
        context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("deepseek_v4_embedding_and_hc_expand_kernel_ requires a HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_embedding_and_hc_expand_kernel_graph_impl

namespace deepseek_v4_embedding_and_hc_expand_kernel_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4EmbeddingAndHcExpandKernel,
                                       &deepseek_v4_embedding_and_hc_expand_kernel_graph_impl::plan,
                                       &deepseek_v4_embedding_and_hc_expand_kernel_graph_impl::run,
                                       &deepseek_v4_embedding_and_hc_expand_kernel_graph_impl::cleanup);
} // namespace deepseek_v4_embedding_and_hc_expand_kernel_register

Tensor deepseek_v4_embedding_and_hc_expand_kernel(const Tensor &input, const Tensor &weight, int64_t hc_mult) {
    auto out = Tensor::empty(output_shape_for_kernel(input, weight, hc_mult, "deepseek_v4_embedding_and_hc_expand_kernel"), weight->dtype(), weight->device());
    deepseek_v4_embedding_and_hc_expand_kernel_(out, input, weight, hc_mult);
    return out;
}

void deepseek_v4_embedding_and_hc_expand_kernel_(Tensor out, const Tensor &input, const Tensor &weight, int64_t hc_mult) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    check_kernel_tensors(out, input, weight, hc_mult, "deepseek_v4_embedding_and_hc_expand_kernel_");
    DeepseekV4EmbeddingAndHcExpandKernel::execute(out, input, weight, hc_mult);
#else
    (void)out;
    (void)input;
    (void)weight;
    (void)hc_mult;
    throw std::runtime_error("deepseek_v4_embedding_and_hc_expand_kernel_ requires a HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op

#include "infinicore/ops/deepseek_v4_expand_hc.hpp"

#include "deepseek_v4_expand_hc_kernel.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4ExpandHcKernel);

namespace {

void check_tensors(const Tensor &output, const Tensor &input, int64_t hc, const char *op_name) {
#if defined(ENABLE_HYGON_API)
    if (input->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error(std::string(op_name) + " expects HYGON tensors in this build.");
    }
#elif defined(ENABLE_NVIDIA_API)
    if (input->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error(std::string(op_name) + " expects NVIDIA tensors in this build.");
    }
#endif
    if (input->ndim() != 2 || output->ndim() != 3 || hc <= 0) {
        throw std::runtime_error(std::string(op_name) + " expects input [tokens, hidden] and output [tokens, hc, hidden].");
    }
    if (output->size(0) != input->size(0) ||
        output->size(1) != static_cast<size_t>(hc) ||
        output->size(2) != input->size(1) ||
        output->dtype() != input->dtype() ||
        output->device() != input->device()) {
        throw std::runtime_error(std::string(op_name) + " output shape, dtype, or device mismatch.");
    }
    if (!input->is_contiguous() || !output->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous input and output.");
    }
    if (input->dtype() != DataType::BF16 && input->dtype() != DataType::F16 && input->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(op_name) + " expects bf16/fp16/fp32 input.");
    }
}

int element_size(DataType dtype) {
    if (dtype == DataType::F32) {
        return 4;
    }
    return 2;
}

void run_impl(Tensor output, const Tensor &input, int64_t hc) {
    constexpr const char *op_name = "deepseek_v4_expand_hc_";
    check_tensors(output, input, hc, op_name);
    if (context::getDevice() != output->device()) {
        context::setDevice(output->device());
    }
    deepseek_v4_expand_hc_impl::launch_expand_hc(output->data(),
                                                 input->data(),
                                                 static_cast<int64_t>(input->size(0)),
                                                 hc,
                                                 static_cast<int64_t>(input->size(1)),
                                                 element_size(input->dtype()),
                                                 context::getStream());
}

} // namespace

DeepseekV4ExpandHcKernel::DeepseekV4ExpandHcKernel(Tensor output, const Tensor &input, int64_t hc) {
    device_graph_capture_supported_ = true;
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, input, hc);
}

void DeepseekV4ExpandHcKernel::execute(Tensor output, const Tensor &input, int64_t hc) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4ExpandHcKernel, output, input, hc);
}

namespace deepseek_v4_expand_hc_graph_impl {

struct PlannedMeta {
    graph::GraphTensor output;
    graph::GraphTensor input;
    int64_t hc;
};

void *plan(Tensor output, const Tensor &input, int64_t hc) {
    check_tensors(output, input, hc, "deepseek_v4_expand_hc_");
    return new PlannedMeta{graph::GraphTensor(output), graph::GraphTensor(input), hc};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    run_impl(planned->output, planned->input, planned->hc);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_expand_hc_graph_impl

namespace deepseek_v4_expand_hc_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4ExpandHcKernel,
                                       &deepseek_v4_expand_hc_graph_impl::plan,
                                       &deepseek_v4_expand_hc_graph_impl::run,
                                       &deepseek_v4_expand_hc_graph_impl::cleanup);
} // namespace deepseek_v4_expand_hc_register

void deepseek_v4_expand_hc_(Tensor output, const Tensor &input, int64_t hc) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    DeepseekV4ExpandHcKernel::execute(output, input, hc);
#else
    (void)output;
    (void)input;
    (void)hc;
    throw std::runtime_error("deepseek_v4_expand_hc_ requires a HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op

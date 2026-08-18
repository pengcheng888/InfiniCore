#include "infinicore/ops/deepseek_v4_silu_and_mul.hpp"

#include "deepseek_v4_silu_and_mul_kernel.hpp"

#include "../../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#include <stdexcept>
#include <string>

namespace infinicore::op {

namespace {

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

void check_kernel_tensors(const Tensor &out, const Tensor &x, const char *op_name) {
    check_accelerator_tensor(x, op_name);
    if (x->ndim() < 1 || x->size(x->ndim() - 1) % 2 != 0) {
        throw std::runtime_error(std::string(op_name) + " expects input last dim to be even.");
    }
    if (out->ndim() != x->ndim()) {
        throw std::runtime_error(std::string(op_name) + " output rank mismatch.");
    }
    for (size_t i = 0; i + 1 < x->ndim(); ++i) {
        if (out->size(i) != x->size(i)) {
            throw std::runtime_error(std::string(op_name) + " output shape mismatch.");
        }
    }
    if (out->size(out->ndim() - 1) * 2 != x->size(x->ndim() - 1)) {
        throw std::runtime_error(std::string(op_name) + " output last dim mismatch.");
    }
    if (x->dtype() != out->dtype() || (x->dtype() != DataType::BF16 && x->dtype() != DataType::F16)) {
        throw std::runtime_error(std::string(op_name) + " expects bf16/fp16 input and matching output dtype.");
    }
    if (!out->is_contiguous() || !x->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous tensors.");
    }
}

int64_t token_count(const Tensor &x) {
    const auto hidden2 = x->size(x->ndim() - 1);
    return static_cast<int64_t>(x->numel() / hidden2);
}

} // namespace

namespace deepseek_v4 {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(SiluAndMul);

SiluAndMul::SiluAndMul(Tensor out, const Tensor &x) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, x);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, x);
}

void SiluAndMul::execute(Tensor out, const Tensor &x) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(SiluAndMul, out, x);
}

namespace deepseek_v4_silu_and_mul_impl {

struct PlannedMeta {
    graph::GraphTensor out;
    graph::GraphTensor x;
    int64_t tokens;
    int64_t hidden;
    DataType dtype;
};

void *plan(Tensor out, const Tensor &x) {
    check_kernel_tensors(out, x, "deepseek_v4_silu_and_mul_kernel_");
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(x),
        token_count(x),
        static_cast<int64_t>(out->size(out->ndim() - 1)),
        x->dtype()};
}

void run(void *planned_meta) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    ::infinicore::op::deepseek_v4_silu_and_mul_impl::launch_silu_and_mul(
        planned->out->data(),
        planned->x->data(),
        planned->tokens,
        planned->hidden,
        planned->dtype,
        context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("deepseek_v4_silu_and_mul_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_silu_and_mul_impl

namespace deepseek_v4_silu_and_mul_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(SiluAndMul,
                                       &deepseek_v4_silu_and_mul_impl::plan,
                                       &deepseek_v4_silu_and_mul_impl::run,
                                       &deepseek_v4_silu_and_mul_impl::cleanup);
} // namespace deepseek_v4_silu_and_mul_register

} // namespace deepseek_v4

void deepseek_v4_silu_and_mul_kernel_(Tensor out, const Tensor &x) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    check_kernel_tensors(out, x, "deepseek_v4_silu_and_mul_kernel_");
    deepseek_v4::SiluAndMul::execute(out, x);
#else
    (void)out;
    (void)x;
    throw std::runtime_error("deepseek_v4_silu_and_mul_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op

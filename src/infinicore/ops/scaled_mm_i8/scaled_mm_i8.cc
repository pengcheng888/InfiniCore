#include "infinicore/ops/scaled_mm_i8.hpp"

#include "../../utils.hpp"

#if defined(ENABLE_ILUVATAR_VENDOR_OPS)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/iluvatar_vendor_adaptor.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#include <ATen/ATen.h>
#include <optional>
#include <stdexcept>
#include <string>
#endif

namespace infinicore::op {

#if defined(ENABLE_ILUVATAR_VENDOR_OPS)
namespace scaled_mm_i8_impl::iluvatar {
namespace {

void validate_inputs(Tensor c,
                     const Tensor &a_p,
                     const Tensor &a_s,
                     const Tensor &b_p,
                     const Tensor &b_s,
                     std::optional<Tensor> bias,
                     const char *op_name) {
    if (!c || !a_p || !a_s || !b_p || !b_s
        || c->device().getType() != Device::Type::ILUVATAR
        || a_p->device().getType() != Device::Type::ILUVATAR
        || a_s->device().getType() != Device::Type::ILUVATAR
        || b_p->device().getType() != Device::Type::ILUVATAR
        || b_s->device().getType() != Device::Type::ILUVATAR
        || (bias.has_value() && bias.value()
            && bias.value()->device().getType() != Device::Type::ILUVATAR)) {
        throw std::runtime_error(std::string(op_name) + " expects ILUVATAR tensors.");
    }
    if (c->ndim() != 2 || a_p->ndim() != 2 || a_s->ndim() != 2
        || b_p->ndim() != 2 || b_s->ndim() != 2) {
        throw std::runtime_error(std::string(op_name) + " expects 2D tensors.");
    }
    if (a_p->dtype() != DataType::I8 || b_p->dtype() != DataType::I8
        || a_s->dtype() != DataType::F32 || b_s->dtype() != DataType::F32
        || (c->dtype() != DataType::F16 && c->dtype() != DataType::BF16)) {
        throw std::runtime_error(std::string(op_name) + " expects int8 operands, float32 scales, and fp16/bf16 output.");
    }
    if (a_p->size(0) != c->size(0) || a_p->size(1) != b_p->size(0)
        || b_p->size(1) != c->size(1)) {
        throw std::runtime_error(std::string(op_name) + " shape mismatch.");
    }
    if (a_s->shape() != Shape{a_p->size(0), 1} || b_s->shape() != Shape{c->size(1), 1}) {
        throw std::runtime_error(std::string(op_name) + " expects scales [M,1] and [N,1].");
    }
    if (bias.has_value() && bias.value()) {
        const auto &bias_value = bias.value();
        const bool row_broadcast = bias_value->ndim() == 2
                                   && bias_value->size(0) == c->size(0)
                                   && bias_value->size(1) == c->size(1)
                                   && bias_value->strides()[0] == 0
                                   && bias_value->strides()[1] == 1;
        const bool row_vector = bias_value->ndim() == 1
                                && bias_value->size(0) == c->size(1);
        const bool bias_contiguous = row_broadcast || bias_value->is_contiguous();
        if ((row_broadcast || row_vector) && bias_value->dtype() != c->dtype()) {
            throw std::runtime_error(std::string(op_name) + " bias dtype must match output.");
        }
        if (!row_broadcast && !row_vector) {
            throw std::runtime_error(std::string(op_name) + " expects bias [N] or broadcast [M,N].");
        }
        if (!bias_contiguous) {
            throw std::runtime_error(std::string(op_name) + " expects a contiguous or row-broadcast bias.");
        }
    }
    if (!c->is_contiguous() || !a_p->is_contiguous() || !a_s->is_contiguous()
        || !b_s->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous output, activation, scales, and bias.");
    }
}

struct PlannedMeta {
    graph::GraphTensor c;
    graph::GraphTensor a_p;
    graph::GraphTensor a_s;
    graph::GraphTensor b_p;
    graph::GraphTensor b_s;
    std::optional<graph::GraphTensor> bias;
};

void *plan(Tensor c,
           const Tensor &a_p,
           const Tensor &a_s,
           const Tensor &b_p,
           const Tensor &b_s,
           std::optional<Tensor> bias) {
    validate_inputs(c, a_p, a_s, b_p, b_s, bias, "scaled_mm_i8_");
    return new PlannedMeta{
        graph::GraphTensor(c),
        graph::GraphTensor(a_p),
        graph::GraphTensor(a_s),
        graph::GraphTensor(b_p),
        graph::GraphTensor(b_s),
        bias.has_value() && bias.value()
            ? std::optional<graph::GraphTensor>(graph::GraphTensor(bias.value()))
            : std::nullopt};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    if (!adaptor::iluvatar_vendor::scaled_mm_w8a8_available()) {
        throw std::runtime_error("scaled_mm_i8_ requires the Iluvatar scaled_mm_w8a8 vendor extension");
    }

    adaptor::set_aten_stream_to_infinicore();
    auto c_at = adaptor::to_aten_tensor(planned->c);
    auto a_at = adaptor::to_aten_tensor(planned->a_p);
    auto a_s_at = adaptor::to_aten_tensor(planned->a_s);
    auto b_at = adaptor::to_aten_tensor(planned->b_p);
    auto b_s_at = adaptor::to_aten_tensor(planned->b_s);
    std::optional<at::Tensor> bias_at;
    if (planned->bias.has_value()) {
        bias_at = adaptor::to_aten_tensor(planned->bias.value());
        if (bias_at->dim() == 2) {
            if (bias_at->stride(0) != 0) {
                throw std::runtime_error("scaled_mm_i8_ broadcast bias rows must share storage.");
            }
            bias_at = bias_at->select(0, 0);
        }
    }

    // linear_w8a8i8 passes a [K,N] transpose view of row-major [N,K] weights.
    // Recover the zero-copy [N,K] view for the vendor's trans_weight path;
    // keep the NN path for callers that already provide contiguous [K,N].
    bool trans_weight = false;
    if (!b_at.is_contiguous()) {
        if (b_at.stride(0) != 1 || b_at.stride(1) != b_at.size(0)) {
            throw std::runtime_error("scaled_mm_i8_ expects contiguous b [K,N] or a transposed [N,K] weight view.");
        }
        b_at = b_at.transpose(0, 1);
        trans_weight = true;
    }

    adaptor::iluvatar_vendor::scaled_mm_w8a8(
        c_at, a_at, b_at, a_s_at, b_s_at, bias_at, trans_weight);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace

static bool registered = []() {
    I8Gemm::plan_dispatcher().registerDevice(Device::Type::ILUVATAR, &plan);
    I8Gemm::run_dispatcher().registerDevice(Device::Type::ILUVATAR, &run);
    I8Gemm::cleanup_dispatcher().registerDevice(Device::Type::ILUVATAR, &cleanup);
    return true;
}();

} // namespace scaled_mm_i8_impl::iluvatar
#endif

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(I8Gemm);

I8Gemm::I8Gemm(Tensor c, const Tensor &a_p, const Tensor &a_s, const Tensor &b_p, const Tensor &b_s, std::optional<Tensor> bias) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a_p, a_s, b_p, b_s);
    INFINICORE_GRAPH_OP_DISPATCH(c->device().getType(), c, a_p, a_s, b_p, b_s, bias);
}
void I8Gemm::execute(Tensor c, const Tensor &a_p, const Tensor &a_s, const Tensor &b_p, const Tensor &b_s, std::optional<Tensor> bias) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(I8Gemm, c, a_p, a_s, b_p, b_s, bias);
}

void scaled_mm_i8_(Tensor c, const Tensor &a_p, const Tensor &a_s, const Tensor &b_p, const Tensor &b_s, std::optional<Tensor> bias) {
    I8Gemm::execute(c, a_p, a_s, b_p, b_s, bias);
}

} // namespace infinicore::op

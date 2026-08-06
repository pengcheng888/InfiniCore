#include "infinicore/ops/deepseek_v4_compressor_kv_score.hpp"

#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/ops/linear.hpp"

#include <optional>
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

void check_common(const Tensor &out, const Tensor &x, Size expected_out_features, const char *op_name) {
    check_accelerator_tensor(x, op_name);
    if (x->ndim() != 2) {
        throw std::runtime_error(std::string(op_name) + " expects 2D input tensor.");
    }
    if (out->shape() != Shape{x->size(0), expected_out_features}) {
        throw std::runtime_error(std::string(op_name) + " output shape mismatch.");
    }
    if (x->dtype() != DataType::BF16) {
        throw std::runtime_error(std::string(op_name) + " expects bf16 input tensor.");
    }
    if (out->dtype() != DataType::BF16) {
        throw std::runtime_error(std::string(op_name) + " output dtype must be bf16.");
    }
    if (!out->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous output tensor.");
    }
    if (!x->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous input tensor.");
    }
}

void check_weight(const Tensor &x, const Tensor &weight, const char *name, const char *op_name) {
    if (weight->ndim() != 2) {
        throw std::runtime_error(std::string(op_name) + " expects 2D " + name + " tensor.");
    }
    if (x->size(1) != weight->size(1)) {
        throw std::runtime_error(std::string(op_name) + " input/" + name + " K dimension mismatch.");
    }
    if (weight->dtype() != DataType::BF16) {
        throw std::runtime_error(std::string(op_name) + " expects bf16 " + name + " tensor.");
    }
    if (!weight->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous " + name + " tensor.");
    }
}

} // namespace

void deepseek_v4_compressor_kv_score_packed_(Tensor out, const Tensor &x, const Tensor &wkv_gate) {
    constexpr const char *op_name = "deepseek_v4_compressor_kv_score_packed_";
    check_weight(x, wkv_gate, "wkv_gate", op_name);
    check_common(out, x, wkv_gate->size(0), op_name);
    linear_(out, x, wkv_gate, std::nullopt, 1.0f);
    return;
}

void deepseek_v4_compressor_kv_score_unpacked_(Tensor out, const Tensor &x, const Tensor &wkv, const Tensor &wgate) {
    constexpr const char *op_name = "deepseek_v4_compressor_kv_score_unpacked_";
    check_weight(x, wkv, "wkv", op_name);
    check_weight(x, wgate, "wgate", op_name);
    if (wkv->size(0) != wgate->size(0)) {
        throw std::runtime_error(std::string(op_name) + " wkv/wgate output dimension mismatch.");
    }
    const Size proj_size = wkv->size(0);
    check_common(out, x, proj_size * 2, op_name);
    auto kv_out = out->narrow({{1, 0, proj_size}});
    auto gate_out = out->narrow({{1, proj_size, proj_size}});
    linear_(kv_out, x, wkv, std::nullopt, 1.0f);
    linear_(gate_out, x, wgate, std::nullopt, 1.0f);
    return;
}

} // namespace infinicore::op

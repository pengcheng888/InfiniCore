#include "infinicore/ops/qwen3_silu_and_mul.hpp"

#include "infinicore/device.hpp"
#include "infinicore/ops/silu_and_mul.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#if defined(ENABLE_NVIDIA_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#include <stdexcept>

namespace infinicore::op {

namespace {

#if defined(ENABLE_ATEN) && defined(ENABLE_NVIDIA_API)
void qwen3_silu_and_mul_cuda_(Tensor out, const Tensor &x) {
    auto x_shape = x->shape();
    if (x_shape.empty() || x_shape.back() % 2 != 0) {
        throw std::runtime_error("qwen3_silu_and_mul input last dim must be even.");
    }
    auto out_shape = out->shape();
    if (out_shape.size() != x_shape.size()) {
        throw std::runtime_error("qwen3_silu_and_mul output rank mismatch.");
    }
    for (size_t i = 0; i + 1 < x_shape.size(); ++i) {
        if (out_shape[i] != x_shape[i]) {
            throw std::runtime_error("qwen3_silu_and_mul output shape mismatch.");
        }
    }
    if (out_shape.back() * 2 != x_shape.back()) {
        throw std::runtime_error("qwen3_silu_and_mul output last dim mismatch.");
    }

    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
    auto input = infinicore::adaptor::to_aten_tensor(x);
    auto output = infinicore::adaptor::to_aten_tensor(out);
    const auto split_dim = static_cast<int64_t>(x_shape.back() / 2);
    auto gate = input.narrow(-1, 0, split_dim);
    auto up = input.narrow(-1, split_dim, split_dim);
    output.copy_(at::silu(gate) * up);
}
#endif

} // namespace

Tensor qwen3_silu_and_mul(const Tensor &x) {
    Shape shape = x->shape();
    if (shape.empty() || shape.back() % 2 != 0) {
        throw std::runtime_error("qwen3_silu_and_mul input last dim must be even.");
    }
    shape.back() /= 2;
    auto out = Tensor::empty(shape, x->dtype(), x->device());
    qwen3_silu_and_mul_(out, x);
    return out;
}

void qwen3_silu_and_mul_(Tensor out, const Tensor &x) {
#if defined(ENABLE_ATEN) && defined(ENABLE_NVIDIA_API)
    if (x->device().getType() == Device::Type::NVIDIA) {
        qwen3_silu_and_mul_cuda_(out, x);
        return;
    }
#endif
    silu_and_mul_(out, x);
}

} // namespace infinicore::op

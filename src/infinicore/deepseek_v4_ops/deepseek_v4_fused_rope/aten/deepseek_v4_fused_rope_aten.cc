#include "infinicore/ops/deepseek_v4_fused_rope.hpp"

#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#elif defined(ENABLE_NVIDIA_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::op {

namespace {

constexpr int64_t kDsv4RopeDim = 64;

void check_tensor_device(const Tensor &tensor, const char *op_name) {
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

void check_rope_tensor(const Tensor &tensor, const char *name) {
    if (tensor->ndim() != 2 && tensor->ndim() != 3) {
        throw std::runtime_error(std::string("deepseek_v4_fused_rope_aten_ expects ") + name + " rank 2 or 3.");
    }
    if (tensor->size(tensor->ndim() - 1) != static_cast<size_t>(kDsv4RopeDim)) {
        throw std::runtime_error(std::string("deepseek_v4_fused_rope_aten_ expects ") + name + " last dim 64.");
    }
    if (tensor->dtype() != DataType::BF16) {
        throw std::runtime_error(std::string("deepseek_v4_fused_rope_aten_ expects ") + name + " dtype bf16.");
    }
}

#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
void apply_rope_aten_(at::Tensor x, const at::Tensor &freqs_cis, const at::Tensor &positions, bool inverse) {
    if (x.dim() != 2 && x.dim() != 3) {
        throw std::runtime_error("deepseek_v4_fused_rope_aten_ ATen tensor rank must be 2 or 3.");
    }
    if (x.size(-1) != kDsv4RopeDim) {
        throw std::runtime_error("deepseek_v4_fused_rope_aten_ ATen tensor last dim must be 64.");
    }
    const int64_t batch = x.size(0);
    if (positions.numel() != batch) {
        throw std::runtime_error("deepseek_v4_fused_rope_aten_ positions length mismatch.");
    }
    if (batch == 0) {
        return;
    }

    auto pos_long = positions.reshape({batch}).to(at::kLong);
    auto selected = freqs_cis.index_select(0, pos_long).to(at::kFloat).reshape({batch, kDsv4RopeDim / 2, 2});
    auto freq_real = selected.select(-1, 0);
    auto freq_imag = selected.select(-1, 1);

    std::vector<int64_t> pair_shape;
    pair_shape.reserve(static_cast<size_t>(x.dim() + 1));
    for (int64_t i = 0; i < x.dim() - 1; ++i) {
        pair_shape.push_back(x.size(i));
    }
    pair_shape.push_back(kDsv4RopeDim / 2);
    pair_shape.push_back(2);

    auto x_pair = x.to(at::kFloat).reshape(pair_shape);
    auto x_real = x_pair.select(-1, 0);
    auto x_imag = x_pair.select(-1, 1);
    if (x.dim() == 3) {
        freq_real = freq_real.unsqueeze(1);
        freq_imag = freq_imag.unsqueeze(1);
    }

    at::Tensor out_real;
    at::Tensor out_imag;
    if (inverse) {
        out_real = x_real * freq_real + x_imag * freq_imag;
        out_imag = x_imag * freq_real - x_real * freq_imag;
    } else {
        out_real = x_real * freq_real - x_imag * freq_imag;
        out_imag = x_real * freq_imag + x_imag * freq_real;
    }
    auto result = at::stack({out_real, out_imag}, -1).reshape(x.sizes()).to(x.scalar_type());
    x.copy_(result);
}
#endif

} // namespace

void deepseek_v4_fused_rope_aten_(Tensor query,
                                  std::optional<Tensor> key,
                                  const Tensor &freqs_cis,
                                  const Tensor &positions,
                                  bool inverse) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    check_tensor_device(query, "deepseek_v4_fused_rope_aten_");
    check_tensor_device(freqs_cis, "deepseek_v4_fused_rope_aten_");
    check_tensor_device(positions, "deepseek_v4_fused_rope_aten_");
    check_rope_tensor(query, "query");
    if (key.has_value()) {
        check_tensor_device(key.value(), "deepseek_v4_fused_rope_aten_");
        check_rope_tensor(key.value(), "key");
        if (key.value()->size(0) != query->size(0)) {
            throw std::runtime_error("deepseek_v4_fused_rope_aten_ key/query batch mismatch.");
        }
    }
    if (freqs_cis->ndim() != 2 || freqs_cis->size(1) != static_cast<size_t>(kDsv4RopeDim) || freqs_cis->dtype() != DataType::F32) {
        throw std::runtime_error("deepseek_v4_fused_rope_aten_ expects freqs_cis [max_pos, 64] float32.");
    }
    if (positions->ndim() != 1 || (positions->dtype() != DataType::I32 && positions->dtype() != DataType::I64)) {
        throw std::runtime_error("deepseek_v4_fused_rope_aten_ expects positions [tokens] int32/int64.");
    }

    auto freqs_at = infinicore::adaptor::to_aten_tensor(freqs_cis);
    auto positions_at = infinicore::adaptor::to_aten_tensor(positions);
    apply_rope_aten_(infinicore::adaptor::to_aten_tensor(query), freqs_at, positions_at, inverse);
    if (key.has_value()) {
        apply_rope_aten_(infinicore::adaptor::to_aten_tensor(key.value()), freqs_at, positions_at, inverse);
    }
#else
    (void)query;
    (void)key;
    (void)freqs_cis;
    (void)positions;
    (void)inverse;
    throw std::runtime_error("deepseek_v4_fused_rope_aten_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op

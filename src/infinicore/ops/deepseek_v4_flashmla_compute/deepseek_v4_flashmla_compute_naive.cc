#include "infinicore/ops/deepseek_v4_flashmla_compute.hpp"

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

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::op {

namespace {

constexpr int64_t kDsv4FlashMlaQDim = 512;

void check_hygon_or_nvidia_tensor(const Tensor &tensor, const char *op_name) {
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

} // namespace

Tensor deepseek_v4_c4_compress_prefill_naive(const Tensor &kv_score_input,
                                                 const Tensor &ape) {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    check_hygon_or_nvidia_tensor(kv_score_input, "deepseek_v4_c4_compress_prefill_naive");
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());

    if (kv_score_input->ndim() != 2 || kv_score_input->size(1) % 4 != 0) {
        throw std::runtime_error("deepseek_v4_c4_compress_prefill_naive expects kv_score_input [tokens, 4 * head_dim].");
    }
    const int64_t tokens = static_cast<int64_t>(kv_score_input->size(0));
    const int64_t head_dim = static_cast<int64_t>(kv_score_input->size(1) / 4);
    if (head_dim != kDsv4FlashMlaQDim) {
        throw std::runtime_error("deepseek_v4_c4_compress_prefill_naive expects head_dim 512.");
    }
    if (ape->ndim() != 2) {
        throw std::runtime_error("deepseek_v4_c4_compress_prefill_naive expects ape rank 2.");
    }

    auto output = Tensor::zeros({static_cast<size_t>(tokens), static_cast<size_t>(head_dim)}, kv_score_input->dtype(), kv_score_input->device());
    if (tokens == 0) {
        return output;
    }

    auto kv_score_at = infinicore::adaptor::to_aten_tensor(kv_score_input).contiguous().to(at::kFloat)
                           .reshape({tokens, 4, head_dim});
    auto ape_at = infinicore::adaptor::to_aten_tensor(ape).contiguous().to(at::kFloat);
    at::Tensor ape_view;
    if (ape_at.dim() == 2 && ape_at.size(0) == 4 && ape_at.size(1) == 2 * head_dim) {
        auto ape_chunks = ape_at.reshape({4, 2, head_dim});
        // SGLang applies the non-2604 C4 APE hotfix after loading: [score, overlap] -> [overlap, score].
        ape_view = at::cat({ape_chunks.select(1, 1), ape_chunks.select(1, 0)}, 0).contiguous();
    } else if (ape_at.dim() == 2 && ape_at.size(0) == 8 && ape_at.size(1) == head_dim) {
        ape_view = ape_at;
    } else {
        throw std::runtime_error("deepseek_v4_c4_compress_prefill_naive expects ape [4, 1024] or [8, 512].");
    }

    auto output_at = infinicore::adaptor::to_aten_tensor(output);
    for (int64_t end = 3; end < tokens; end += 4) {
        std::vector<at::Tensor> kv_parts;
        std::vector<at::Tensor> score_parts;
        std::vector<at::Tensor> bias_parts;
        const int64_t overlap_start = std::max<int64_t>(0, end - 7);
        const int64_t overlap_end = end - 3;
        if (overlap_end > overlap_start) {
            const int64_t overlap_len = overlap_end - overlap_start;
            auto overlap = kv_score_at.slice(0, overlap_start, overlap_end);
            kv_parts.push_back(overlap.select(1, 0));
            score_parts.push_back(overlap.select(1, 2));
            bias_parts.push_back(ape_view.slice(0, 4 - overlap_len, 4));
        }

        const int64_t normal_start = std::max<int64_t>(0, end - 3);
        const int64_t normal_end = end + 1;
        auto normal = kv_score_at.slice(0, normal_start, normal_end);
        const int64_t normal_len = normal_end - normal_start;
        kv_parts.push_back(normal.select(1, 1));
        score_parts.push_back(normal.select(1, 3));
        bias_parts.push_back(ape_view.slice(0, 8 - normal_len, 8));

        auto kv_window = at::cat(kv_parts, 0);
        auto score_window = at::cat(score_parts, 0) + at::cat(bias_parts, 0);
        auto prob = at::softmax(score_window, 0);
        auto compressed = (kv_window * prob).sum(0);
        output_at.select(0, end).copy_(compressed.to(output_at.scalar_type()));
    }
    return output;
#else
    (void)kv_score_input;
    (void)ape;
    throw std::runtime_error("deepseek_v4_c4_compress_prefill_naive requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}






} // namespace infinicore::op

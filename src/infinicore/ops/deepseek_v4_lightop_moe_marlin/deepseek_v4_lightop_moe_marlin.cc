#include "infinicore/ops/deepseek_v4_lightop_moe_marlin.hpp"

#include "infinicore/device.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#elif defined(ENABLE_NVIDIA_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::op {

namespace {

#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
void guard_device(const Tensor &tensor, const char *op_name) {
#if defined(ENABLE_HYGON_API)
    if (tensor->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error(std::string(op_name) + " expects HYGON tensors in this build.");
    }
#else
    if (tensor->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error(std::string(op_name) + " expects NVIDIA tensors in this build.");
    }
#endif
}

template <typename Fn>
Fn checked_symbol(void *handle, const char *name) {
    dlerror();
    void *symbol = dlsym(handle, name);
    const char *error = dlerror();
    if (error != nullptr || symbol == nullptr) {
        throw std::runtime_error(std::string("lightop SO is missing required symbol ") + name +
                                 (error != nullptr ? std::string(": ") + error : ""));
    }
    return reinterpret_cast<Fn>(symbol);
}

void *open_lightop_so() {
    std::vector<std::string> candidates;
    if (const char *env_path = std::getenv("INFINICORE_LIGHTOP_OP_SO")) {
        if (env_path[0] != '\0') {
            candidates.emplace_back(env_path);
        }
    }
    candidates.emplace_back("/usr/local/lib/python3.10/dist-packages/lightop/op.cpython-310-x86_64-linux-gnu.so");
    candidates.emplace_back("/usr/local/lib/python3.11/dist-packages/lightop/op.cpython-311-x86_64-linux-gnu.so");
    candidates.emplace_back("op.cpython-310-x86_64-linux-gnu.so");
    candidates.emplace_back("op.cpython-311-x86_64-linux-gnu.so");

    std::ostringstream errors;
    for (const auto &path : candidates) {
        void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle != nullptr) {
            return handle;
        }
        if (const char *error = dlerror()) {
            errors << "\n  " << path << ": " << error;
        }
    }
    throw std::runtime_error("failed to load lightop op SO. Set INFINICORE_LIGHTOP_OP_SO to lightop/op*.so." + errors.str());
}

struct LightopSymbols {
    using MoeAlignFn = void (*)(at::Tensor,
                                long,
                                long,
                                at::Tensor,
                                at::Tensor,
                                at::Tensor,
                                const std::optional<at::Tensor> &,
                                const std::optional<at::Tensor> &,
                                const std::optional<at::Tensor> &,
                                bool,
                                bool);
    using MoeGemmFn = at::Tensor (*)(at::Tensor,
                                     at::Tensor,
                                     at::Tensor,
                                     at::Tensor,
                                     at::Tensor,
                                     std::optional<at::Tensor>,
                                     at::Tensor,
                                     at::Tensor,
                                     at::Tensor,
                                     long,
                                     int,
                                     int);
    using MoeGemmAsmFn = at::Tensor (*)(at::Tensor,
                                        at::Tensor,
                                        at::Tensor,
                                        at::Tensor,
                                        at::Tensor,
                                        std::optional<at::Tensor>,
                                        at::Tensor,
                                        at::Tensor,
                                        at::Tensor,
                                        unsigned int,
                                        int,
                                        int);
    using FuseSiluMulQuantFn = void (*)(at::Tensor &,
                                        at::Tensor &,
                                        at::Tensor &,
                                        std::optional<at::Tensor> &,
                                        int,
                                        int,
                                        std::optional<at::Tensor> &);
    using MoeSumFn = void (*)(at::Tensor &,
                              at::Tensor &,
                              const std::optional<at::Tensor> &,
                              const std::optional<at::Tensor> &,
                              const std::optional<at::Tensor> &,
                              float,
                              int);

    void *handle{nullptr};
    MoeAlignFn moe_align{nullptr};
    MoeGemmFn moe_gemm{nullptr};
    MoeGemmAsmFn moe_gemm_asm{nullptr};
    FuseSiluMulQuantFn fuse_silu_mul_quant{nullptr};
    MoeSumFn moe_sum{nullptr};
};

const LightopSymbols &lightop_symbols() {
    static LightopSymbols symbols;
    static std::once_flag once;
    std::call_once(once, [] {
        symbols.handle = open_lightop_so();
        symbols.moe_align = checked_symbol<LightopSymbols::MoeAlignFn>(
            symbols.handle, "_ZN2at6native20moe_align_block_sizeENS_6TensorEllS1_S1_S1_RKSt8optionalIS1_ES5_S5_bb");
        symbols.moe_gemm = checked_symbol<LightopSymbols::MoeGemmFn>(
            symbols.handle, "_ZN2at6native20moe_gemm_marlin_w8a8ENS_6TensorES1_S1_S1_S1_St8optionalIS1_ES1_S1_S1_lii");
        symbols.moe_gemm_asm = checked_symbol<LightopSymbols::MoeGemmAsmFn>(
            symbols.handle, "_ZN2at6native19moe_marlin_w8a8_asmENS_6TensorES1_S1_S1_S1_St8optionalIS1_ES1_S1_S1_jii");
        symbols.fuse_silu_mul_quant = checked_symbol<LightopSymbols::FuseSiluMulQuantFn>(
            symbols.handle, "_ZN2at6native19fuse_silu_mul_quantERNS_6TensorES2_S2_RSt8optionalIS1_EiiS5_");
        symbols.moe_sum = checked_symbol<LightopSymbols::MoeSumFn>(
            symbols.handle, "_ZN2at6native7moe_sumERNS_6TensorES2_RKSt8optionalIS1_ES6_S6_fi");
    });
    return symbols;
}

std::optional<at::Tensor> to_optional_aten(const std::optional<Tensor> &tensor) {
    if (tensor.has_value()) {
        return infinicore::adaptor::to_aten_tensor(*tensor);
    }
    return std::nullopt;
}
#endif

} // namespace

void deepseek_v4_lightop_moe_align_block_size_(const Tensor &topk_ids,
                                                int num_experts,
                                                int block_size,
                                                Tensor sorted_token_ids,
                                                Tensor expert_ids,
                                                Tensor num_tokens_post_pad,
                                                bool is_fuse_fill) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    guard_device(topk_ids, "deepseek_v4_lightop_moe_align_block_size_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    const auto &symbols = lightop_symbols();
    const std::optional<at::Tensor> none = std::nullopt;
    symbols.moe_align(infinicore::adaptor::to_aten_tensor(topk_ids),
                      static_cast<long>(num_experts),
                      static_cast<long>(block_size),
                      infinicore::adaptor::to_aten_tensor(sorted_token_ids),
                      infinicore::adaptor::to_aten_tensor(expert_ids),
                      infinicore::adaptor::to_aten_tensor(num_tokens_post_pad),
                      none,
                      none,
                      none,
                      false,
                      is_fuse_fill);
#else
    (void)topk_ids; (void)num_experts; (void)block_size; (void)sorted_token_ids; (void)expert_ids;
    (void)num_tokens_post_pad; (void)is_fuse_fill;
    throw std::runtime_error("deepseek_v4_lightop_moe_align_block_size_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_lightop_moe_gemm_marlin_w8a8_(const Tensor &input,
                                                const Tensor &b_qweight,
                                                Tensor output,
                                                const Tensor &a_scale,
                                                const Tensor &b_scale,
                                                const std::optional<Tensor> &topk_weights,
                                                const Tensor &sorted_token_ids,
                                                const Tensor &expert_ids,
                                                const Tensor &num_tokens_post_pad,
                                                int top_k,
                                                int mode,
                                                int delta) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    guard_device(input, "deepseek_v4_lightop_moe_gemm_marlin_w8a8_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    const auto &symbols = lightop_symbols();
    auto input_at = infinicore::adaptor::to_aten_tensor(input);
    auto b_qweight_at = infinicore::adaptor::to_aten_tensor(b_qweight);
    auto output_at = infinicore::adaptor::to_aten_tensor(output);
    auto a_scale_at = infinicore::adaptor::to_aten_tensor(a_scale);
    auto b_scale_at = infinicore::adaptor::to_aten_tensor(b_scale);
    auto topk_weights_at = to_optional_aten(topk_weights);
    auto sorted_token_ids_at = infinicore::adaptor::to_aten_tensor(sorted_token_ids);
    auto expert_ids_at = infinicore::adaptor::to_aten_tensor(expert_ids);
    auto num_tokens_post_pad_at = infinicore::adaptor::to_aten_tensor(num_tokens_post_pad);
    if (mode < 1000) {
        symbols.moe_gemm(input_at,
                         b_qweight_at,
                         output_at,
                         a_scale_at,
                         b_scale_at,
                         topk_weights_at,
                         sorted_token_ids_at,
                         expert_ids_at,
                         num_tokens_post_pad_at,
                         static_cast<long>(top_k),
                         mode,
                         delta);
    } else {
        symbols.moe_gemm_asm(input_at,
                             b_qweight_at,
                             output_at,
                             a_scale_at,
                             b_scale_at,
                             topk_weights_at,
                             sorted_token_ids_at,
                             expert_ids_at,
                             num_tokens_post_pad_at,
                             static_cast<unsigned int>(top_k),
                             mode,
                             delta);
    }
#else
    (void)input; (void)b_qweight; (void)output; (void)a_scale; (void)b_scale; (void)topk_weights;
    (void)sorted_token_ids; (void)expert_ids; (void)num_tokens_post_pad; (void)top_k; (void)mode; (void)delta;
    throw std::runtime_error("deepseek_v4_lightop_moe_gemm_marlin_w8a8_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_lightop_fuse_silu_mul_quant_(Tensor output,
                                               Tensor scales,
                                               const Tensor &input,
                                               const std::optional<Tensor> &num_local_tokens_tensor,
                                               int topk,
                                               int expect_m,
                                               const std::optional<Tensor> &expert_ids) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    guard_device(input, "deepseek_v4_lightop_fuse_silu_mul_quant_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    const auto &symbols = lightop_symbols();
    auto input_at = infinicore::adaptor::to_aten_tensor(input);
    auto output_at = infinicore::adaptor::to_aten_tensor(output);
    auto scales_at = infinicore::adaptor::to_aten_tensor(scales);
    auto num_local_tokens_at = to_optional_aten(num_local_tokens_tensor);
    auto expert_ids_at = to_optional_aten(expert_ids);
    symbols.fuse_silu_mul_quant(input_at,
                                output_at,
                                scales_at,
                                num_local_tokens_at,
                                topk,
                                expect_m,
                                expert_ids_at);
#else
    (void)output; (void)scales; (void)input; (void)num_local_tokens_tensor; (void)topk; (void)expect_m; (void)expert_ids;
    throw std::runtime_error("deepseek_v4_lightop_fuse_silu_mul_quant_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_lightop_moe_sum_(Tensor output,
                                   const Tensor &input,
                                   const std::optional<Tensor> &bias,
                                   const std::optional<Tensor> &expert_mask,
                                   const std::optional<Tensor> &num_local_tokens,
                                   float factor,
                                   int expect_m) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    guard_device(input, "deepseek_v4_lightop_moe_sum_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    const auto &symbols = lightop_symbols();
    auto input_at = infinicore::adaptor::to_aten_tensor(input);
    auto output_at = infinicore::adaptor::to_aten_tensor(output);
    auto bias_at = to_optional_aten(bias);
    auto expert_mask_at = to_optional_aten(expert_mask);
    auto num_local_tokens_at = to_optional_aten(num_local_tokens);
    symbols.moe_sum(input_at,
                    output_at,
                    bias_at,
                    expert_mask_at,
                    num_local_tokens_at,
                    factor,
                    expect_m);
#else
    (void)output; (void)input; (void)bias; (void)expert_mask; (void)num_local_tokens; (void)factor; (void)expect_m;
    throw std::runtime_error("deepseek_v4_lightop_moe_sum_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op

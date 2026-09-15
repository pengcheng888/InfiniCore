#include "infinicore/ops/deepseek_v4/fused_store_flashmla_cache.hpp"

#include "../../aten_utils.hpp"
#include "../fused_store_flashmla_cache_common.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#endif
#endif

#include <stdexcept>

namespace infinicore::op::deepseek_v4 {
namespace {

constexpr double kFp8E4M3Max = 448.0;

#if defined(ENABLE_ATEN) && defined(INFINICORE_DSV4_ACCELERATOR_API)
at::Tensor arange_like_cols(int64_t cols, const at::Tensor &ref) {
    return at::arange(cols, ref.options().dtype(at::kLong));
}
#endif

} // namespace

void fused_store_flashmla_cache_aten_(const Tensor &input,
                                      Tensor cache,
                                      const Tensor &indices,
                                      int page_size) {
#if defined(ENABLE_ATEN) && defined(INFINICORE_DSV4_ACCELERATOR_API)
    detail::prepare_aten_call(input, "fused_store_flashmla_cache_aten_");
    if (input->dtype() == DataType::F32) {
        throw std::runtime_error("fused_store_flashmla_cache_ expects bf16/fp16 input because RoPE bytes are copied verbatim.");
    }
    fused_store_flashmla_cache_detail::check_shapes(input, cache, indices, page_size);

    auto input_at = infinicore::adaptor::to_aten_tensor(input).contiguous();
    auto cache_at = infinicore::adaptor::to_aten_tensor(cache);
    auto indices_at = infinicore::adaptor::to_aten_tensor(indices).to(at::kLong);

    auto valid_mask = indices_at >= 0;
    auto valid_rows = at::nonzero(valid_mask).reshape({-1});
    if (valid_rows.numel() == 0) {
        return;
    }
    if (valid_rows.numel() != indices_at.numel()) {
        input_at = input_at.index_select(0, valid_rows);
        indices_at = indices_at.index_select(0, valid_rows);
    }

    const int64_t num_tokens = input_at.size(0);
    const int64_t page_size_i64 = static_cast<int64_t>(page_size);
    const int64_t page_bytes = fused_store_flashmla_cache_detail::page_bytes(page_size);
    const auto page = at::floor_divide(indices_at, page_size_i64);
    const auto offset = at::remainder(indices_at, page_size_i64);

    auto no_pe = input_at.slice(1, 0, fused_store_flashmla_cache_detail::kNopeDim)
                     .reshape({num_tokens, 7, 64})
                     .to(at::kFloat);
    auto scale_raw = at::clamp_min(at::amax(at::abs(no_pe), {-1}, true), 1.0e-4) / kFp8E4M3Max;
    auto scale_exp = at::clamp(at::ceil(at::log2(scale_raw)).to(at::kInt) + 127, 0, 255).to(at::kByte);
    auto scale = at::pow(at::scalar_tensor(2.0, scale_raw.options()), scale_exp.to(at::kFloat) - 127.0);
    auto quant_fp8 = at::clamp(no_pe / scale, -kFp8E4M3Max, kFp8E4M3Max)
                         .to(at::ScalarType::Float8_e4m3fn)
                         .view(at::kByte)
                         .reshape({num_tokens, fused_store_flashmla_cache_detail::kNopeDim});

    auto rope_bytes = input_at.slice(1,
                                     fused_store_flashmla_cache_detail::kNopeDim,
                                     fused_store_flashmla_cache_detail::kInputDim)
                          .contiguous()
                          .view(at::kByte)
                          .reshape({num_tokens, fused_store_flashmla_cache_detail::kRopeDim * 2});

    auto flat_cache = cache_at.reshape({cache_at.size(0) * cache_at.size(1)});
    auto token_base = page * page_bytes + offset * fused_store_flashmla_cache_detail::kValueBytesPerToken;
    auto nope_cols = arange_like_cols(fused_store_flashmla_cache_detail::kNopeDim, indices_at);
    auto rope_cols = arange_like_cols(fused_store_flashmla_cache_detail::kRopeDim * 2, indices_at);
    auto scale_cols = arange_like_cols(7, indices_at);

    auto nope_pos = (token_base.unsqueeze(1) + nope_cols.unsqueeze(0)).reshape({-1});
    auto rope_pos = (token_base.unsqueeze(1) + fused_store_flashmla_cache_detail::kNopeDim
                     + rope_cols.unsqueeze(0))
                        .reshape({-1});
    auto scale_pos = (page * page_bytes
                      + fused_store_flashmla_cache_detail::kValueBytesPerToken * page_size_i64
                      + offset * fused_store_flashmla_cache_detail::kScaleBytesPerToken)
                         .unsqueeze(1)
                   + scale_cols.unsqueeze(0);

    flat_cache.index_put_({nope_pos}, quant_fp8.reshape({-1}));
    flat_cache.index_put_({rope_pos}, rope_bytes.reshape({-1}));
    flat_cache.index_put_({scale_pos.reshape({-1})}, scale_exp.reshape({num_tokens, 7}).reshape({-1}));
#else
    (void)input;
    (void)cache;
    (void)indices;
    (void)page_size;
    throw std::runtime_error("fused_store_flashmla_cache_aten_ requires an ATen-enabled HYGON/NVIDIA/METAX/ILUVATAR build.");
#endif
}

} // namespace infinicore::op::deepseek_v4

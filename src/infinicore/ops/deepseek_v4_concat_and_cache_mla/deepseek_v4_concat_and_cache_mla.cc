#include "infinicore/ops/deepseek_v4_concat_and_cache_mla.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/graph/graph.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/core/dispatch/Dispatcher.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#elif defined(ENABLE_NVIDIA_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace infinicore::op {
namespace {

void run_deepseek_v4_concat_and_cache_mla(const Tensor &kv_c,
                                          const Tensor &k_pe,
                                          Tensor kv_cache,
                                          const Tensor &slot_mapping,
                                          const std::string &kv_cache_dtype,
                                          const Tensor &scale) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
    if (kv_c->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("deepseek_v4_concat_and_cache_mla_ expects HYGON tensors in this build.");
    }
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    if (kv_c->device().getType() != Device::Type::NVIDIA) {
        throw std::runtime_error("deepseek_v4_concat_and_cache_mla_ expects NVIDIA tensors in this build.");
    }
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto kv_c_at = infinicore::adaptor::to_aten_tensor(kv_c);
    auto k_pe_at = infinicore::adaptor::to_aten_tensor(k_pe);
    auto kv_cache_at = infinicore::adaptor::to_aten_tensor(kv_cache);
    auto slot_mapping_at = infinicore::adaptor::to_aten_tensor(slot_mapping);
    auto scale_at = infinicore::adaptor::to_aten_tensor(scale);

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("_C_cache_ops::concat_and_cache_mla", "")
                         .typed<void(at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, const std::string &, at::Tensor &)>();
    op.call(kv_c_at, k_pe_at, kv_cache_at, slot_mapping_at, kv_cache_dtype, scale_at);
#else
    (void)kv_c;
    (void)k_pe;
    (void)kv_cache;
    (void)slot_mapping;
    (void)kv_cache_dtype;
    (void)scale;
    throw std::runtime_error("deepseek_v4_concat_and_cache_mla_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

class DeepseekV4ConcatAndCacheMlaGraphOp final : public graph::GraphOperator {
public:
    DeepseekV4ConcatAndCacheMlaGraphOp(const Tensor &kv_c,
                                       const Tensor &k_pe,
                                       Tensor kv_cache,
                                       const Tensor &slot_mapping,
                                       std::string kv_cache_dtype,
                                       const Tensor &scale)
        : kv_c_(kv_c),
          k_pe_(k_pe),
          kv_cache_(std::move(kv_cache)),
          slot_mapping_(slot_mapping),
          kv_cache_dtype_(std::move(kv_cache_dtype)),
          scale_(scale) {}

    void run() const override {
        run_deepseek_v4_concat_and_cache_mla(
            kv_c_,
            k_pe_,
            kv_cache_,
            slot_mapping_,
            kv_cache_dtype_,
            scale_);
    }

    bool is_device_graph_capture_safe() const override {
        return false;
    }

private:
    Tensor kv_c_;
    Tensor k_pe_;
    Tensor kv_cache_;
    Tensor slot_mapping_;
    std::string kv_cache_dtype_;
    Tensor scale_;
};

} // namespace

void deepseek_v4_concat_and_cache_mla_(const Tensor &kv_c,
                                       const Tensor &k_pe,
                                       Tensor kv_cache,
                                       const Tensor &slot_mapping,
                                       const std::string &kv_cache_dtype,
                                       const Tensor &scale) {
    auto op = std::make_shared<DeepseekV4ConcatAndCacheMlaGraphOp>(
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        kv_cache_dtype,
        scale);
    if (context::isGraphRecording()) {
        context::addGraphOperator(op);
    } else {
        op->run();
    }
}

} // namespace infinicore::op

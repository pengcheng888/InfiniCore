#include "infinicore/ops/deepseek_v4/fused_store_flashmla_cache.hpp"

#include "fused_store_flashmla_cache_common.hpp"
#include "kernel/fused_store_flashmla_cache_kernel.hpp"

#include "../graph_deferred.hpp"
#include "../platform.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/graph/graph.hpp"

#include "../../../utils.hpp"

#include <stdexcept>

namespace infinicore::op::deepseek_v4 {

INFINICORE_GRAPH_OP_CLASS(FusedStoreFlashMlaCacheKernel, const Tensor &, Tensor, const Tensor &, int);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(FusedStoreFlashMlaCacheKernel);

namespace {

int input_scalar_type(const Tensor &input) {
    return input->dtype() == DataType::BF16 ? fused_store_flashmla_cache::BF16
                                            : fused_store_flashmla_cache::F16;
}

} // namespace

void fused_store_flashmla_cache_(const Tensor &input,
                                 Tensor cache,
                                 const Tensor &indices,
                                 int page_size) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    fused_store_flashmla_cache_kernel_(input, cache, indices, page_size);
#elif defined(ENABLE_ATEN) && (defined(ENABLE_METAX_API) || defined(ENABLE_ILUVATAR_API))
    auto input_graph = graph::GraphTensor(input);
    auto cache_graph = graph::GraphTensor(cache);
    auto indices_graph = graph::GraphTensor(indices);
    detail::record_or_run_host_graph_op(
        [input_graph, cache_graph, indices_graph, page_size]() mutable {
            fused_store_flashmla_cache_aten_(
                input_graph, cache_graph, indices_graph, page_size);
        });
#else
    (void)input;
    (void)cache;
    (void)indices;
    (void)page_size;
    throw std::runtime_error(
        "fused_store_flashmla_cache_ requires an ATen-enabled "
        "HYGON/NVIDIA/METAX/ILUVATAR build.");
#endif
}

FusedStoreFlashMlaCacheKernel::FusedStoreFlashMlaCacheKernel(const Tensor &input,
                                                             Tensor cache,
                                                             const Tensor &indices,
                                                             int page_size) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(input, cache, indices);
    INFINICORE_GRAPH_OP_DISPATCH(input->device().getType(), input, cache, indices, page_size);
}

void FusedStoreFlashMlaCacheKernel::execute(const Tensor &input,
                                            Tensor cache,
                                            const Tensor &indices,
                                            int page_size) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(FusedStoreFlashMlaCacheKernel, input, cache, indices, page_size);
}

namespace fused_store_flashmla_cache_impl {

struct PlannedMeta {
    graph::GraphTensor input;
    graph::GraphTensor cache;
    graph::GraphTensor indices;
    int input_dtype;
    bool indices_i64;
    int64_t tokens;
    int page_size;
    int64_t page_bytes;
};

void *plan(const Tensor &input, Tensor cache, const Tensor &indices, int page_size) {
    fused_store_flashmla_cache_detail::check_shapes(input, cache, indices, page_size);
    detail::check_build_device(input, "fused_store_flashmla_cache_kernel_");
    return new PlannedMeta{graph::GraphTensor(input),
                           graph::GraphTensor(cache),
                           graph::GraphTensor(indices),
                           input_scalar_type(input),
                           indices->dtype() == DataType::I64,
                           static_cast<int64_t>(input->size(0)),
                           page_size,
                           fused_store_flashmla_cache_detail::page_bytes(page_size)};
}

void run(void *planned_meta) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    fused_store_flashmla_cache::launch_fused_store_flashmla_cache(
        planned->input->data(),
        planned->input_dtype,
        reinterpret_cast<uint8_t *>(planned->cache->data()),
        planned->indices->data(),
        planned->indices_i64,
        planned->tokens,
        planned->page_size,
        planned->page_bytes,
        context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("fused_store_flashmla_cache_kernel_ requires a HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace fused_store_flashmla_cache_impl

namespace fused_store_flashmla_cache_register {
INFINICORE_DSV4_NATIVE_GRAPH_OP_REGISTER_BUILD_DEVICE(
    FusedStoreFlashMlaCacheKernel,
    &fused_store_flashmla_cache_impl::plan,
    &fused_store_flashmla_cache_impl::run,
    &fused_store_flashmla_cache_impl::cleanup);
} // namespace fused_store_flashmla_cache_register

void fused_store_flashmla_cache_kernel_(const Tensor &input,
                                        Tensor cache,
                                        const Tensor &indices,
                                        int page_size) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    fused_store_flashmla_cache_detail::check_shapes(input, cache, indices, page_size);
    detail::check_build_device(input, "fused_store_flashmla_cache_kernel_");
    FusedStoreFlashMlaCacheKernel::execute(input, cache, indices, page_size);
#else
    (void)input;
    (void)cache;
    (void)indices;
    (void)page_size;
    throw std::runtime_error("fused_store_flashmla_cache_kernel_ requires a HYGON/NVIDIA build.");
#endif
}

} // namespace infinicore::op::deepseek_v4

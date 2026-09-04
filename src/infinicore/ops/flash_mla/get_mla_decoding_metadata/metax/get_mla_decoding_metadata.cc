#include "get_mla_decoding_metadata.hpp"

#if defined(ENABLE_ATEN) && defined(ENABLE_METAX_API)

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdlib>
#include <dlfcn.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::op::flash_mla::get_mla_decoding_metadata_metax {
namespace {

using FlashMlaMetadataFn = std::vector<at::Tensor> (*)(
    at::Tensor &,
    int,
    int,
    std::optional<int>,
    bool,
    std::optional<int>);

constexpr const char *kFlashMlaMetadataSymbol = "_Z25get_mla_decoding_metadataRN2at6TensorEiiSt8optionalIiEbS3_";
constexpr const char *kDefaultFlashMlaSoPath = "/opt/conda/lib/python3.12/site-packages/flash_mla_cuda.cpython-312-x86_64-linux-gnu.so";

void check_device(const Tensor &tensor, const char *op_name) {
    if (!tensor || tensor->device().getType() != Device::Type::METAX) {
        throw std::runtime_error(std::string(op_name) + " expects METAX tensors.");
    }
}

DataType from_at_scalar_type(at::ScalarType dtype) {
    switch (dtype) {
    case at::kFloat:
        return DataType::F32;
    case at::kHalf:
        return DataType::F16;
    case at::kBFloat16:
        return DataType::BF16;
    case at::kChar:
        return DataType::I8;
    case at::kInt:
        return DataType::I32;
    case at::kLong:
        return DataType::I64;
    case at::kByte:
        return DataType::U8;
    case at::kFloat8_e4m3fn:
        return DataType::F8;
    default:
        throw std::runtime_error("get_mla_decoding_metadata_impl: unsupported FlashMLA return dtype.");
    }
}

Device from_at_device(const at::Device &device) {
    if (device.is_cpu()) {
        return Device(Device::Type::CPU, 0);
    }
    if (!device.is_cuda()) {
        throw std::runtime_error("get_mla_decoding_metadata_impl: unsupported FlashMLA return device.");
    }
    return Device(Device::Type::METAX, static_cast<Device::Index>(device.index()));
}

Shape shape_from_at_tensor(const at::Tensor &tensor) {
    Shape shape;
    shape.reserve(static_cast<size_t>(tensor.dim()));
    for (const auto dim : tensor.sizes()) {
        shape.push_back(static_cast<size_t>(dim));
    }
    return shape;
}

void copy_flashmla_return_tensor_exact(Tensor &dst, at::Tensor src, const char *name) {
    if (!src.defined()) {
        throw std::runtime_error(std::string("get_mla_decoding_metadata_impl: FlashMLA returned undefined ") + name + ".");
    }
    src = src.contiguous();
    const auto expected_shape = shape_from_at_tensor(src);
    const auto expected_dtype = from_at_scalar_type(src.scalar_type());
    const auto expected_device = from_at_device(src.device());
    if (!dst) {
        dst = Tensor::empty(expected_shape, expected_dtype, expected_device);
    }
    if (dst->shape() != expected_shape) {
        throw std::runtime_error(std::string("get_mla_decoding_metadata_impl: ") + name + " shape mismatch.");
    }
    if (dst->dtype() != expected_dtype) {
        throw std::runtime_error(std::string("get_mla_decoding_metadata_impl: ") + name + " dtype mismatch.");
    }
    if (dst->device() != expected_device) {
        throw std::runtime_error(std::string("get_mla_decoding_metadata_impl: ") + name + " device mismatch.");
    }
    if (!dst->is_contiguous()) {
        throw std::runtime_error(std::string("get_mla_decoding_metadata_impl: ") + name + " must be contiguous.");
    }
    auto dst_at = infinicore::adaptor::to_aten_tensor(dst);
    dst_at.copy_(src);
}

void *resolve_flashmla_so_symbol(const char *symbol, const char *op_name) {
    if (void *fn = dlsym(RTLD_DEFAULT, symbol)) {
        return fn;
    }

    const char *so_path = std::getenv("INFINICORE_METAX_FLASH_MLA_SO");
    if (so_path == nullptr || so_path[0] == '\0') {
        so_path = std::getenv("INFINICORE_DSV4_FLASHMLA_SO");
    }
    if (so_path == nullptr || so_path[0] == '\0') {
        so_path = kDefaultFlashMlaSoPath;
    }

    void *handle = dlopen(so_path, RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        const char *err = dlerror();
        throw std::runtime_error(std::string(op_name) + " requires flash_mla_cuda; failed to dlopen " + so_path + (err == nullptr ? "" : std::string(": ") + err));
    }
    if (void *fn = dlsym(handle, symbol)) {
        return fn;
    }
    throw std::runtime_error(std::string(op_name) + " missing flash_mla_cuda symbol: " + symbol);
}

FlashMlaMetadataFn flashmla_metadata_fn(const char *op_name) {
    static auto fn = reinterpret_cast<FlashMlaMetadataFn>(
        resolve_flashmla_so_symbol(kFlashMlaMetadataSymbol, op_name));
    return fn;
}

std::optional<int> to_optional_int(std::optional<int64_t> value, const char *name) {
    if (!value.has_value()) {
        return std::nullopt;
    }
    if (value.value() < static_cast<int64_t>(std::numeric_limits<int>::min())
        || value.value() > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("get_mla_decoding_metadata_impl: ") + name + " is out of int range.");
    }
    return static_cast<int>(value.value());
}

} // namespace

void get_mla_decoding_metadata_impl(
    Tensor &tile_scheduler_metadata,
    Tensor &num_splits,
    const Tensor &cache_seqlens,
    int64_t num_q_tokens_per_head_k,
    int64_t num_heads_k,
    std::optional<int64_t> num_heads_q,
    bool is_fp8_kvcache,
    std::optional<int64_t> topk) {
    constexpr const char *op_name = "get_mla_decoding_metadata_impl";
    check_device(cache_seqlens, op_name);
    if (cache_seqlens->dtype() != DataType::I32 || !cache_seqlens->is_contiguous()) {
        throw std::runtime_error("get_mla_decoding_metadata_impl expects contiguous int32 cache_seqlens.");
    }
    if (num_q_tokens_per_head_k <= 0 || num_heads_k <= 0) {
        throw std::runtime_error("get_mla_decoding_metadata_impl expects positive head counts.");
    }

    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
    auto cache_seqlens_at = infinicore::adaptor::to_aten_tensor(cache_seqlens);
    auto metadata = flashmla_metadata_fn(op_name)(cache_seqlens_at,
                                                  static_cast<int>(num_q_tokens_per_head_k),
                                                  static_cast<int>(num_heads_k),
                                                  to_optional_int(num_heads_q, "num_heads_q"),
                                                  is_fp8_kvcache,
                                                  to_optional_int(topk, "topk"));
    if (metadata.size() != 2) {
        throw std::runtime_error("get_mla_decoding_metadata_impl: flash_mla_cuda.get_mla_metadata must return two tensors.");
    }

    copy_flashmla_return_tensor_exact(tile_scheduler_metadata, metadata[0], "tile_scheduler_metadata");
    copy_flashmla_return_tensor_exact(num_splits, metadata[1], "num_splits");
}

namespace {

struct PlannedMeta {
    graph::GraphTensor tile_scheduler_metadata;
    graph::GraphTensor num_splits;
    graph::GraphTensor cache_seqlens;
    int64_t num_q_tokens_per_head_k;
    int64_t num_heads_k;
    std::optional<int64_t> num_heads_q;
    bool is_fp8_kvcache;
    std::optional<int64_t> topk;
};

void *plan(Tensor tile_scheduler_metadata,
           Tensor num_splits,
           const Tensor &cache_seqlens,
           int64_t num_q_tokens_per_head_k,
           int64_t num_heads_k,
           std::optional<int64_t> num_heads_q,
           bool is_fp8_kvcache,
           std::optional<int64_t> topk) {
    return new PlannedMeta{graph::GraphTensor(tile_scheduler_metadata),
                           graph::GraphTensor(num_splits),
                           graph::GraphTensor(cache_seqlens),
                           num_q_tokens_per_head_k,
                           num_heads_k,
                           num_heads_q,
                           is_fp8_kvcache,
                           topk};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    get_mla_decoding_metadata_impl(planned->tile_scheduler_metadata,
                                   planned->num_splits,
                                   planned->cache_seqlens,
                                   planned->num_q_tokens_per_head_k,
                                   planned->num_heads_k,
                                   planned->num_heads_q,
                                   planned->is_fp8_kvcache,
                                   planned->topk);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace

static bool registered = []() {
    ::infinicore::op::flash_mla::GetMlaDecodingMetadata::plan_dispatcher().registerDevice(Device::Type::METAX, &plan);
    ::infinicore::op::flash_mla::GetMlaDecodingMetadata::run_dispatcher().registerDevice(Device::Type::METAX, &run);
    ::infinicore::op::flash_mla::GetMlaDecodingMetadata::cleanup_dispatcher().registerDevice(Device::Type::METAX, &cleanup);
    ::infinicore::op::flash_mla::get_mla_decoding_metadata_impl_dispatcher().registerDevice(Device::Type::METAX, &get_mla_decoding_metadata_impl);
    return true;
}();

} // namespace infinicore::op::flash_mla::get_mla_decoding_metadata_metax

#endif

#include "infinicore/ops/deepseek_v4_flashmla_compute.hpp"

#include "deepseek_v4_flashmla_compute_kernel.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <elf.h>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#elif defined(ENABLE_NVIDIA_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4FlashMlaSparseAttentionWithMetadata);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4CompressFusedNormRopeKernel);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4C4CompressStatefulKernel);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4C128CompressStatefulKernel);

namespace {

constexpr int64_t kDsv4FlashMlaQDim = 512;
constexpr int64_t kDsv4FlashMlaValueBytesPerToken = 576;
constexpr int64_t kDsv4FlashMlaScaleBytesPerToken = 8;
constexpr int64_t kDsv4FlashMlaBytesPerToken = kDsv4FlashMlaValueBytesPerToken + kDsv4FlashMlaScaleBytesPerToken;

int64_t div_ceil_i64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

int64_t dsv4_flashmla_page_bytes(int page_size) {
    const auto bytes = kDsv4FlashMlaBytesPerToken * static_cast<int64_t>(page_size);
    return div_ceil_i64(bytes, kDsv4FlashMlaValueBytesPerToken) * kDsv4FlashMlaValueBytesPerToken;
}

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

void check_sparse_attention_shapes(const Tensor &q,
                                   const Tensor &raw_cache,
                                   const Tensor &indices,
                                   const Tensor &topk_lengths,
                                   const Tensor &output,
                                   int page_size,
                                   int head_dim_v) {
    if (q->ndim() != 3 && q->ndim() != 4) {
        throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ expects q [tokens, heads, 512] or [batch, seq, heads, 512].");
    }
    if (q->size(q->ndim() - 1) != static_cast<size_t>(kDsv4FlashMlaQDim)) {
        throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ expects q head dim 512.");
    }
    if (raw_cache->ndim() != 2) {
        throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ expects raw cache [blocks, page_bytes].");
    }
    if (indices->ndim() != 2 && indices->ndim() != 3) {
        throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ expects indices [tokens, topk] or [batch, seq, topk].");
    }
    if (topk_lengths->ndim() != 1 && topk_lengths->ndim() != 2) {
        throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ expects topk_lengths [tokens] or [batch, seq].");
    }
    if (output->ndim() != q->ndim()) {
        throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ expects output rank to match q rank.");
    }
    if (q->dtype() != DataType::BF16) {
        throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ FlashMLA sparse decode expects bf16 q.");
    }
    if (output->dtype() != DataType::BF16) {
        throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ FlashMLA sparse decode expects bf16 output.");
    }
    if (raw_cache->dtype() != DataType::U8) {
        throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ raw cache dtype must be uint8.");
    }
    if ((indices->dtype() != DataType::I32 && indices->dtype() != DataType::I64) ||
        (topk_lengths->dtype() != DataType::I32 && topk_lengths->dtype() != DataType::I64)) {
        throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ metadata dtype must be int32 or int64.");
    }
    if (page_size <= 0 || (page_size & (page_size - 1)) != 0) {
        throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ page_size must be a positive power of two.");
    }
    if (head_dim_v <= 0 || head_dim_v > static_cast<int>(kDsv4FlashMlaQDim)) {
        throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ head_dim_v must be in (0, 512].");
    }
    const auto expected_page_bytes = static_cast<size_t>(dsv4_flashmla_page_bytes(page_size));
    if (raw_cache->size(1) != expected_page_bytes) {
        throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ raw cache page_bytes mismatch.");
    }
    for (int i = 0; i < q->ndim() - 2; ++i) {
        if (output->size(i) != q->size(i)) {
            throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ output leading shape mismatch.");
        }
    }
    if (output->size(output->ndim() - 2) != q->size(q->ndim() - 2) ||
        output->size(output->ndim() - 1) != static_cast<size_t>(head_dim_v)) {
        throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ output head/head_dim shape mismatch.");
    }
}

void check_compress_fused_norm_rope_shapes(const Tensor &input,
                                           const Tensor &norm_weight,
                                           const Tensor &freqs_cis,
                                           const Tensor &positions) {
    if (input->ndim() != 2 || input->size(1) < 64) {
        throw std::runtime_error("deepseek_v4_compress_fused_norm_rope_ expects input [tokens, dim>=64].");
    }
    if (input->dtype() != DataType::BF16) {
        throw std::runtime_error("deepseek_v4_compress_fused_norm_rope_ expects bf16 input.");
    }
    if (norm_weight->numel() != input->size(1)) {
        throw std::runtime_error("deepseek_v4_compress_fused_norm_rope_ norm_weight size mismatch.");
    }
    if (freqs_cis->ndim() != 2 || freqs_cis->size(1) != 64 || freqs_cis->dtype() != DataType::F32) {
        throw std::runtime_error("deepseek_v4_compress_fused_norm_rope_ expects freqs_cis [max_pos, 64] float32.");
    }
    if (positions->ndim() != 1 || positions->numel() != input->size(0) ||
        (positions->dtype() != DataType::I32 && positions->dtype() != DataType::I64)) {
        throw std::runtime_error("deepseek_v4_compress_fused_norm_rope_ expects positions [tokens] int32/int64.");
    }
}


int dsv4_scalar_type_for_kernel(const Tensor &tensor, const char *op_name) {
    if (tensor->dtype() == DataType::BF16) {
        return deepseek_v4_flashmla_compute_kernel::kDsv4BF16;
    }
    if (tensor->dtype() == DataType::F16) {
        return deepseek_v4_flashmla_compute_kernel::kDsv4F16;
    }
    if (tensor->dtype() == DataType::F32) {
        return deepseek_v4_flashmla_compute_kernel::kDsv4F32;
    }
    throw std::runtime_error(std::string(op_name) + " supports bf16/fp16/fp32 tensors only.");
}

void check_common_accel_tensor(const Tensor &tensor, const char *op_name) {
    check_hygon_or_nvidia_tensor(tensor, op_name);
    if (!tensor->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous tensors.");
    }
}

int c4_ape_layout(const Tensor &ape, int64_t head_dim, const char *op_name) {
    if (ape->ndim() != 2) {
        throw std::runtime_error(std::string(op_name) + " expects ape rank 2.");
    }
    if (ape->size(0) == 8 && ape->size(1) == static_cast<size_t>(head_dim)) {
        return 0;
    }
    if (ape->size(0) == 4 && ape->size(1) == static_cast<size_t>(2 * head_dim)) {
        return 1;
    }
    throw std::runtime_error(std::string(op_name) + " expects ape [8, head_dim] or [4, 2 * head_dim].");
}

#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
void apply_rope_2d_last64_aten_(at::Tensor rope, const at::Tensor &freqs_cis, const at::Tensor &positions) {
    constexpr int64_t rope_dim = 64;
    const int64_t tokens = rope.size(0);
    if (tokens == 0) {
        return;
    }
    auto pos_long = positions.reshape({tokens}).to(at::kLong);
    auto selected = freqs_cis.index_select(0, pos_long).to(at::kFloat).reshape({tokens, rope_dim / 2, 2});
    auto freq_real = selected.select(-1, 0);
    auto freq_imag = selected.select(-1, 1);

    auto rope_pair = rope.to(at::kFloat).reshape({tokens, rope_dim / 2, 2});
    auto x_real = rope_pair.select(-1, 0);
    auto x_imag = rope_pair.select(-1, 1);
    auto out_real = x_real * freq_real - x_imag * freq_imag;
    auto out_imag = x_real * freq_imag + x_imag * freq_real;
    auto result = at::stack({out_real, out_imag}, -1).reshape(rope.sizes()).to(rope.scalar_type());
    rope.copy_(result);
}
#endif

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
constexpr const char *kFlashMlaSparseDecodeInterfaceSymbol =
    "_ZL28sparse_attn_decode_interfaceRKN2at6TensorES2_S2_RKSt8optionalIS0_ES6_RS4_S7_S6_S6_S6_if";
constexpr const char *kDefaultFlashMlaSoPath =
    "/usr/local/lib/python3.10/dist-packages/flash_mla/cuda.cpython-310-x86_64-linux-gnu.so";
constexpr const char *kFlashMlaAnchorSymbol = "PyInit_cuda";

using FlashMlaSparseDecodeFn = std::tuple<at::Tensor,
                                          at::Tensor,
                                          std::optional<at::Tensor>,
                                          std::optional<at::Tensor>> (*)(const at::Tensor &,
                                                                        const at::Tensor &,
                                                                        const at::Tensor &,
                                                                        const std::optional<at::Tensor> &,
                                                                        const std::optional<at::Tensor> &,
                                                                        std::optional<at::Tensor> &,
                                                                        std::optional<at::Tensor> &,
                                                                        const std::optional<at::Tensor> &,
                                                                        const std::optional<at::Tensor> &,
                                                                        const std::optional<at::Tensor> &,
                                                                        int,
                                                                        float);

std::optional<at::Tensor> optional_i32_aten_tensor(std::optional<Tensor> tensor,
                                                   const char *name,
                                                   const char *op_name) {
    if (!tensor.has_value() || !tensor.value()) {
        return std::nullopt;
    }
    check_hygon_or_nvidia_tensor(tensor.value(), op_name);
    if (tensor.value()->dtype() != DataType::I32) {
        throw std::runtime_error(std::string(op_name) + " expects " + name + " dtype int32.");
    }
    if (!tensor.value()->is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " expects contiguous " + name + ".");
    }
    return infinicore::adaptor::to_aten_tensor(tensor.value());
}

Tensor wrap_i32_aten_tensor_to_infinicore(const std::optional<at::Tensor> &tensor,
                                          const Device &device,
                                          const char *name,
                                          const char *op_name) {
    (void)device;
    if (!tensor.has_value() || !tensor.value().defined()) {
        return Tensor{};
    }
    auto source = tensor.value();
    if (source.scalar_type() != at::kInt) {
        throw std::runtime_error(std::string(op_name) + " returned non-int32 " + name + ".");
    }
    return infinicore::adaptor::from_aten_tensor(source.contiguous());
}

DeepseekV4FlashMLASparseAttentionSchedule build_flashmla_sparse_schedule_result(
    std::optional<Tensor> input_tile_scheduler_metadata,
    std::optional<Tensor> input_num_splits,
    const std::optional<at::Tensor> &new_tile_scheduler_metadata,
    const std::optional<at::Tensor> &new_num_splits,
    const Device &device,
    const char *op_name) {
    return {
        input_tile_scheduler_metadata.has_value() && input_tile_scheduler_metadata.value()
            ? input_tile_scheduler_metadata.value()
            : wrap_i32_aten_tensor_to_infinicore(new_tile_scheduler_metadata, device, "tile_scheduler_metadata", op_name),
        input_num_splits.has_value() && input_num_splits.value()
            ? input_num_splits.value()
            : wrap_i32_aten_tensor_to_infinicore(new_num_splits, device, "num_splits", op_name)};
}

uintptr_t find_elf_symbol_value(const std::string &path, const char *symbol) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open flash_mla SO for local symbol lookup: " + path);
    }
    std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (data.size() < sizeof(Elf64_Ehdr)) {
        throw std::runtime_error("flash_mla SO is too small to be a valid ELF file: " + path);
    }

    const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(data.data());
    if (std::memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0 || ehdr->e_ident[EI_CLASS] != ELFCLASS64) {
        throw std::runtime_error("flash_mla SO is not an ELF64 shared object: " + path);
    }
    const auto sh_end = ehdr->e_shoff + static_cast<uint64_t>(ehdr->e_shnum) * sizeof(Elf64_Shdr);
    if (ehdr->e_shoff >= data.size() || sh_end > data.size()) {
        throw std::runtime_error("flash_mla SO has invalid section table: " + path);
    }
    const auto *sections = reinterpret_cast<const Elf64_Shdr *>(data.data() + ehdr->e_shoff);

    for (int i = 0; i < ehdr->e_shnum; ++i) {
        const auto &symtab = sections[i];
        if (symtab.sh_type != SHT_SYMTAB && symtab.sh_type != SHT_DYNSYM) {
            continue;
        }
        if (symtab.sh_link >= ehdr->e_shnum) {
            continue;
        }
        const auto &strtab = sections[symtab.sh_link];
        if (symtab.sh_offset + symtab.sh_size > data.size() ||
            strtab.sh_offset + strtab.sh_size > data.size() ||
            symtab.sh_entsize != sizeof(Elf64_Sym)) {
            continue;
        }

        const auto *symbols = reinterpret_cast<const Elf64_Sym *>(data.data() + symtab.sh_offset);
        const auto *names = data.data() + strtab.sh_offset;
        const auto count = symtab.sh_size / sizeof(Elf64_Sym);
        for (uint64_t j = 0; j < count; ++j) {
            if (symbols[j].st_name >= strtab.sh_size) {
                continue;
            }
            const char *name = names + symbols[j].st_name;
            if (std::strcmp(name, symbol) == 0) {
                return static_cast<uintptr_t>(symbols[j].st_value);
            }
        }
    }

    throw std::runtime_error(std::string("missing local flash_mla symbol in ELF symtab: ") + symbol);
}

void *resolve_flashmla_sparse_decode(const char *op_name) {
    if (void *fn = dlsym(RTLD_DEFAULT, kFlashMlaSparseDecodeInterfaceSymbol)) {
        return fn;
    }

    const char *so_path = std::getenv("INFINICORE_DSV4_FLASHMLA_SO");
    if (so_path == nullptr || so_path[0] == '\0') {
        so_path = kDefaultFlashMlaSoPath;
    }

    void *handle = dlopen(so_path, RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        const char *err = dlerror();
        throw std::runtime_error(std::string(op_name) +
                                 " requires flash_mla.cuda.so; failed to dlopen " + so_path +
                                 (err == nullptr ? "" : std::string(": ") + err));
    }
    if (void *fn = dlsym(handle, kFlashMlaSparseDecodeInterfaceSymbol)) {
        return fn;
    }

    void *anchor = dlsym(handle, kFlashMlaAnchorSymbol);
    if (anchor == nullptr) {
        throw std::runtime_error(std::string(op_name) +
                                 " requires flash_mla.cuda.so anchor symbol: " + kFlashMlaAnchorSymbol);
    }

    Dl_info info;
    if (dladdr(anchor, &info) == 0 || info.dli_fbase == nullptr) {
        throw std::runtime_error(std::string(op_name) + " failed to resolve flash_mla SO load base.");
    }
    const std::string loaded_path = info.dli_fname == nullptr ? std::string(so_path) : std::string(info.dli_fname);
    const uintptr_t symbol_value = find_elf_symbol_value(loaded_path, kFlashMlaSparseDecodeInterfaceSymbol);
    return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(info.dli_fbase) + symbol_value);
}

#endif

} // namespace



DeepseekV4CompressFusedNormRopeKernel::DeepseekV4CompressFusedNormRopeKernel(
    Tensor input,
    const Tensor &norm_weight,
    float epsilon,
    const Tensor &freqs_cis,
    const Tensor &positions) {
    INFINICORE_GRAPH_OP_DISPATCH(input->device().getType(), input, norm_weight, epsilon, freqs_cis, positions);
}

void DeepseekV4CompressFusedNormRopeKernel::execute(Tensor input,
                                                    const Tensor &norm_weight,
                                                    float epsilon,
                                                    const Tensor &freqs_cis,
                                                    const Tensor &positions) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4CompressFusedNormRopeKernel,
                                      input,
                                      norm_weight,
                                      epsilon,
                                      freqs_cis,
                                      positions);
}

namespace deepseek_v4_compress_fused_norm_rope_graph_impl {

struct PlannedMeta {
    graph::GraphTensor input;
    graph::GraphTensor norm_weight;
    graph::GraphTensor freqs_cis;
    graph::GraphTensor positions;
    int input_dtype;
    int norm_weight_dtype;
    bool positions_i64;
    int64_t tokens;
    int64_t dim;
    float epsilon;
};

void *plan(Tensor input,
           const Tensor &norm_weight,
           float epsilon,
           const Tensor &freqs_cis,
           const Tensor &positions) {
    check_compress_fused_norm_rope_shapes(input, norm_weight, freqs_cis, positions);
    check_common_accel_tensor(input, "DeepseekV4CompressFusedNormRopeKernel");
    check_common_accel_tensor(norm_weight, "DeepseekV4CompressFusedNormRopeKernel");
    check_common_accel_tensor(freqs_cis, "DeepseekV4CompressFusedNormRopeKernel");
    check_common_accel_tensor(positions, "DeepseekV4CompressFusedNormRopeKernel");
    return new PlannedMeta{graph::GraphTensor(input),
                           graph::GraphTensor(norm_weight),
                           graph::GraphTensor(freqs_cis),
                           graph::GraphTensor(positions),
                           dsv4_scalar_type_for_kernel(input, "DeepseekV4CompressFusedNormRopeKernel"),
                           dsv4_scalar_type_for_kernel(norm_weight, "DeepseekV4CompressFusedNormRopeKernel"),
                           positions->dtype() == DataType::I64,
                           static_cast<int64_t>(input->size(0)),
                           static_cast<int64_t>(input->size(1)),
                           epsilon};
}

void run(void *planned_meta) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    deepseek_v4_flashmla_compute_kernel::launch_compress_fused_norm_rope(
        planned->input->data(),
        planned->input_dtype,
        planned->norm_weight->data(),
        planned->norm_weight_dtype,
        reinterpret_cast<const float *>(planned->freqs_cis->data()),
        planned->positions->data(),
        planned->positions_i64,
        planned->tokens,
        planned->dim,
        planned->epsilon,
        context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("DeepseekV4CompressFusedNormRopeKernel requires a HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_compress_fused_norm_rope_graph_impl

namespace deepseek_v4_compress_fused_norm_rope_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4CompressFusedNormRopeKernel,
                                       &deepseek_v4_compress_fused_norm_rope_graph_impl::plan,
                                       &deepseek_v4_compress_fused_norm_rope_graph_impl::run,
                                       &deepseek_v4_compress_fused_norm_rope_graph_impl::cleanup);
} // namespace deepseek_v4_compress_fused_norm_rope_register

DeepseekV4C4CompressStatefulKernel::DeepseekV4C4CompressStatefulKernel(
    Tensor output,
    const Tensor &kv_score_input,
    const Tensor &ape,
    Tensor compressor_state,
    const Tensor &write_loc,
    const Tensor &extra_loc,
    const Tensor &positions) {
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(),
                                 output,
                                 kv_score_input,
                                 ape,
                                 compressor_state,
                                 write_loc,
                                 extra_loc,
                                 positions);
}

void DeepseekV4C4CompressStatefulKernel::execute(Tensor output,
                                                 const Tensor &kv_score_input,
                                                 const Tensor &ape,
                                                 Tensor compressor_state,
                                                 const Tensor &write_loc,
                                                 const Tensor &extra_loc,
                                                 const Tensor &positions) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4C4CompressStatefulKernel,
                                      output,
                                      kv_score_input,
                                      ape,
                                      compressor_state,
                                      write_loc,
                                      extra_loc,
                                      positions);
}

namespace deepseek_v4_c4_compress_stateful_graph_impl {

struct PlannedMeta {
    graph::GraphTensor output;
    graph::GraphTensor kv_score_input;
    graph::GraphTensor ape;
    graph::GraphTensor compressor_state;
    graph::GraphTensor write_loc;
    graph::GraphTensor extra_loc;
    graph::GraphTensor positions;
    Tensor output_owner;
    int output_dtype;
    int kv_score_dtype;
    int state_dtype;
    int ape_dtype;
    int ape_layout;
    bool write_loc_i64;
    bool extra_loc_i64;
    bool positions_i64;
    int64_t extra_cols;
    int64_t tokens;
    int64_t head_dim;
};

void *plan(Tensor output,
           const Tensor &kv_score_input,
           const Tensor &ape,
           Tensor compressor_state,
           const Tensor &write_loc,
           const Tensor &extra_loc,
           const Tensor &positions) {
    check_common_accel_tensor(kv_score_input, "DeepseekV4C4CompressStatefulKernel");
    check_common_accel_tensor(ape, "DeepseekV4C4CompressStatefulKernel");
    check_common_accel_tensor(compressor_state, "DeepseekV4C4CompressStatefulKernel");
    check_common_accel_tensor(write_loc, "DeepseekV4C4CompressStatefulKernel");
    check_common_accel_tensor(extra_loc, "DeepseekV4C4CompressStatefulKernel");
    check_common_accel_tensor(positions, "DeepseekV4C4CompressStatefulKernel");
    if (kv_score_input->ndim() != 2 || kv_score_input->size(1) % 4 != 0) {
        throw std::runtime_error("DeepseekV4C4CompressStatefulKernel expects kv_score_input [tokens, 4 * head_dim].");
    }
    const int64_t tokens = static_cast<int64_t>(kv_score_input->size(0));
    const int64_t head_dim = static_cast<int64_t>(kv_score_input->size(1) / 4);
    if (output->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(head_dim)}) {
        throw std::runtime_error("DeepseekV4C4CompressStatefulKernel output shape mismatch.");
    }
    if (compressor_state->ndim() != 2 || compressor_state->size(1) != static_cast<size_t>(4 * head_dim) || compressor_state->size(0) % 4 != 0) {
        throw std::runtime_error("DeepseekV4C4CompressStatefulKernel compressor_state shape mismatch.");
    }
    if (write_loc->numel() != static_cast<size_t>(tokens) || positions->numel() != static_cast<size_t>(tokens)) {
        throw std::runtime_error("DeepseekV4C4CompressStatefulKernel metadata token count mismatch.");
    }
    int64_t extra_cols = 1;
    if (extra_loc->ndim() == 2) {
        if (extra_loc->size(0) != static_cast<size_t>(tokens) || extra_loc->size(1) < 1) {
            throw std::runtime_error("DeepseekV4C4CompressStatefulKernel expects extra_loc [tokens, >=1].");
        }
        extra_cols = static_cast<int64_t>(extra_loc->size(1));
    } else if (extra_loc->ndim() != 1 || extra_loc->size(0) != static_cast<size_t>(tokens)) {
        throw std::runtime_error("DeepseekV4C4CompressStatefulKernel expects extra_loc rank 1 or 2.");
    }
    const int ape_layout = c4_ape_layout(ape, head_dim, "DeepseekV4C4CompressStatefulKernel");
    return new PlannedMeta{graph::GraphTensor(output),
                           graph::GraphTensor(kv_score_input),
                           graph::GraphTensor(ape),
                           graph::GraphTensor(compressor_state),
                           graph::GraphTensor(write_loc),
                           graph::GraphTensor(extra_loc),
                           graph::GraphTensor(positions),
                           output,
                           dsv4_scalar_type_for_kernel(output, "DeepseekV4C4CompressStatefulKernel"),
                           dsv4_scalar_type_for_kernel(kv_score_input, "DeepseekV4C4CompressStatefulKernel"),
                           dsv4_scalar_type_for_kernel(compressor_state, "DeepseekV4C4CompressStatefulKernel"),
                           dsv4_scalar_type_for_kernel(ape, "DeepseekV4C4CompressStatefulKernel"),
                           ape_layout,
                           write_loc->dtype() == DataType::I64,
                           extra_loc->dtype() == DataType::I64,
                           positions->dtype() == DataType::I64,
                           extra_cols,
                           tokens,
                           head_dim};
}

void run(void *planned_meta) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    deepseek_v4_flashmla_compute_kernel::launch_c4_compress_stateful(
        planned->output->data(),
        planned->output_dtype,
        planned->kv_score_input->data(),
        planned->kv_score_dtype,
        planned->compressor_state->data(),
        planned->state_dtype,
        planned->ape->data(),
        planned->ape_dtype,
        planned->ape_layout,
        planned->write_loc->data(),
        planned->write_loc_i64,
        planned->extra_loc->data(),
        planned->extra_loc_i64,
        planned->extra_cols,
        planned->positions->data(),
        planned->positions_i64,
        planned->tokens,
        planned->head_dim,
        context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("DeepseekV4C4CompressStatefulKernel requires a HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_c4_compress_stateful_graph_impl

namespace deepseek_v4_c4_compress_stateful_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4C4CompressStatefulKernel,
                                       &deepseek_v4_c4_compress_stateful_graph_impl::plan,
                                       &deepseek_v4_c4_compress_stateful_graph_impl::run,
                                       &deepseek_v4_c4_compress_stateful_graph_impl::cleanup);
} // namespace deepseek_v4_c4_compress_stateful_register

DeepseekV4C128CompressStatefulKernel::DeepseekV4C128CompressStatefulKernel(
    Tensor output,
    const Tensor &kv_score_input,
    const Tensor &ape,
    Tensor compressor_state,
    const Tensor &write_loc,
    const Tensor &positions) {
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(),
                                 output,
                                 kv_score_input,
                                 ape,
                                 compressor_state,
                                 write_loc,
                                 positions);
}

void DeepseekV4C128CompressStatefulKernel::execute(Tensor output,
                                                   const Tensor &kv_score_input,
                                                   const Tensor &ape,
                                                   Tensor compressor_state,
                                                   const Tensor &write_loc,
                                                   const Tensor &positions) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4C128CompressStatefulKernel,
                                      output,
                                      kv_score_input,
                                      ape,
                                      compressor_state,
                                      write_loc,
                                      positions);
}

namespace deepseek_v4_c128_compress_stateful_graph_impl {

struct PlannedMeta {
    graph::GraphTensor output;
    graph::GraphTensor kv_score_input;
    graph::GraphTensor ape;
    graph::GraphTensor compressor_state;
    graph::GraphTensor write_loc;
    graph::GraphTensor positions;
    Tensor output_owner;
    int output_dtype;
    int kv_score_dtype;
    int state_dtype;
    int ape_dtype;
    bool write_loc_i64;
    bool positions_i64;
    int64_t tokens;
    int64_t head_dim;
};

void *plan(Tensor output,
           const Tensor &kv_score_input,
           const Tensor &ape,
           Tensor compressor_state,
           const Tensor &write_loc,
           const Tensor &positions) {
    check_common_accel_tensor(kv_score_input, "DeepseekV4C128CompressStatefulKernel");
    check_common_accel_tensor(ape, "DeepseekV4C128CompressStatefulKernel");
    check_common_accel_tensor(compressor_state, "DeepseekV4C128CompressStatefulKernel");
    check_common_accel_tensor(write_loc, "DeepseekV4C128CompressStatefulKernel");
    check_common_accel_tensor(positions, "DeepseekV4C128CompressStatefulKernel");
    if (kv_score_input->ndim() != 2 || kv_score_input->size(1) % 2 != 0) {
        throw std::runtime_error("DeepseekV4C128CompressStatefulKernel expects kv_score_input [tokens, 2 * head_dim].");
    }
    const int64_t tokens = static_cast<int64_t>(kv_score_input->size(0));
    const int64_t head_dim = static_cast<int64_t>(kv_score_input->size(1) / 2);
    if (head_dim != kDsv4FlashMlaQDim) {
        throw std::runtime_error("DeepseekV4C128CompressStatefulKernel expects head_dim 512.");
    }
    if (output->shape() != Shape{static_cast<size_t>(tokens), static_cast<size_t>(head_dim)}) {
        throw std::runtime_error("DeepseekV4C128CompressStatefulKernel output shape mismatch.");
    }
    if (compressor_state->ndim() != 2 || compressor_state->size(1) != static_cast<size_t>(2 * head_dim) || compressor_state->size(0) % 128 != 0) {
        throw std::runtime_error("DeepseekV4C128CompressStatefulKernel compressor_state shape mismatch.");
    }
    if (write_loc->numel() != static_cast<size_t>(tokens) || positions->numel() != static_cast<size_t>(tokens)) {
        throw std::runtime_error("DeepseekV4C128CompressStatefulKernel metadata token count mismatch.");
    }
    if (ape->ndim() != 2 || ape->size(0) < 128 || ape->size(1) != static_cast<size_t>(head_dim)) {
        throw std::runtime_error("DeepseekV4C128CompressStatefulKernel expects ape [>=128, head_dim].");
    }
    return new PlannedMeta{graph::GraphTensor(output),
                           graph::GraphTensor(kv_score_input),
                           graph::GraphTensor(ape),
                           graph::GraphTensor(compressor_state),
                           graph::GraphTensor(write_loc),
                           graph::GraphTensor(positions),
                           output,
                           dsv4_scalar_type_for_kernel(output, "DeepseekV4C128CompressStatefulKernel"),
                           dsv4_scalar_type_for_kernel(kv_score_input, "DeepseekV4C128CompressStatefulKernel"),
                           dsv4_scalar_type_for_kernel(compressor_state, "DeepseekV4C128CompressStatefulKernel"),
                           dsv4_scalar_type_for_kernel(ape, "DeepseekV4C128CompressStatefulKernel"),
                           write_loc->dtype() == DataType::I64,
                           positions->dtype() == DataType::I64,
                           tokens,
                           head_dim};
}

void run(void *planned_meta) {
#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    deepseek_v4_flashmla_compute_kernel::launch_c128_compress_stateful(
        planned->output->data(),
        planned->output_dtype,
        planned->kv_score_input->data(),
        planned->kv_score_dtype,
        planned->compressor_state->data(),
        planned->state_dtype,
        planned->ape->data(),
        planned->ape_dtype,
        planned->write_loc->data(),
        planned->write_loc_i64,
        planned->positions->data(),
        planned->positions_i64,
        planned->tokens,
        planned->head_dim,
        context::getStream());
#else
    (void)planned_meta;
    throw std::runtime_error("DeepseekV4C128CompressStatefulKernel requires a HYGON/NVIDIA build.");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_c128_compress_stateful_graph_impl

namespace deepseek_v4_c128_compress_stateful_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4C128CompressStatefulKernel,
                                       &deepseek_v4_c128_compress_stateful_graph_impl::plan,
                                       &deepseek_v4_c128_compress_stateful_graph_impl::run,
                                       &deepseek_v4_c128_compress_stateful_graph_impl::cleanup);
} // namespace deepseek_v4_c128_compress_stateful_register

void deepseek_v4_compress_fused_norm_rope_kernel_(Tensor input,
                                                  const Tensor &norm_weight,
                                                  float epsilon,
                                                  const Tensor &freqs_cis,
                                                  const Tensor &positions) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
#else
#endif
    check_compress_fused_norm_rope_shapes(input, norm_weight, freqs_cis, positions);
    check_common_accel_tensor(input, "deepseek_v4_compress_fused_norm_rope_kernel_");
    check_common_accel_tensor(norm_weight, "deepseek_v4_compress_fused_norm_rope_kernel_");
    check_common_accel_tensor(freqs_cis, "deepseek_v4_compress_fused_norm_rope_kernel_");
    check_common_accel_tensor(positions, "deepseek_v4_compress_fused_norm_rope_kernel_");
    DeepseekV4CompressFusedNormRopeKernel::execute(input,
                                                     norm_weight,
                                                     epsilon,
                                                     freqs_cis,
                                                     positions);
#else
    (void)input;
    (void)norm_weight;
    (void)epsilon;
    (void)freqs_cis;
    (void)positions;
    throw std::runtime_error("deepseek_v4_compress_fused_norm_rope_kernel_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_compress_fused_norm_rope_(Tensor input,
                                           const Tensor &norm_weight,
                                           float epsilon,
                                           const Tensor &freqs_cis,
                                           const Tensor &positions) {
    deepseek_v4_compress_fused_norm_rope_kernel_(input, norm_weight, epsilon, freqs_cis, positions);
}


Tensor deepseek_v4_c4_compress_stateful_kernel(const Tensor &kv_score_input,
                                               const Tensor &ape,
                                               Tensor compressor_state,
                                               const Tensor &write_loc,
                                               const Tensor &extra_loc,
                                               const Tensor &positions) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
#else
#endif
    check_common_accel_tensor(kv_score_input, "deepseek_v4_c4_compress_stateful_kernel");
    check_common_accel_tensor(ape, "deepseek_v4_c4_compress_stateful_kernel");
    check_common_accel_tensor(compressor_state, "deepseek_v4_c4_compress_stateful_kernel");
    check_common_accel_tensor(write_loc, "deepseek_v4_c4_compress_stateful_kernel");
    check_common_accel_tensor(extra_loc, "deepseek_v4_c4_compress_stateful_kernel");
    check_common_accel_tensor(positions, "deepseek_v4_c4_compress_stateful_kernel");
    if (kv_score_input->ndim() != 2 || kv_score_input->size(1) % 4 != 0) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_kernel expects kv_score_input [tokens, 4 * head_dim].");
    }
    const int64_t tokens = static_cast<int64_t>(kv_score_input->size(0));
    const int64_t head_dim = static_cast<int64_t>(kv_score_input->size(1) / 4);
    if (compressor_state->ndim() != 2 || compressor_state->size(1) != static_cast<size_t>(4 * head_dim) || compressor_state->size(0) % 4 != 0) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_kernel expects compressor_state [4 * groups, 4 * head_dim].");
    }
    if (write_loc->numel() != static_cast<size_t>(tokens) || positions->numel() != static_cast<size_t>(tokens)) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_kernel metadata token count mismatch.");
    }
    if (write_loc->dtype() != DataType::I32 && write_loc->dtype() != DataType::I64) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_kernel write_loc must be int32/int64.");
    }
    if (positions->dtype() != DataType::I32 && positions->dtype() != DataType::I64) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_kernel positions must be int32/int64.");
    }
    if (extra_loc->dtype() != DataType::I32 && extra_loc->dtype() != DataType::I64) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_kernel extra_loc must be int32/int64.");
    }
    int64_t extra_cols = 1;
    if (extra_loc->ndim() == 2) {
        if (extra_loc->size(0) != static_cast<size_t>(tokens) || extra_loc->size(1) < 1) {
            throw std::runtime_error("deepseek_v4_c4_compress_stateful_kernel expects extra_loc [tokens, >=1].");
        }
        extra_cols = static_cast<int64_t>(extra_loc->size(1));
    } else if (extra_loc->ndim() != 1 || extra_loc->size(0) != static_cast<size_t>(tokens)) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_kernel expects extra_loc rank 1 or 2.");
    }
    const int ape_layout = c4_ape_layout(ape, head_dim, "deepseek_v4_c4_compress_stateful_kernel");
    auto output = Tensor::empty({static_cast<size_t>(tokens), static_cast<size_t>(head_dim)}, kv_score_input->dtype(), kv_score_input->device());
    DeepseekV4C4CompressStatefulKernel::execute(output,
                                                  kv_score_input,
                                                  ape,
                                                  compressor_state,
                                                  write_loc,
                                                  extra_loc,
                                                  positions);
    return output;
#else
    (void)kv_score_input;
    (void)ape;
    (void)compressor_state;
    (void)write_loc;
    (void)extra_loc;
    (void)positions;
    throw std::runtime_error("deepseek_v4_c4_compress_stateful_kernel requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

Tensor deepseek_v4_c4_compress_stateful(const Tensor &kv_score_input,
                                        const Tensor &ape,
                                        Tensor compressor_state,
                                        const Tensor &write_loc,
                                        const Tensor &extra_loc,
                                        const Tensor &positions) {
    return deepseek_v4_c4_compress_stateful_kernel(kv_score_input,
                                                   ape,
                                                   compressor_state,
                                                   write_loc,
                                                   extra_loc,
                                                   positions);
}


Tensor deepseek_v4_c128_compress_stateful_kernel(const Tensor &kv_score_input,
                                                 const Tensor &ape,
                                                 Tensor compressor_state,
                                                 const Tensor &write_loc,
                                                 const Tensor &positions) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
#if defined(ENABLE_HYGON_API)
#else
#endif
    check_common_accel_tensor(kv_score_input, "deepseek_v4_c128_compress_stateful_kernel");
    check_common_accel_tensor(ape, "deepseek_v4_c128_compress_stateful_kernel");
    check_common_accel_tensor(compressor_state, "deepseek_v4_c128_compress_stateful_kernel");
    check_common_accel_tensor(write_loc, "deepseek_v4_c128_compress_stateful_kernel");
    check_common_accel_tensor(positions, "deepseek_v4_c128_compress_stateful_kernel");
    if (kv_score_input->ndim() != 2 || kv_score_input->size(1) % 2 != 0) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_kernel expects kv_score_input [tokens, 2 * head_dim].");
    }
    const int64_t tokens = static_cast<int64_t>(kv_score_input->size(0));
    const int64_t head_dim = static_cast<int64_t>(kv_score_input->size(1) / 2);
    if (head_dim != kDsv4FlashMlaQDim) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_kernel expects head_dim 512.");
    }
    if (compressor_state->ndim() != 2 || compressor_state->size(1) != static_cast<size_t>(2 * head_dim) || compressor_state->size(0) % 128 != 0) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_kernel expects compressor_state [128 * groups, 2 * head_dim].");
    }
    if (write_loc->numel() != static_cast<size_t>(tokens) || positions->numel() != static_cast<size_t>(tokens)) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_kernel metadata token count mismatch.");
    }
    if (write_loc->dtype() != DataType::I32 && write_loc->dtype() != DataType::I64) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_kernel write_loc must be int32/int64.");
    }
    if (positions->dtype() != DataType::I32 && positions->dtype() != DataType::I64) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_kernel positions must be int32/int64.");
    }
    if (ape->ndim() != 2 || ape->size(0) < 128 || ape->size(1) != static_cast<size_t>(head_dim)) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_kernel expects ape [>=128, head_dim].");
    }
    auto output = Tensor::empty({static_cast<size_t>(tokens), static_cast<size_t>(head_dim)}, kv_score_input->dtype(), kv_score_input->device());
    DeepseekV4C128CompressStatefulKernel::execute(output,
                                                    kv_score_input,
                                                    ape,
                                                    compressor_state,
                                                    write_loc,
                                                    positions);
    return output;
#else
    (void)kv_score_input;
    (void)ape;
    (void)compressor_state;
    (void)write_loc;
    (void)positions;
    throw std::runtime_error("deepseek_v4_c128_compress_stateful_kernel requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

Tensor deepseek_v4_c128_compress_stateful(const Tensor &kv_score_input,
                                          const Tensor &ape,
                                          Tensor compressor_state,
                                          const Tensor &write_loc,
                                          const Tensor &positions) {
    return deepseek_v4_c128_compress_stateful_kernel(kv_score_input,
                                                     ape,
                                                     compressor_state,
                                                     write_loc,
                                                     positions);
}


DeepseekV4FlashMLASparseAttentionSchedule deepseek_v4_flashmla_sparse_attention_with_metadata_impl(
    const Tensor &q,
    const Tensor &raw_cache,
    const Tensor &indices,
    const Tensor &topk_lengths,
    std::optional<Tensor> attn_sink,
    Tensor output,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits,
    float softmax_scale,
    int page_size,
    int head_dim_v,
    std::optional<Tensor> extra_raw_cache,
    std::optional<Tensor> extra_indices,
    std::optional<Tensor> extra_topk_lengths,
    int extra_page_size) {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    check_hygon_or_nvidia_tensor(q, "deepseek_v4_flashmla_sparse_attention_");
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());

    check_sparse_attention_shapes(q, raw_cache, indices, topk_lengths, output, page_size, head_dim_v);

    const bool has_extra = extra_raw_cache.has_value() || extra_indices.has_value() || extra_topk_lengths.has_value();
    if (has_extra && !(extra_raw_cache.has_value() && extra_indices.has_value() && extra_topk_lengths.has_value())) {
        throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ requires all extra cache tensors when any extra tensor is provided.");
    }
    if (has_extra && extra_page_size <= 0) {
        throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ requires positive extra_page_size for extra cache.");
    }

    auto q_at_original = infinicore::adaptor::to_aten_tensor(q).contiguous();
    auto raw_cache_at = infinicore::adaptor::to_aten_tensor(raw_cache);
    auto indices_at = infinicore::adaptor::to_aten_tensor(indices);
    auto topk_lengths_at = infinicore::adaptor::to_aten_tensor(topk_lengths);
    auto output_at = infinicore::adaptor::to_aten_tensor(output);

    if (!raw_cache_at.is_contiguous()) {
        throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ expects contiguous raw cache.");
    }
    if (indices_at.scalar_type() != at::kInt) {
        indices_at = indices_at.to(at::kInt);
    }
    if (topk_lengths_at.scalar_type() != at::kInt) {
        topk_lengths_at = topk_lengths_at.to(at::kInt);
    }
    indices_at = indices_at.contiguous();
    topk_lengths_at = topk_lengths_at.contiguous();

    const bool q_was_3d = q_at_original.dim() == 3;
    const int64_t tokens = q_was_3d ? q_at_original.size(0) : q_at_original.size(0) * q_at_original.size(1);
    const int64_t heads = q_at_original.size(q_at_original.dim() - 2);
    if (tokens == 0) {
        return {};
    }
    auto q_flash = q_at_original.reshape({tokens, heads, kDsv4FlashMlaQDim}).unsqueeze(1);

    auto reshape_indices = [](at::Tensor meta, int64_t token_count, const char *name) -> at::Tensor {
        if (meta.dim() == 2) {
            if (meta.size(0) != token_count) {
                throw std::runtime_error(std::string("deepseek_v4_flashmla_sparse_attention_ ") + name + " token count mismatch.");
            }
            return meta.reshape({token_count, 1, meta.size(1)});
        }
        if (meta.dim() == 3) {
            if (meta.size(0) * meta.size(1) != token_count) {
                throw std::runtime_error(std::string("deepseek_v4_flashmla_sparse_attention_ ") + name + " token count mismatch.");
            }
            return meta.reshape({token_count, 1, meta.size(2)});
        }
        throw std::runtime_error(std::string("deepseek_v4_flashmla_sparse_attention_ expects ") + name + " rank 2 or 3.");
    };

    auto indices_flash = reshape_indices(indices_at, tokens, "indices");
    if (topk_lengths_at.numel() != tokens) {
        throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ topk_lengths token count mismatch.");
    }
    auto topk_lengths_flat = topk_lengths_at.reshape({tokens});

    auto view_raw_cache = [](const at::Tensor &cache_at, int cache_page_size, const char *name) -> at::Tensor {
        if (cache_at.dim() != 2) {
            throw std::runtime_error(std::string("deepseek_v4_flashmla_sparse_attention_ expects ") + name + " [blocks, page_bytes].");
        }
        const int64_t expected_page_bytes = dsv4_flashmla_page_bytes(cache_page_size);
        if (cache_at.size(1) != expected_page_bytes) {
            throw std::runtime_error(std::string("deepseek_v4_flashmla_sparse_attention_ ") + name + " page_bytes mismatch.");
        }
        const int64_t cache_bytes_per_page = static_cast<int64_t>(cache_page_size) * kDsv4FlashMlaBytesPerToken;
        auto cache_bytes = cache_at.slice(1, 0, cache_bytes_per_page);
        return cache_bytes.view(at::ScalarType::Float8_e4m3fn)
            .reshape({cache_at.size(0), static_cast<int64_t>(cache_page_size), 1, kDsv4FlashMlaBytesPerToken});
    };

    auto k_cache_fp8 = view_raw_cache(raw_cache_at, page_size, "raw cache");

    std::optional<at::Tensor> topk_lengths_opt = topk_lengths_flat;
    std::optional<at::Tensor> attn_sink_opt = std::nullopt;
    at::Tensor attn_sink_storage;
    if (attn_sink.has_value()) {
        attn_sink_storage = infinicore::adaptor::to_aten_tensor(attn_sink.value()).to(at::kFloat).contiguous();
        if (attn_sink_storage.numel() < heads) {
            throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ attn_sink has fewer heads than q.");
        }
        attn_sink_opt = attn_sink_storage.reshape({attn_sink_storage.numel()}).slice(0, 0, heads);
    }

    std::optional<at::Tensor> extra_k_cache_opt = std::nullopt;
    std::optional<at::Tensor> extra_indices_opt = std::nullopt;
    std::optional<at::Tensor> extra_topk_lengths_opt = std::nullopt;
    at::Tensor extra_k_cache_storage;
    at::Tensor extra_indices_storage;
    at::Tensor extra_topk_lengths_storage;
    if (has_extra) {
        auto extra_raw_cache_at = infinicore::adaptor::to_aten_tensor(extra_raw_cache.value());
        if (!extra_raw_cache_at.is_contiguous()) {
            throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ expects contiguous extra raw cache.");
        }
        extra_k_cache_storage = view_raw_cache(extra_raw_cache_at, extra_page_size, "extra raw cache");

        extra_indices_storage = infinicore::adaptor::to_aten_tensor(extra_indices.value());
        if (extra_indices_storage.scalar_type() != at::kInt) {
            extra_indices_storage = extra_indices_storage.to(at::kInt);
        }
        extra_indices_storage = reshape_indices(extra_indices_storage.contiguous(), tokens, "extra_indices");

        extra_topk_lengths_storage = infinicore::adaptor::to_aten_tensor(extra_topk_lengths.value());
        if (extra_topk_lengths_storage.scalar_type() != at::kInt) {
            extra_topk_lengths_storage = extra_topk_lengths_storage.to(at::kInt);
        }
        extra_topk_lengths_storage = extra_topk_lengths_storage.contiguous();
        if (extra_topk_lengths_storage.numel() != tokens) {
            throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ extra_topk_lengths token count mismatch.");
        }
        extra_topk_lengths_storage = extra_topk_lengths_storage.reshape({tokens});

        extra_k_cache_opt = extra_k_cache_storage;
        extra_indices_opt = extra_indices_storage;
        extra_topk_lengths_opt = extra_topk_lengths_storage;
    }

    auto tile_scheduler_metadata_at = optional_i32_aten_tensor(
        tile_scheduler_metadata, "tile_scheduler_metadata", "deepseek_v4_flashmla_sparse_attention_with_metadata_");
    auto num_splits_at = optional_i32_aten_tensor(
        num_splits, "num_splits", "deepseek_v4_flashmla_sparse_attention_with_metadata_");
    static auto flash_mla_sparse_decode_fn = reinterpret_cast<FlashMlaSparseDecodeFn>(
        resolve_flashmla_sparse_decode("deepseek_v4_flashmla_sparse_attention_"));
    auto flash_out = flash_mla_sparse_decode_fn(q_flash,
                                                k_cache_fp8,
                                                indices_flash,
                                                topk_lengths_opt,
                                                attn_sink_opt,
                                                tile_scheduler_metadata_at,
                                                num_splits_at,
                                                extra_k_cache_opt,
                                                extra_indices_opt,
                                                extra_topk_lengths_opt,
                                                head_dim_v,
                                                softmax_scale);
    auto result = std::get<0>(flash_out).reshape({tokens, heads, static_cast<int64_t>(head_dim_v)});
    if (!q_was_3d) {
        result = result.reshape(output_at.sizes());
    }
    if (!output_at.sizes().equals(result.sizes())) {
        throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ output shape mismatch.");
    }
    output_at.copy_(result);
    return build_flashmla_sparse_schedule_result(tile_scheduler_metadata,
                                                num_splits,
                                                std::get<2>(flash_out),
                                                std::get<3>(flash_out),
                                                q->device(),
                                                "deepseek_v4_flashmla_sparse_attention_with_metadata_");
#else
    (void)q;
    (void)raw_cache;
    (void)indices;
    (void)topk_lengths;
    (void)attn_sink;
    (void)output;
    (void)tile_scheduler_metadata;
    (void)num_splits;
    (void)softmax_scale;
    (void)page_size;
    (void)head_dim_v;
    (void)extra_raw_cache;
    (void)extra_indices;
    (void)extra_topk_lengths;
    (void)extra_page_size;
    throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

DeepseekV4FlashMlaSparseAttentionWithMetadata::DeepseekV4FlashMlaSparseAttentionWithMetadata(
    const Tensor &q,
    const Tensor &raw_cache,
    const Tensor &indices,
    const Tensor &topk_lengths,
    std::optional<Tensor> attn_sink,
    Tensor output,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits,
    float softmax_scale,
    int page_size,
    int head_dim_v,
    std::optional<Tensor> extra_raw_cache,
    std::optional<Tensor> extra_indices,
    std::optional<Tensor> extra_topk_lengths,
    int extra_page_size) {
    // FlashMLA schedule tensors are owned by the graph plan, so the wrapper
    // can participate in device graph capture while replaying the third-party SO.
    INFINICORE_GRAPH_OP_DISPATCH(q->device().getType(),
                                 q,
                                 raw_cache,
                                 indices,
                                 topk_lengths,
                                 attn_sink,
                                 output,
                                 tile_scheduler_metadata,
                                 num_splits,
                                 softmax_scale,
                                 page_size,
                                 head_dim_v,
                                 extra_raw_cache,
                                 extra_indices,
                                 extra_topk_lengths,
                                 extra_page_size);
}

void DeepseekV4FlashMlaSparseAttentionWithMetadata::execute(
    const Tensor &q,
    const Tensor &raw_cache,
    const Tensor &indices,
    const Tensor &topk_lengths,
    std::optional<Tensor> attn_sink,
    Tensor output,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits,
    float softmax_scale,
    int page_size,
    int head_dim_v,
    std::optional<Tensor> extra_raw_cache,
    std::optional<Tensor> extra_indices,
    std::optional<Tensor> extra_topk_lengths,
    int extra_page_size) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4FlashMlaSparseAttentionWithMetadata,
                                      q,
                                      raw_cache,
                                      indices,
                                      topk_lengths,
                                      attn_sink,
                                      output,
                                      tile_scheduler_metadata,
                                      num_splits,
                                      softmax_scale,
                                      page_size,
                                      head_dim_v,
                                      extra_raw_cache,
                                      extra_indices,
                                      extra_topk_lengths,
                                      extra_page_size);
}

namespace deepseek_v4_flashmla_sparse_attention_graph_impl {

struct PlannedMeta {
    graph::GraphTensor q;
    graph::GraphTensor raw_cache;
    graph::GraphTensor indices;
    graph::GraphTensor topk_lengths;
    std::optional<graph::GraphTensor> attn_sink;
    graph::GraphTensor output;
    graph::GraphTensor tile_scheduler_metadata;
    graph::GraphTensor num_splits;
    Tensor tile_scheduler_metadata_owner;
    Tensor num_splits_owner;
    float softmax_scale;
    int page_size;
    int head_dim_v;
    std::optional<graph::GraphTensor> extra_raw_cache;
    std::optional<graph::GraphTensor> extra_indices;
    std::optional<graph::GraphTensor> extra_topk_lengths;
    int extra_page_size;
};

std::optional<graph::GraphTensor> to_graph_optional(const std::optional<Tensor> &tensor) {
    if (!tensor.has_value() || !tensor.value()) {
        return std::nullopt;
    }
    return graph::GraphTensor(tensor.value());
}

std::optional<Tensor> to_tensor_optional(const std::optional<graph::GraphTensor> &tensor) {
    if (!tensor.has_value()) {
        return std::nullopt;
    }
    return tensor.value();
}

void *plan(const Tensor &q,
           const Tensor &raw_cache,
           const Tensor &indices,
           const Tensor &topk_lengths,
           std::optional<Tensor> attn_sink,
           Tensor output,
           std::optional<Tensor> tile_scheduler_metadata,
           std::optional<Tensor> num_splits,
           float softmax_scale,
           int page_size,
           int head_dim_v,
           std::optional<Tensor> extra_raw_cache,
           std::optional<Tensor> extra_indices,
           std::optional<Tensor> extra_topk_lengths,
           int extra_page_size) {
    check_hygon_or_nvidia_tensor(q, "deepseek_v4_flashmla_sparse_attention_with_metadata_");
    check_sparse_attention_shapes(q, raw_cache, indices, topk_lengths, output, page_size, head_dim_v);
    if (!tile_scheduler_metadata.has_value() || !tile_scheduler_metadata.value() ||
        !num_splits.has_value() || !num_splits.value()) {
        throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_with_metadata_ graph path requires prebuilt FlashMLA metadata.");
    }
    return new PlannedMeta{graph::GraphTensor(q),
                           graph::GraphTensor(raw_cache),
                           graph::GraphTensor(indices),
                           graph::GraphTensor(topk_lengths),
                           to_graph_optional(attn_sink),
                           graph::GraphTensor(output),
                           graph::GraphTensor(tile_scheduler_metadata.value()),
                           graph::GraphTensor(num_splits.value()),
                           tile_scheduler_metadata.value(),
                           num_splits.value(),
                           softmax_scale,
                           page_size,
                           head_dim_v,
                           to_graph_optional(extra_raw_cache),
                           to_graph_optional(extra_indices),
                           to_graph_optional(extra_topk_lengths),
                           extra_page_size};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    (void)deepseek_v4_flashmla_sparse_attention_with_metadata_impl(planned->q,
                                                                   planned->raw_cache,
                                                                   planned->indices,
                                                                   planned->topk_lengths,
                                                                   to_tensor_optional(planned->attn_sink),
                                                                   planned->output,
                                                                   planned->tile_scheduler_metadata,
                                                                   planned->num_splits,
                                                                   planned->softmax_scale,
                                                                   planned->page_size,
                                                                   planned->head_dim_v,
                                                                   to_tensor_optional(planned->extra_raw_cache),
                                                                   to_tensor_optional(planned->extra_indices),
                                                                   to_tensor_optional(planned->extra_topk_lengths),
                                                                   planned->extra_page_size);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_flashmla_sparse_attention_graph_impl

namespace deepseek_v4_flashmla_sparse_attention_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4FlashMlaSparseAttentionWithMetadata,
                                       &deepseek_v4_flashmla_sparse_attention_graph_impl::plan,
                                       &deepseek_v4_flashmla_sparse_attention_graph_impl::run,
                                       &deepseek_v4_flashmla_sparse_attention_graph_impl::cleanup);
} // namespace deepseek_v4_flashmla_sparse_attention_register

DeepseekV4FlashMLASparseAttentionSchedule deepseek_v4_flashmla_sparse_attention_with_metadata_(
    const Tensor &q,
    const Tensor &raw_cache,
    const Tensor &indices,
    const Tensor &topk_lengths,
    std::optional<Tensor> attn_sink,
    Tensor output,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits,
    float softmax_scale,
    int page_size,
    int head_dim_v,
    std::optional<Tensor> extra_raw_cache,
    std::optional<Tensor> extra_indices,
    std::optional<Tensor> extra_topk_lengths,
    int extra_page_size) {
    const bool has_schedule = tile_scheduler_metadata.has_value() && tile_scheduler_metadata.value() &&
                              num_splits.has_value() && num_splits.value();
    if (has_schedule) {
        DeepseekV4FlashMlaSparseAttentionWithMetadata::execute(q,
                                                               raw_cache,
                                                               indices,
                                                               topk_lengths,
                                                               attn_sink,
                                                               output,
                                                               tile_scheduler_metadata,
                                                               num_splits,
                                                               softmax_scale,
                                                               page_size,
                                                               head_dim_v,
                                                               extra_raw_cache,
                                                               extra_indices,
                                                               extra_topk_lengths,
                                                               extra_page_size);
        return {tile_scheduler_metadata.value(), num_splits.value()};
    }
    return deepseek_v4_flashmla_sparse_attention_with_metadata_impl(q,
                                                                   raw_cache,
                                                                   indices,
                                                                   topk_lengths,
                                                                   attn_sink,
                                                                   output,
                                                                   tile_scheduler_metadata,
                                                                   num_splits,
                                                                   softmax_scale,
                                                                   page_size,
                                                                   head_dim_v,
                                                                   extra_raw_cache,
                                                                   extra_indices,
                                                                   extra_topk_lengths,
                                                                   extra_page_size);
}

void deepseek_v4_flashmla_sparse_attention_(const Tensor &q,
                                            const Tensor &raw_cache,
                                            const Tensor &indices,
                                            const Tensor &topk_lengths,
                                            std::optional<Tensor> attn_sink,
                                            Tensor output,
                                            float softmax_scale,
                                            int page_size,
                                            int head_dim_v,
                                            std::optional<Tensor> extra_raw_cache,
                                            std::optional<Tensor> extra_indices,
                                            std::optional<Tensor> extra_topk_lengths,
                                            int extra_page_size) {
    (void)deepseek_v4_flashmla_sparse_attention_with_metadata_(q,
                                                              raw_cache,
                                                              indices,
                                                              topk_lengths,
                                                              attn_sink,
                                                              output,
                                                              std::nullopt,
                                                              std::nullopt,
                                                              softmax_scale,
                                                              page_size,
                                                              head_dim_v,
                                                              extra_raw_cache,
                                                              extra_indices,
                                                              extra_topk_lengths,
                                                              extra_page_size);
}
} // namespace infinicore::op

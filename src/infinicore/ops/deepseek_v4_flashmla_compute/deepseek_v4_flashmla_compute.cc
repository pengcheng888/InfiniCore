#include "infinicore/ops/deepseek_v4_flashmla_compute.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <elf.h>
#include <fstream>
#include <limits>
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
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4FlashMlaSparseAttentionOutWorkspace);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4FlashMlaSparseAttentionMetadata);

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
    if ((indices->dtype() != DataType::I32 && indices->dtype() != DataType::I64) || (topk_lengths->dtype() != DataType::I32 && topk_lengths->dtype() != DataType::I64)) {
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
    if (output->size(output->ndim() - 2) != q->size(q->ndim() - 2) || output->size(output->ndim() - 1) != static_cast<size_t>(head_dim_v)) {
        throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ output head/head_dim shape mismatch.");
    }
}

struct FlashMlaSparseCaptureOwners {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    at::Tensor q;
    at::Tensor q_flash;
    at::Tensor k_cache;
    at::Tensor indices;
    at::Tensor topk_lengths;
    at::Tensor attn_sink;
    at::Tensor tile_scheduler_metadata;
    at::Tensor num_splits;
    at::Tensor extra_k_cache;
    at::Tensor extra_indices;
    at::Tensor extra_topk_lengths;
    at::Tensor out;
    at::Tensor lse;
#endif
};

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
constexpr const char *kFlashMlaSparseDecodeInterfaceSymbol = "_ZL28sparse_attn_decode_interfaceRKN2at6TensorES2_S2_RKSt8optionalIS0_ES6_RS4_S7_S6_S6_S6_if";
constexpr const char *kFlashMlaSparseDecodeModel1H16Symbol = "_ZN5gfx936decode10sparse_fp839run_flash_splitkv_mla_fp8_sparse_kernelIL9ModelType1ELi16EEEvRK22SparseAttnDecodeParams";
constexpr const char *kFlashMlaSparseDecodeModel1H64Symbol = "_ZN5gfx936decode10sparse_fp839run_flash_splitkv_mla_fp8_sparse_kernelIL9ModelType1ELi64EEEvRK22SparseAttnDecodeParams";
constexpr const char *kFlashMlaSparseDecodeModel1H128Symbol = "_ZN5gfx936decode10sparse_fp839run_flash_splitkv_mla_fp8_sparse_kernelIL9ModelType1ELi128EEEvRK22SparseAttnDecodeParams";
constexpr const char *kFlashMlaSparseDecodeV32H16Symbol = "_ZN5gfx936decode10sparse_fp839run_flash_splitkv_mla_fp8_sparse_kernelIL9ModelType0ELi16EEEvRK22SparseAttnDecodeParams";
constexpr const char *kFlashMlaSparseDecodeV32H64Symbol = "_ZN5gfx936decode10sparse_fp839run_flash_splitkv_mla_fp8_sparse_kernelIL9ModelType0ELi64EEEvRK22SparseAttnDecodeParams";
constexpr const char *kFlashMlaSparseDecodeV32H128Symbol = "_ZN5gfx936decode10sparse_fp839run_flash_splitkv_mla_fp8_sparse_kernelIL9ModelType0ELi128EEEvRK22SparseAttnDecodeParams";
constexpr const char *kFlashMlaSparseDecodeMetadataSymbol = "_ZN4gfx96decode34run_get_decoding_sched_meta_kernelER24GetDecodeSchedMetaParams";
constexpr const char *kFlashMlaCombineBf16Symbol = "_ZN4gfx96decode28run_flash_mla_combine_kernelIN7cutlass10bfloat16_tEEEvR13CombineParams";
constexpr const char *kDefaultFlashMlaSoPath = "/usr/local/lib/python3.10/dist-packages/flash_mla/cuda.cpython-310-x86_64-linux-gnu.so";
constexpr const char *kFlashMlaAnchorSymbol = "PyInit_cuda";
constexpr float kFlashMlaLog2E = 1.4426950408889634f;

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

enum class FlashMlaModelType : int {
    V32 = 0,
    MODEL1 = 1,
};

struct FlashMlaDecodingSchedMeta {
    int begin_req_idx;
    int end_req_idx;
    int begin_block_idx;
    int end_block_idx;
    int begin_split_idx;
    int is_first_req_splitted;
    int is_last_req_splitted;
    int pad;
};
static_assert(sizeof(FlashMlaDecodingSchedMeta) == 8 * sizeof(int),
              "FlashMLA DecodingSchedMeta ABI mismatch.");

struct FlashMlaSparseDecodeParams {
    int b, s_q;
    int h_q, h_kv;
    int d_qk, d_v;
    float sm_scale, sm_scale_div_log2;
    int num_blocks, page_block_size, topk;
    FlashMlaModelType model_type;

    void *q;
    void *kv;
    int *indices;
    int *topk_length;
    float *attn_sink;

    float *lse;
    void *out;

    int extra_num_blocks, extra_page_block_size, extra_topk;
    void *extra_kv;
    int *extra_indices;
    int *extra_topk_length;

    int stride_q_b, stride_q_s_q, stride_q_h_q;
    int stride_kv_block, stride_kv_row;
    int stride_indices_b, stride_indices_s_q;
    int stride_lse_b, stride_lse_s_q;
    int stride_o_b, stride_o_s_q, stride_o_h_q;
    int stride_extra_kv_block, stride_extra_kv_row;
    int stride_extra_indices_b, stride_extra_indices_s_q;

    void *stream;

    float *lse_accum;
    float *o_accum;
    int stride_lse_accum_split, stride_lse_accum_s_q;
    int stride_o_accum_split, stride_o_accum_s_q, stride_o_accum_h_q;
    FlashMlaDecodingSchedMeta *tile_scheduler_metadata_ptr;
    int *num_splits_ptr;
    int num_sm_parts;
};

struct FlashMlaCombineParams {
    int b, s_q, h_q, d_v;

    float *lse;
    void *out;
    int stride_lse_b, stride_lse_s_q;
    int stride_o_b, stride_o_s_q, stride_o_h_q;

    float *lse_accum;
    float *o_accum;
    int stride_lse_accum_split, stride_lse_accum_s_q;
    int stride_o_accum_split, stride_o_accum_s_q, stride_o_accum_h_q;

    FlashMlaDecodingSchedMeta *tile_scheduler_metadata_ptr;
    int *num_splits_ptr;
    int num_sm_parts;

    float *attn_sink;

    void *stream;
    bool use_split_kv;
    int num_splits;
    int *seqlens_k_ptr;
    int partition_block_nums;
};
static_assert(sizeof(FlashMlaSparseDecodeParams) == 280,
              "FlashMLA SparseAttnDecodeParams ABI mismatch.");
static_assert(sizeof(FlashMlaCombineParams) == 160,
              "FlashMLA CombineParams ABI mismatch.");

struct FlashMlaGetDecodeSchedMetaParams {
    int b;
    int s_q;
    int block_size_n;
    int fixed_overhead_num_blocks;
    int topk;
    int extra_topk;
    int *topk_length;
    int *extra_topk_length;
    int *seqlens_k_ptr;
    FlashMlaDecodingSchedMeta *tile_scheduler_metadata_ptr;
    int *num_splits_ptr;
    int num_sm_parts;
    void *stream;
};
static_assert(sizeof(FlashMlaGetDecodeSchedMetaParams) == 80,
              "FlashMLA GetDecodeSchedMetaParams ABI mismatch.");

using FlashMlaSparseDecodeKernelFn = void (*)(const FlashMlaSparseDecodeParams &);
using FlashMlaCombineBf16Fn = void (*)(FlashMlaCombineParams &);
using FlashMlaSparseDecodeMetadataFn = void (*)(FlashMlaGetDecodeSchedMetaParams &);

struct FlashMlaSparseDecodeOutWorkspaceFns {
    FlashMlaSparseDecodeKernelFn model1_h16;
    FlashMlaSparseDecodeKernelFn model1_h64;
    FlashMlaSparseDecodeKernelFn model1_h128;
    FlashMlaSparseDecodeKernelFn v32_h16;
    FlashMlaSparseDecodeKernelFn v32_h64;
    FlashMlaSparseDecodeKernelFn v32_h128;
    FlashMlaCombineBf16Fn combine_bf16;
};

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
        if (symtab.sh_offset + symtab.sh_size > data.size() || strtab.sh_offset + strtab.sh_size > data.size() || symtab.sh_entsize != sizeof(Elf64_Sym)) {
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

void *resolve_flashmla_so_symbol(const char *symbol, const char *op_name) {
    if (void *fn = dlsym(RTLD_DEFAULT, symbol)) {
        return fn;
    }

    const char *so_path = std::getenv("INFINICORE_DSV4_FLASHMLA_SO");
    if (so_path == nullptr || so_path[0] == '\0') {
        so_path = kDefaultFlashMlaSoPath;
    }

    void *handle = dlopen(so_path, RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        const char *err = dlerror();
        throw std::runtime_error(std::string(op_name) + " requires flash_mla.cuda.so; failed to dlopen " + so_path + (err == nullptr ? "" : std::string(": ") + err));
    }
    if (void *fn = dlsym(handle, symbol)) {
        return fn;
    }

    void *anchor = dlsym(handle, kFlashMlaAnchorSymbol);
    if (anchor == nullptr) {
        throw std::runtime_error(std::string(op_name) + " requires flash_mla.cuda.so anchor symbol: " + kFlashMlaAnchorSymbol);
    }

    Dl_info info;
    if (dladdr(anchor, &info) == 0 || info.dli_fbase == nullptr) {
        throw std::runtime_error(std::string(op_name) + " failed to resolve flash_mla SO load base.");
    }
    const std::string loaded_path = info.dli_fname == nullptr ? std::string(so_path) : std::string(info.dli_fname);
    const uintptr_t symbol_value = find_elf_symbol_value(loaded_path, symbol);
    return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(info.dli_fbase) + symbol_value);
}

void *resolve_flashmla_sparse_decode(const char *op_name) {
    return resolve_flashmla_so_symbol(kFlashMlaSparseDecodeInterfaceSymbol, op_name);
}

const FlashMlaSparseDecodeOutWorkspaceFns &resolve_flashmla_sparse_decode_out_workspace(const char *op_name) {
    static const FlashMlaSparseDecodeOutWorkspaceFns fns{
        reinterpret_cast<FlashMlaSparseDecodeKernelFn>(
            resolve_flashmla_so_symbol(kFlashMlaSparseDecodeModel1H16Symbol, op_name)),
        reinterpret_cast<FlashMlaSparseDecodeKernelFn>(
            resolve_flashmla_so_symbol(kFlashMlaSparseDecodeModel1H64Symbol, op_name)),
        reinterpret_cast<FlashMlaSparseDecodeKernelFn>(
            resolve_flashmla_so_symbol(kFlashMlaSparseDecodeModel1H128Symbol, op_name)),
        reinterpret_cast<FlashMlaSparseDecodeKernelFn>(
            resolve_flashmla_so_symbol(kFlashMlaSparseDecodeV32H16Symbol, op_name)),
        reinterpret_cast<FlashMlaSparseDecodeKernelFn>(
            resolve_flashmla_so_symbol(kFlashMlaSparseDecodeV32H64Symbol, op_name)),
        reinterpret_cast<FlashMlaSparseDecodeKernelFn>(
            resolve_flashmla_so_symbol(kFlashMlaSparseDecodeV32H128Symbol, op_name)),
        reinterpret_cast<FlashMlaCombineBf16Fn>(
            resolve_flashmla_so_symbol(kFlashMlaCombineBf16Symbol, op_name)),
    };
    return fns;
}

FlashMlaSparseDecodeMetadataFn resolve_flashmla_sparse_decode_metadata(const char *op_name) {
    static auto fn = reinterpret_cast<FlashMlaSparseDecodeMetadataFn>(
        resolve_flashmla_so_symbol(kFlashMlaSparseDecodeMetadataSymbol, op_name));
    return fn;
}

#endif

} // namespace

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
    int extra_page_size,
    FlashMlaSparseCaptureOwners *capture_owners = nullptr) {
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
    if (capture_owners != nullptr) {
        capture_owners->q = q_at_original;
        capture_owners->q_flash = q_flash;
        capture_owners->k_cache = k_cache_fp8;
        capture_owners->indices = indices_flash;
        capture_owners->topk_lengths = topk_lengths_flat;
        capture_owners->attn_sink = attn_sink_opt.has_value() ? attn_sink_opt.value() : at::Tensor();
        capture_owners->tile_scheduler_metadata = tile_scheduler_metadata_at.has_value() ? tile_scheduler_metadata_at.value() : at::Tensor();
        capture_owners->num_splits = num_splits_at.has_value() ? num_splits_at.value() : at::Tensor();
        capture_owners->extra_k_cache = extra_k_cache_opt.has_value() ? extra_k_cache_opt.value() : at::Tensor();
        capture_owners->extra_indices = extra_indices_opt.has_value() ? extra_indices_opt.value() : at::Tensor();
        capture_owners->extra_topk_lengths = extra_topk_lengths_opt.has_value() ? extra_topk_lengths_opt.value() : at::Tensor();
    }
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
    if (capture_owners != nullptr) {
        capture_owners->out = std::get<0>(flash_out);
        capture_owners->lse = std::get<1>(flash_out);
    }
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
    (void)capture_owners;
    throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}

void deepseek_v4_flashmla_sparse_attention_out_workspace_impl(
    const Tensor &q,
    const Tensor &raw_cache,
    const Tensor &indices,
    const Tensor &topk_lengths,
    std::optional<Tensor> attn_sink,
    Tensor output,
    Tensor lse,
    Tensor lse_accum,
    Tensor o_accum,
    Tensor tile_scheduler_metadata,
    Tensor num_splits,
    float softmax_scale,
    int page_size,
    int head_dim_v,
    std::optional<Tensor> extra_raw_cache,
    std::optional<Tensor> extra_indices,
    std::optional<Tensor> extra_topk_lengths,
    int extra_page_size) {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    constexpr const char *op_name = "deepseek_v4_flashmla_sparse_attention_out_workspace_";
    check_hygon_or_nvidia_tensor(q, op_name);
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());

    check_sparse_attention_shapes(q, raw_cache, indices, topk_lengths, output, page_size, head_dim_v);
    const bool has_extra = extra_raw_cache.has_value() || extra_indices.has_value() || extra_topk_lengths.has_value();
    if (has_extra && !(extra_raw_cache.has_value() && extra_indices.has_value() && extra_topk_lengths.has_value())) {
        throw std::runtime_error(std::string(op_name) + " requires all extra cache tensors when any extra tensor is provided.");
    }
    if (has_extra && extra_page_size <= 0) {
        throw std::runtime_error(std::string(op_name) + " requires positive extra_page_size for extra cache.");
    }
    auto require_contiguous = [](const Tensor &tensor, const char *name) {
        if (!tensor->is_contiguous()) {
            throw std::runtime_error(std::string("deepseek_v4_flashmla_sparse_attention_out_workspace_ expects contiguous ") + name + ".");
        }
    };
    auto require_dtype = [](const Tensor &tensor, DataType dtype, const char *name) {
        if (tensor->dtype() != dtype) {
            throw std::runtime_error(std::string("deepseek_v4_flashmla_sparse_attention_out_workspace_ expects ") + name + " to have the pre-normalized dtype required by FlashMLA.");
        }
    };

    require_contiguous(q, "q");
    require_contiguous(raw_cache, "raw_cache");
    require_contiguous(indices, "indices");
    require_contiguous(topk_lengths, "topk_lengths");
    require_contiguous(output, "output");
    require_contiguous(lse, "lse");
    require_contiguous(lse_accum, "lse_accum");
    require_contiguous(o_accum, "o_accum");
    require_contiguous(tile_scheduler_metadata, "tile_scheduler_metadata");
    require_contiguous(num_splits, "num_splits");
    require_dtype(indices, DataType::I32, "indices");
    require_dtype(topk_lengths, DataType::I32, "topk_lengths");
    require_dtype(lse, DataType::F32, "lse");
    require_dtype(lse_accum, DataType::F32, "lse_accum");
    require_dtype(o_accum, DataType::F32, "o_accum");
    require_dtype(tile_scheduler_metadata, DataType::I32, "tile_scheduler_metadata");
    require_dtype(num_splits, DataType::I32, "num_splits");
    if (tile_scheduler_metadata->ndim() != 2 || tile_scheduler_metadata->size(1) != 8) {
        throw std::runtime_error(std::string(op_name) + " expects tile_scheduler_metadata [num_sm_parts, 8] int32.");
    }

    auto q_at_original = infinicore::adaptor::to_aten_tensor(q);
    auto raw_cache_at = infinicore::adaptor::to_aten_tensor(raw_cache);
    auto indices_at = infinicore::adaptor::to_aten_tensor(indices);
    auto topk_lengths_at = infinicore::adaptor::to_aten_tensor(topk_lengths);
    auto output_at = infinicore::adaptor::to_aten_tensor(output);
    auto lse_at = infinicore::adaptor::to_aten_tensor(lse);
    auto lse_accum_at = infinicore::adaptor::to_aten_tensor(lse_accum);
    auto o_accum_at = infinicore::adaptor::to_aten_tensor(o_accum);
    auto tile_scheduler_metadata_at = infinicore::adaptor::to_aten_tensor(tile_scheduler_metadata);
    auto num_splits_at = infinicore::adaptor::to_aten_tensor(num_splits);

    const bool q_was_3d = q_at_original.dim() == 3;
    const int64_t tokens = q_was_3d ? q_at_original.size(0) : q_at_original.size(0) * q_at_original.size(1);
    const int64_t heads = q_at_original.size(q_at_original.dim() - 2);
    if (tokens == 0) {
        return;
    }
    if (num_splits->numel() != static_cast<size_t>(tokens + 1)) {
        throw std::runtime_error(std::string(op_name) + " expects num_splits numel == flattened_tokens + 1.");
    }

    auto q_flash = q_at_original.reshape({tokens, heads, kDsv4FlashMlaQDim}).unsqueeze(1);
    auto output_flash = output_at.reshape({tokens, heads, static_cast<int64_t>(head_dim_v)}).unsqueeze(1);
    if (lse_at.numel() != tokens * heads) {
        throw std::runtime_error(std::string(op_name) + " expects lse numel == flattened_tokens * heads.");
    }
    auto lse_flash = lse_at.reshape({tokens, 1, heads});
    if (lse_accum_at.numel() % heads != 0) {
        throw std::runtime_error(std::string(op_name) + " expects lse_accum numel divisible by heads.");
    }
    const int64_t total_num_splits = lse_accum_at.numel() / heads;
    if (total_num_splits <= 0 || o_accum_at.numel() != total_num_splits * heads * static_cast<int64_t>(head_dim_v)) {
        throw std::runtime_error(std::string(op_name) + " o_accum shape is inconsistent with lse_accum and head_dim_v.");
    }
    auto lse_accum_flash = lse_accum_at.reshape({total_num_splits, 1, heads});
    auto o_accum_flash = o_accum_at.reshape({total_num_splits, 1, heads, static_cast<int64_t>(head_dim_v)});

    auto reshape_indices = [](at::Tensor meta, int64_t token_count, const char *name) -> at::Tensor {
        if (meta.dim() == 2) {
            if (meta.size(0) != token_count) {
                throw std::runtime_error(std::string("deepseek_v4_flashmla_sparse_attention_out_workspace_ ") + name + " token count mismatch.");
            }
            return meta.reshape({token_count, 1, meta.size(1)});
        }
        if (meta.dim() == 3) {
            if (meta.size(0) * meta.size(1) != token_count) {
                throw std::runtime_error(std::string("deepseek_v4_flashmla_sparse_attention_out_workspace_ ") + name + " token count mismatch.");
            }
            return meta.reshape({token_count, 1, meta.size(2)});
        }
        throw std::runtime_error(std::string("deepseek_v4_flashmla_sparse_attention_out_workspace_ expects ") + name + " rank 2 or 3.");
    };

    auto view_raw_cache = [](const at::Tensor &cache_at, int cache_page_size, const char *name) -> at::Tensor {
        if (cache_at.dim() != 2) {
            throw std::runtime_error(std::string("deepseek_v4_flashmla_sparse_attention_out_workspace_ expects ") + name + " [blocks, page_bytes].");
        }
        const int64_t expected_page_bytes = dsv4_flashmla_page_bytes(cache_page_size);
        if (cache_at.size(1) != expected_page_bytes) {
            throw std::runtime_error(std::string("deepseek_v4_flashmla_sparse_attention_out_workspace_ ") + name + " page_bytes mismatch.");
        }
        const int64_t cache_bytes_per_page = static_cast<int64_t>(cache_page_size) * kDsv4FlashMlaBytesPerToken;
        auto cache_bytes = cache_at.slice(1, 0, cache_bytes_per_page);
        return cache_bytes.view(at::ScalarType::Float8_e4m3fn)
            .reshape({cache_at.size(0), static_cast<int64_t>(cache_page_size), 1, kDsv4FlashMlaBytesPerToken});
    };

    auto k_cache_fp8 = view_raw_cache(raw_cache_at, page_size, "raw_cache");
    auto indices_flash = reshape_indices(indices_at, tokens, "indices");
    if (topk_lengths_at.numel() != tokens) {
        throw std::runtime_error(std::string(op_name) + " topk_lengths token count mismatch.");
    }
    auto topk_lengths_flat = topk_lengths_at.reshape({tokens});

    std::optional<at::Tensor> attn_sink_opt = std::nullopt;
    at::Tensor attn_sink_storage;
    if (attn_sink.has_value()) {
        require_contiguous(attn_sink.value(), "attn_sink");
        require_dtype(attn_sink.value(), DataType::F32, "attn_sink");
        attn_sink_storage = infinicore::adaptor::to_aten_tensor(attn_sink.value());
        if (attn_sink_storage.numel() < heads) {
            throw std::runtime_error(std::string(op_name) + " attn_sink has fewer heads than q.");
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
        require_contiguous(extra_raw_cache.value(), "extra_raw_cache");
        require_contiguous(extra_indices.value(), "extra_indices");
        require_contiguous(extra_topk_lengths.value(), "extra_topk_lengths");
        require_dtype(extra_indices.value(), DataType::I32, "extra_indices");
        require_dtype(extra_topk_lengths.value(), DataType::I32, "extra_topk_lengths");
        extra_k_cache_storage = view_raw_cache(infinicore::adaptor::to_aten_tensor(extra_raw_cache.value()), extra_page_size, "extra_raw_cache");
        extra_indices_storage = reshape_indices(infinicore::adaptor::to_aten_tensor(extra_indices.value()), tokens, "extra_indices");
        extra_topk_lengths_storage = infinicore::adaptor::to_aten_tensor(extra_topk_lengths.value()).reshape({tokens});
        if (extra_topk_lengths_storage.numel() != tokens) {
            throw std::runtime_error(std::string(op_name) + " extra_topk_lengths token count mismatch.");
        }
        extra_k_cache_opt = extra_k_cache_storage;
        extra_indices_opt = extra_indices_storage;
        extra_topk_lengths_opt = extra_topk_lengths_storage;
    }

    auto to_i32 = [](int64_t value, const char *name) -> int {
        if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
            throw std::runtime_error(std::string("deepseek_v4_flashmla_sparse_attention_out_workspace_ ") + name + " does not fit in int32.");
        }
        return static_cast<int>(value);
    };
    const int num_sm_parts = to_i32(tile_scheduler_metadata_at.size(0), "num_sm_parts");
    const int required_total_num_splits = to_i32(tokens + num_sm_parts, "required_total_num_splits");
    if (total_num_splits < required_total_num_splits) {
        throw std::runtime_error(std::string(op_name) + " expects lse_accum/o_accum first dimension >= flattened_tokens + num_sm_parts.");
    }
    if (q_flash.size(3) != 512 && q_flash.size(3) != 576) {
        throw std::runtime_error(std::string(op_name) + " supports q head dim 512 or 576 only.");
    }
    const auto model_type = q_flash.size(3) == 576 ? FlashMlaModelType::V32 : FlashMlaModelType::MODEL1;
    const auto &flash_mla = resolve_flashmla_sparse_decode_out_workspace(op_name);
    FlashMlaSparseDecodeKernelFn sparse_decode_kernel = nullptr;
    if (model_type == FlashMlaModelType::MODEL1) {
        if (heads <= 16) {
            sparse_decode_kernel = flash_mla.model1_h16;
        } else if (heads == 64) {
            sparse_decode_kernel = flash_mla.model1_h64;
        } else if (heads == 128) {
            sparse_decode_kernel = flash_mla.model1_h128;
        }
    } else {
        if (heads <= 16) {
            sparse_decode_kernel = flash_mla.v32_h16;
        } else if (heads == 64) {
            sparse_decode_kernel = flash_mla.v32_h64;
        } else if (heads == 128) {
            sparse_decode_kernel = flash_mla.v32_h128;
        }
    }
    if (sparse_decode_kernel == nullptr) {
        throw std::runtime_error(std::string(op_name) + " supports FlashMLA local head counts up to 16, or exactly 64/128.");
    }

    const int extra_num_blocks = extra_k_cache_opt.has_value() ? to_i32(extra_k_cache_opt.value().size(0), "extra_num_blocks") : 0;
    const int extra_page_block_size = extra_k_cache_opt.has_value() ? to_i32(extra_k_cache_opt.value().size(1), "extra_page_block_size") : 0;
    const int extra_topk = extra_indices_opt.has_value() ? to_i32(extra_indices_opt.value().size(2), "extra_topk") : 0;
    auto *stream = context::getStream();

    FlashMlaSparseDecodeParams params{
        to_i32(tokens, "b"),
        1,
        to_i32(heads, "h_q"),
        1,
        to_i32(q_flash.size(3), "d_qk"),
        head_dim_v,
        softmax_scale,
        softmax_scale * kFlashMlaLog2E,
        to_i32(k_cache_fp8.size(0), "num_blocks"),
        to_i32(k_cache_fp8.size(1), "page_block_size"),
        to_i32(indices_flash.size(2), "topk"),
        model_type,

        q_flash.data_ptr(),
        k_cache_fp8.data_ptr(),
        indices_flash.data_ptr<int>(),
        topk_lengths_flat.data_ptr<int>(),
        attn_sink_opt.has_value() ? attn_sink_opt.value().data_ptr<float>() : nullptr,

        lse_flash.data_ptr<float>(),
        output_flash.data_ptr(),

        extra_num_blocks,
        extra_page_block_size,
        extra_topk,
        extra_k_cache_opt.has_value() ? extra_k_cache_opt.value().data_ptr() : nullptr,
        extra_indices_opt.has_value() ? extra_indices_opt.value().data_ptr<int>() : nullptr,
        extra_topk_lengths_opt.has_value() ? extra_topk_lengths_opt.value().data_ptr<int>() : nullptr,

        to_i32(q_flash.stride(0), "stride_q_b"),
        to_i32(q_flash.stride(1), "stride_q_s_q"),
        to_i32(q_flash.stride(2), "stride_q_h_q"),
        to_i32(k_cache_fp8.stride(0), "stride_kv_block"),
        to_i32(k_cache_fp8.stride(1), "stride_kv_row"),
        to_i32(indices_flash.stride(0), "stride_indices_b"),
        to_i32(indices_flash.stride(1), "stride_indices_s_q"),
        to_i32(lse_flash.stride(0), "stride_lse_b"),
        to_i32(lse_flash.stride(1), "stride_lse_s_q"),
        to_i32(output_flash.stride(0), "stride_o_b"),
        to_i32(output_flash.stride(1), "stride_o_s_q"),
        to_i32(output_flash.stride(2), "stride_o_h_q"),
        extra_k_cache_opt.has_value() ? to_i32(extra_k_cache_opt.value().stride(0), "stride_extra_kv_block") : 0,
        extra_k_cache_opt.has_value() ? to_i32(extra_k_cache_opt.value().stride(1), "stride_extra_kv_row") : 0,
        extra_indices_opt.has_value() ? to_i32(extra_indices_opt.value().stride(0), "stride_extra_indices_b") : 0,
        extra_indices_opt.has_value() ? to_i32(extra_indices_opt.value().stride(1), "stride_extra_indices_s_q") : 0,

        stream,

        lse_accum_flash.data_ptr<float>(),
        o_accum_flash.data_ptr<float>(),
        to_i32(lse_accum_flash.stride(0), "stride_lse_accum_split"),
        to_i32(lse_accum_flash.stride(1), "stride_lse_accum_s_q"),
        to_i32(o_accum_flash.stride(0), "stride_o_accum_split"),
        to_i32(o_accum_flash.stride(1), "stride_o_accum_s_q"),
        to_i32(o_accum_flash.stride(2), "stride_o_accum_h_q"),
        reinterpret_cast<FlashMlaDecodingSchedMeta *>(tile_scheduler_metadata_at.data_ptr<int>()),
        num_splits_at.data_ptr<int>(),
        num_sm_parts,
    };
    sparse_decode_kernel(params);

    FlashMlaCombineParams combine_params{
        params.b,
        params.s_q,
        params.h_q,
        params.d_v,

        params.lse,
        params.out,
        params.stride_lse_b,
        params.stride_lse_s_q,
        params.stride_o_b,
        params.stride_o_s_q,
        params.stride_o_h_q,

        params.lse_accum,
        params.o_accum,
        params.stride_lse_accum_split,
        params.stride_lse_accum_s_q,
        params.stride_o_accum_split,
        params.stride_o_accum_s_q,
        params.stride_o_accum_h_q,

        params.tile_scheduler_metadata_ptr,
        params.num_splits_ptr,
        params.num_sm_parts,

        params.attn_sink,

        stream,
        false,
        0,
        nullptr,
        0,
    };
    flash_mla.combine_bf16(combine_params);
#else
    (void)q;
    (void)raw_cache;
    (void)indices;
    (void)topk_lengths;
    (void)attn_sink;
    (void)output;
    (void)lse;
    (void)lse_accum;
    (void)o_accum;
    (void)tile_scheduler_metadata;
    (void)num_splits;
    (void)softmax_scale;
    (void)page_size;
    (void)head_dim_v;
    (void)extra_raw_cache;
    (void)extra_indices;
    (void)extra_topk_lengths;
    (void)extra_page_size;
    throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_out_workspace_ requires an ATen-enabled HYGON build.");
#endif
}

void deepseek_v4_flashmla_sparse_attention_metadata_impl(Tensor tile_scheduler_metadata,
                                                         Tensor num_splits,
                                                         const Tensor &topk_lengths,
                                                         int topk,
                                                         std::optional<Tensor> extra_topk_lengths,
                                                         int extra_topk) {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    constexpr const char *op_name = "deepseek_v4_flashmla_sparse_attention_metadata_";
    check_hygon_or_nvidia_tensor(tile_scheduler_metadata, op_name);
    auto require_contiguous = [](const Tensor &tensor, const char *name) {
        if (!tensor->is_contiguous()) {
            throw std::runtime_error(std::string("deepseek_v4_flashmla_sparse_attention_metadata_ expects contiguous ") + name + ".");
        }
    };
    auto require_i32 = [](const Tensor &tensor, const char *name) {
        if (tensor->dtype() != DataType::I32) {
            throw std::runtime_error(std::string("deepseek_v4_flashmla_sparse_attention_metadata_ expects ") + name + " dtype int32.");
        }
    };
    auto to_i32 = [](int64_t value, const char *name) -> int {
        if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
            throw std::runtime_error(std::string("deepseek_v4_flashmla_sparse_attention_metadata_ ") + name + " does not fit in int32.");
        }
        return static_cast<int>(value);
    };

    require_contiguous(tile_scheduler_metadata, "tile_scheduler_metadata");
    require_contiguous(num_splits, "num_splits");
    require_contiguous(topk_lengths, "topk_lengths");
    require_i32(tile_scheduler_metadata, "tile_scheduler_metadata");
    require_i32(num_splits, "num_splits");
    require_i32(topk_lengths, "topk_lengths");
    if (tile_scheduler_metadata->ndim() != 2 || tile_scheduler_metadata->size(1) != 8) {
        throw std::runtime_error(std::string(op_name) + " expects tile_scheduler_metadata [num_sm_parts, 8] int32.");
    }
    if (num_splits->ndim() != 1) {
        throw std::runtime_error(std::string(op_name) + " expects num_splits [tokens + 1] int32.");
    }
    const int tokens = to_i32(topk_lengths->numel(), "tokens");
    if (tokens <= 0) {
        return;
    }
    if (num_splits->numel() != static_cast<size_t>(tokens + 1)) {
        throw std::runtime_error(std::string(op_name) + " expects num_splits numel == topk_lengths.numel() + 1.");
    }
    if (topk <= 0) {
        throw std::runtime_error(std::string(op_name) + " expects positive topk.");
    }

    int *extra_topk_length_ptr = nullptr;
    if (extra_topk_lengths.has_value() && extra_topk_lengths.value()) {
        require_contiguous(extra_topk_lengths.value(), "extra_topk_lengths");
        require_i32(extra_topk_lengths.value(), "extra_topk_lengths");
        if (extra_topk_lengths.value()->numel() != topk_lengths->numel()) {
            throw std::runtime_error(std::string(op_name) + " extra_topk_lengths token count mismatch.");
        }
        if (extra_topk <= 0) {
            throw std::runtime_error(std::string(op_name) + " expects positive extra_topk when extra_topk_lengths is provided.");
        }
        extra_topk_length_ptr = reinterpret_cast<int *>(extra_topk_lengths.value()->data());
    } else {
        extra_topk = -1;
    }

    FlashMlaGetDecodeSchedMetaParams params{
        tokens,
        1,
        64,
        5,
        topk,
        extra_topk,
        const_cast<int *>(reinterpret_cast<const int *>(topk_lengths->data())),
        extra_topk_length_ptr,
        nullptr,
        reinterpret_cast<FlashMlaDecodingSchedMeta *>(tile_scheduler_metadata->data()),
        reinterpret_cast<int *>(num_splits->data()),
        to_i32(tile_scheduler_metadata->size(0), "num_sm_parts"),
        context::getStream(),
    };
    auto fn = resolve_flashmla_sparse_decode_metadata(op_name);
    fn(params);
#else
    (void)tile_scheduler_metadata;
    (void)num_splits;
    (void)topk_lengths;
    (void)topk;
    (void)extra_topk_lengths;
    (void)extra_topk;
    throw std::runtime_error("deepseek_v4_flashmla_sparse_attention_metadata_ requires an ATen-enabled HYGON build.");
#endif
}

DeepseekV4FlashMlaSparseAttentionMetadata::DeepseekV4FlashMlaSparseAttentionMetadata(
    Tensor tile_scheduler_metadata,
    Tensor num_splits,
    const Tensor &topk_lengths,
    int topk,
    std::optional<Tensor> extra_topk_lengths,
    int extra_topk) {
    device_graph_capture_supported_ = true;
    INFINICORE_GRAPH_OP_DISPATCH(tile_scheduler_metadata->device().getType(),
                                 tile_scheduler_metadata,
                                 num_splits,
                                 topk_lengths,
                                 topk,
                                 extra_topk_lengths,
                                 extra_topk);
}

void DeepseekV4FlashMlaSparseAttentionMetadata::execute(Tensor tile_scheduler_metadata,
                                                        Tensor num_splits,
                                                        const Tensor &topk_lengths,
                                                        int topk,
                                                        std::optional<Tensor> extra_topk_lengths,
                                                        int extra_topk) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4FlashMlaSparseAttentionMetadata,
                                      tile_scheduler_metadata,
                                      num_splits,
                                      topk_lengths,
                                      topk,
                                      extra_topk_lengths,
                                      extra_topk);
}

namespace deepseek_v4_flashmla_sparse_attention_metadata_graph_impl {

struct PlannedMeta {
    graph::GraphTensor tile_scheduler_metadata;
    graph::GraphTensor num_splits;
    graph::GraphTensor topk_lengths;
    int topk;
    std::optional<graph::GraphTensor> extra_topk_lengths;
    int extra_topk;
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

void *plan(Tensor tile_scheduler_metadata,
           Tensor num_splits,
           const Tensor &topk_lengths,
           int topk,
           std::optional<Tensor> extra_topk_lengths,
           int extra_topk) {
    check_hygon_or_nvidia_tensor(tile_scheduler_metadata, "deepseek_v4_flashmla_sparse_attention_metadata_");
    return new PlannedMeta{graph::GraphTensor(tile_scheduler_metadata),
                           graph::GraphTensor(num_splits),
                           graph::GraphTensor(topk_lengths),
                           topk,
                           to_graph_optional(extra_topk_lengths),
                           extra_topk};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    deepseek_v4_flashmla_sparse_attention_metadata_impl(planned->tile_scheduler_metadata,
                                                        planned->num_splits,
                                                        planned->topk_lengths,
                                                        planned->topk,
                                                        to_tensor_optional(planned->extra_topk_lengths),
                                                        planned->extra_topk);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace deepseek_v4_flashmla_sparse_attention_metadata_graph_impl

namespace deepseek_v4_flashmla_sparse_attention_metadata_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4FlashMlaSparseAttentionMetadata,
                                       &deepseek_v4_flashmla_sparse_attention_metadata_graph_impl::plan,
                                       &deepseek_v4_flashmla_sparse_attention_metadata_graph_impl::run,
                                       &deepseek_v4_flashmla_sparse_attention_metadata_graph_impl::cleanup);
} // namespace deepseek_v4_flashmla_sparse_attention_metadata_register

void deepseek_v4_flashmla_sparse_attention_metadata_(Tensor tile_scheduler_metadata,
                                                     Tensor num_splits,
                                                     const Tensor &topk_lengths,
                                                     int topk,
                                                     std::optional<Tensor> extra_topk_lengths,
                                                     int extra_topk) {
    DeepseekV4FlashMlaSparseAttentionMetadata::execute(tile_scheduler_metadata,
                                                       num_splits,
                                                       topk_lengths,
                                                       topk,
                                                       extra_topk_lengths,
                                                       extra_topk);
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
    // FlashMLA still goes through an ATen/vendor-SO bridge. Capture is
    // experimental until the bridge exposes all replay-time workspaces.
    const char *enable_capture = std::getenv("INFINICORE_DSV4_FLASHMLA_ENABLE_DEVICE_GRAPH");
    device_graph_capture_supported_ = enable_capture != nullptr && (std::strcmp(enable_capture, "1") == 0 || std::strcmp(enable_capture, "true") == 0 || std::strcmp(enable_capture, "TRUE") == 0 || std::strcmp(enable_capture, "on") == 0 || std::strcmp(enable_capture, "ON") == 0);
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
    FlashMlaSparseCaptureOwners capture_owners;
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
    if (!tile_scheduler_metadata.has_value() || !tile_scheduler_metadata.value() || !num_splits.has_value() || !num_splits.value()) {
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
                           FlashMlaSparseCaptureOwners{},
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
                                                                   planned->extra_page_size,
                                                                   &planned->capture_owners);
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

DeepseekV4FlashMlaSparseAttentionOutWorkspace::DeepseekV4FlashMlaSparseAttentionOutWorkspace(
    const Tensor &q,
    const Tensor &raw_cache,
    const Tensor &indices,
    const Tensor &topk_lengths,
    std::optional<Tensor> attn_sink,
    Tensor output,
    Tensor lse,
    Tensor lse_accum,
    Tensor o_accum,
    Tensor tile_scheduler_metadata,
    Tensor num_splits,
    float softmax_scale,
    int page_size,
    int head_dim_v,
    std::optional<Tensor> extra_raw_cache,
    std::optional<Tensor> extra_indices,
    std::optional<Tensor> extra_topk_lengths,
    int extra_page_size) {
    device_graph_capture_supported_ = true;
    INFINICORE_GRAPH_OP_DISPATCH(q->device().getType(),
                                 q,
                                 raw_cache,
                                 indices,
                                 topk_lengths,
                                 attn_sink,
                                 output,
                                 lse,
                                 lse_accum,
                                 o_accum,
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

void DeepseekV4FlashMlaSparseAttentionOutWorkspace::execute(
    const Tensor &q,
    const Tensor &raw_cache,
    const Tensor &indices,
    const Tensor &topk_lengths,
    std::optional<Tensor> attn_sink,
    Tensor output,
    Tensor lse,
    Tensor lse_accum,
    Tensor o_accum,
    Tensor tile_scheduler_metadata,
    Tensor num_splits,
    float softmax_scale,
    int page_size,
    int head_dim_v,
    std::optional<Tensor> extra_raw_cache,
    std::optional<Tensor> extra_indices,
    std::optional<Tensor> extra_topk_lengths,
    int extra_page_size) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(DeepseekV4FlashMlaSparseAttentionOutWorkspace,
                                      q,
                                      raw_cache,
                                      indices,
                                      topk_lengths,
                                      attn_sink,
                                      output,
                                      lse,
                                      lse_accum,
                                      o_accum,
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

namespace deepseek_v4_flashmla_sparse_attention_out_workspace_graph_impl {

using deepseek_v4_flashmla_sparse_attention_graph_impl::to_graph_optional;
using deepseek_v4_flashmla_sparse_attention_graph_impl::to_tensor_optional;

struct PlannedMeta {
    graph::GraphTensor q; // [ntokens, num_attention_heads , head_dim]
    graph::GraphTensor raw_cache;
    graph::GraphTensor indices;
    graph::GraphTensor topk_lengths;
    std::optional<graph::GraphTensor> attn_sink;
    graph::GraphTensor output;
    graph::GraphTensor lse;
    graph::GraphTensor lse_accum;
    graph::GraphTensor o_accum;
    graph::GraphTensor tile_scheduler_metadata;
    graph::GraphTensor num_splits;
    float softmax_scale;
    int page_size;
    int head_dim_v;
    std::optional<graph::GraphTensor> extra_raw_cache;
    std::optional<graph::GraphTensor> extra_indices;
    std::optional<graph::GraphTensor> extra_topk_lengths;
    int extra_page_size;
};

void *plan(const Tensor &q,
           const Tensor &raw_cache,
           const Tensor &indices,
           const Tensor &topk_lengths,
           std::optional<Tensor> attn_sink,
           Tensor output,
           Tensor lse,
           Tensor lse_accum,
           Tensor o_accum,
           Tensor tile_scheduler_metadata,
           Tensor num_splits,
           float softmax_scale,
           int page_size,
           int head_dim_v,
           std::optional<Tensor> extra_raw_cache,
           std::optional<Tensor> extra_indices,
           std::optional<Tensor> extra_topk_lengths,
           int extra_page_size) {
    check_hygon_or_nvidia_tensor(q, "deepseek_v4_flashmla_sparse_attention_out_workspace_");
    check_sparse_attention_shapes(q, raw_cache, indices, topk_lengths, output, page_size, head_dim_v);
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    (void)resolve_flashmla_sparse_decode_out_workspace("deepseek_v4_flashmla_sparse_attention_out_workspace_");
#endif
    return new PlannedMeta{graph::GraphTensor(q),
                           graph::GraphTensor(raw_cache),
                           graph::GraphTensor(indices),
                           graph::GraphTensor(topk_lengths),
                           to_graph_optional(attn_sink),
                           graph::GraphTensor(output),
                           graph::GraphTensor(lse),
                           graph::GraphTensor(lse_accum),
                           graph::GraphTensor(o_accum),
                           graph::GraphTensor(tile_scheduler_metadata),
                           graph::GraphTensor(num_splits),
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
    deepseek_v4_flashmla_sparse_attention_out_workspace_impl(planned->q,
                                                             planned->raw_cache,
                                                             planned->indices,
                                                             planned->topk_lengths,
                                                             to_tensor_optional(planned->attn_sink),
                                                             planned->output,
                                                             planned->lse,
                                                             planned->lse_accum,
                                                             planned->o_accum,
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

} // namespace deepseek_v4_flashmla_sparse_attention_out_workspace_graph_impl

namespace deepseek_v4_flashmla_sparse_attention_out_workspace_register {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekV4FlashMlaSparseAttentionOutWorkspace,
                                       &deepseek_v4_flashmla_sparse_attention_out_workspace_graph_impl::plan,
                                       &deepseek_v4_flashmla_sparse_attention_out_workspace_graph_impl::run,
                                       &deepseek_v4_flashmla_sparse_attention_out_workspace_graph_impl::cleanup);
} // namespace deepseek_v4_flashmla_sparse_attention_out_workspace_register

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
    const bool has_schedule = tile_scheduler_metadata.has_value() && tile_scheduler_metadata.value() && num_splits.has_value() && num_splits.value();
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

void deepseek_v4_flashmla_sparse_attention_out_workspace_(
    const Tensor &q,
    const Tensor &raw_cache,
    const Tensor &indices,
    const Tensor &topk_lengths,
    std::optional<Tensor> attn_sink,
    Tensor output,
    Tensor lse,
    Tensor lse_accum,
    Tensor o_accum,
    Tensor tile_scheduler_metadata,
    Tensor num_splits,
    float softmax_scale,
    int page_size,
    int head_dim_v,
    std::optional<Tensor> extra_raw_cache,
    std::optional<Tensor> extra_indices,
    std::optional<Tensor> extra_topk_lengths,
    int extra_page_size) {
    DeepseekV4FlashMlaSparseAttentionOutWorkspace::execute(q,
                                                           raw_cache,
                                                           indices,
                                                           topk_lengths,
                                                           attn_sink,
                                                           output,
                                                           lse,
                                                           lse_accum,
                                                           o_accum,
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
} // namespace infinicore::op

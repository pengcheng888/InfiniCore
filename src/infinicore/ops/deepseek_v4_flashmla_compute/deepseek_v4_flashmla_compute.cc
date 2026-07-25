#include "infinicore/ops/deepseek_v4_flashmla_compute.hpp"

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


void deepseek_v4_compress_fused_norm_rope_(Tensor input,
                                           const Tensor &norm_weight,
                                           float epsilon,
                                           const Tensor &freqs_cis,
                                           const Tensor &positions) {
#if defined(ENABLE_ATEN) && (defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API))
    check_hygon_or_nvidia_tensor(input, "deepseek_v4_compress_fused_norm_rope_");
    check_hygon_or_nvidia_tensor(norm_weight, "deepseek_v4_compress_fused_norm_rope_");
    check_hygon_or_nvidia_tensor(freqs_cis, "deepseek_v4_compress_fused_norm_rope_");
    check_hygon_or_nvidia_tensor(positions, "deepseek_v4_compress_fused_norm_rope_");
#if defined(ENABLE_HYGON_API)
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    check_compress_fused_norm_rope_shapes(input, norm_weight, freqs_cis, positions);

    auto input_at = infinicore::adaptor::to_aten_tensor(input);
    if (!input_at.is_contiguous()) {
        throw std::runtime_error("deepseek_v4_compress_fused_norm_rope_ expects contiguous input.");
    }
    const int64_t input_dim = input_at.size(1);
    auto weight_at = infinicore::adaptor::to_aten_tensor(norm_weight).to(at::kFloat).reshape({1, input_dim});
    auto input_float = input_at.to(at::kFloat);
    auto variance = (input_float * input_float).mean({-1}, true);
    auto normalized = input_float * at::rsqrt(variance + static_cast<double>(epsilon)) * weight_at;
    input_at.copy_(normalized.to(input_at.scalar_type()));

    auto rope = input_at.slice(1, input_dim - 64, input_dim);
    apply_rope_2d_last64_aten_(rope,
                               infinicore::adaptor::to_aten_tensor(freqs_cis),
                               infinicore::adaptor::to_aten_tensor(positions));
#else
    (void)input;
    (void)norm_weight;
    (void)epsilon;
    (void)freqs_cis;
    (void)positions;
    throw std::runtime_error("deepseek_v4_compress_fused_norm_rope_ requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}


Tensor deepseek_v4_c4_compress_prefill_reference(const Tensor &kv_score_input,
                                                 const Tensor &ape) {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    check_hygon_or_nvidia_tensor(kv_score_input, "deepseek_v4_c4_compress_prefill_reference");
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());

    if (kv_score_input->ndim() != 2 || kv_score_input->size(1) % 4 != 0) {
        throw std::runtime_error("deepseek_v4_c4_compress_prefill_reference expects kv_score_input [tokens, 4 * head_dim].");
    }
    const int64_t tokens = static_cast<int64_t>(kv_score_input->size(0));
    const int64_t head_dim = static_cast<int64_t>(kv_score_input->size(1) / 4);
    if (head_dim != kDsv4FlashMlaQDim) {
        throw std::runtime_error("deepseek_v4_c4_compress_prefill_reference expects head_dim 512.");
    }
    if (ape->ndim() != 2) {
        throw std::runtime_error("deepseek_v4_c4_compress_prefill_reference expects ape rank 2.");
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
        throw std::runtime_error("deepseek_v4_c4_compress_prefill_reference expects ape [4, 1024] or [8, 512].");
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
    throw std::runtime_error("deepseek_v4_c4_compress_prefill_reference requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}


Tensor deepseek_v4_c4_compress_stateful_reference(const Tensor &kv_score_input,
                                                  const Tensor &ape,
                                                  Tensor compressor_state,
                                                  const Tensor &write_loc,
                                                  const Tensor &extra_loc,
                                                  const Tensor &positions) {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    check_hygon_or_nvidia_tensor(kv_score_input, "deepseek_v4_c4_compress_stateful_reference");
    check_hygon_or_nvidia_tensor(compressor_state, "deepseek_v4_c4_compress_stateful_reference");
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());

    if (kv_score_input->ndim() != 2 || kv_score_input->size(1) % 4 != 0) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_reference expects kv_score_input [tokens, 4 * head_dim].");
    }
    const int64_t tokens = static_cast<int64_t>(kv_score_input->size(0));
    const int64_t head_dim = static_cast<int64_t>(kv_score_input->size(1) / 4);
    if (head_dim <= 0) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_reference expects positive head_dim.");
    }
    if (compressor_state->ndim() != 2 || compressor_state->size(1) != static_cast<size_t>(4 * head_dim) || compressor_state->size(0) % 4 != 0) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_reference expects compressor_state [4 * groups, 4 * head_dim].");
    }
    if (write_loc->numel() != static_cast<size_t>(tokens) || positions->numel() != static_cast<size_t>(tokens)) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_reference metadata token count mismatch.");
    }

    auto output = Tensor::zeros({static_cast<size_t>(tokens), static_cast<size_t>(head_dim)}, kv_score_input->dtype(), kv_score_input->device());
    if (tokens == 0) {
        return output;
    }

    auto state_at = infinicore::adaptor::to_aten_tensor(compressor_state);
    if (!state_at.is_contiguous()) {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_reference expects contiguous compressor_state.");
    }
    auto kv_score_at = infinicore::adaptor::to_aten_tensor(kv_score_input).contiguous().to(state_at.scalar_type())
                           .reshape({tokens, 4 * head_dim});
    auto state_groups = state_at.view({static_cast<int64_t>(compressor_state->size(0)) / 4, 4, 4, head_dim});

    auto write_loc_at = infinicore::adaptor::to_aten_tensor(write_loc).reshape({tokens}).to(at::kLong);
    at::Tensor extra_prev_at;
    auto extra_at_raw = infinicore::adaptor::to_aten_tensor(extra_loc).to(at::kLong);
    if (extra_at_raw.dim() == 2) {
        if (extra_at_raw.size(0) != tokens || extra_at_raw.size(1) < 1) {
            throw std::runtime_error("deepseek_v4_c4_compress_stateful_reference expects extra_loc [tokens, >=1].");
        }
        extra_prev_at = extra_at_raw.select(1, 0).reshape({tokens});
    } else if (extra_at_raw.dim() == 1 && extra_at_raw.size(0) == tokens) {
        extra_prev_at = extra_at_raw;
    } else {
        throw std::runtime_error("deepseek_v4_c4_compress_stateful_reference expects extra_loc rank 1 or 2.");
    }
    auto positions_at = infinicore::adaptor::to_aten_tensor(positions).reshape({tokens}).to(at::kLong);
    auto write_pos_at = positions_at.remainder(4);

    auto valid_write = write_loc_at.ge(0);
    auto valid_rows = at::nonzero(valid_write).reshape({-1});
    if (valid_rows.numel() > 0) {
        auto valid_groups = write_loc_at.index_select(0, valid_rows);
        auto valid_write_pos = write_pos_at.index_select(0, valid_rows);
        auto valid_values = kv_score_at.index_select(0, valid_rows).reshape({valid_rows.numel(), 4, head_dim});
        state_groups.index_put_({valid_groups, valid_write_pos}, valid_values);
    }

    auto boundary_mask = valid_write.logical_and((positions_at + 1).remainder(4).eq(0));
    auto boundary_rows = at::nonzero(boundary_mask).reshape({-1});
    if (boundary_rows.numel() > 0) {
        auto groups = write_loc_at.index_select(0, boundary_rows);
        auto prev_groups = extra_prev_at.index_select(0, boundary_rows).clamp_min(0);
        auto boundary_positions = positions_at.index_select(0, boundary_rows);

        auto normal_state = state_groups.index_select(0, groups).to(at::kFloat);
        auto overlap_state = state_groups.index_select(0, prev_groups).to(at::kFloat);

        auto overlap_kv = overlap_state.select(2, 0);
        auto normal_kv = normal_state.select(2, 1);
        auto overlap_score = overlap_state.select(2, 2);
        auto normal_score = normal_state.select(2, 3);

        auto has_overlap = boundary_positions.ge(7).view({-1, 1, 1});
        overlap_kv = at::where(has_overlap, overlap_kv, at::zeros_like(overlap_kv));
        overlap_score = at::where(has_overlap, overlap_score, at::full_like(overlap_score, -1.0e9));

        auto ape_at = infinicore::adaptor::to_aten_tensor(ape).contiguous().to(at::kFloat);
        at::Tensor ape_view;
        if (ape_at.dim() == 2 && ape_at.size(0) == 4 && ape_at.size(1) == 2 * head_dim) {
            auto ape_chunks = ape_at.reshape({4, 2, head_dim});
            ape_view = at::cat({ape_chunks.select(1, 1), ape_chunks.select(1, 0)}, 0).contiguous();
        } else if (ape_at.dim() == 2 && ape_at.size(0) == 8 && ape_at.size(1) == head_dim) {
            ape_view = ape_at;
        } else {
            throw std::runtime_error("deepseek_v4_c4_compress_stateful_reference expects ape [4, 2 * head_dim] or [8, head_dim].");
        }

        auto kv_window = at::cat({overlap_kv, normal_kv}, 1);
        auto score_window = at::cat({overlap_score, normal_score}, 1) + ape_view.view({1, 8, head_dim});
        auto prob = at::softmax(score_window, 1);
        auto compressed = (kv_window * prob).sum(1);
        auto output_at = infinicore::adaptor::to_aten_tensor(output);
        output_at.index_copy_(0, boundary_rows, compressed.to(output_at.scalar_type()));
    }
    return output;
#else
    (void)kv_score_input;
    (void)ape;
    (void)compressor_state;
    (void)write_loc;
    (void)extra_loc;
    (void)positions;
    throw std::runtime_error("deepseek_v4_c4_compress_stateful_reference requires an ATen-enabled HYGON/NVIDIA build.");
#endif
}


Tensor deepseek_v4_c4_compress_stateful(const Tensor &kv_score_input,
                                        const Tensor &ape,
                                        Tensor compressor_state,
                                        const Tensor &write_loc,
                                        const Tensor &extra_loc,
                                        const Tensor &positions) {
    return deepseek_v4_c4_compress_stateful_reference(kv_score_input,
                                                      ape,
                                                      compressor_state,
                                                      write_loc,
                                                      extra_loc,
                                                      positions);
}


Tensor deepseek_v4_c128_compress_stateful_reference(const Tensor &kv_score_input,
                                                    const Tensor &ape,
                                                    Tensor compressor_state,
                                                    const Tensor &write_loc,
                                                    const Tensor &positions) {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    check_hygon_or_nvidia_tensor(kv_score_input, "deepseek_v4_c128_compress_stateful_reference");
    check_hygon_or_nvidia_tensor(compressor_state, "deepseek_v4_c128_compress_stateful_reference");
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());

    if (kv_score_input->ndim() != 2 || kv_score_input->size(1) % 2 != 0) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_reference expects kv_score_input [tokens, 2 * head_dim].");
    }
    const int64_t tokens = static_cast<int64_t>(kv_score_input->size(0));
    const int64_t head_dim = static_cast<int64_t>(kv_score_input->size(1) / 2);
    if (head_dim != kDsv4FlashMlaQDim) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_reference expects head_dim 512.");
    }
    if (compressor_state->ndim() != 2 || compressor_state->size(1) != static_cast<size_t>(2 * head_dim) || compressor_state->size(0) % 128 != 0) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_reference expects compressor_state [128 * groups, 2 * head_dim].");
    }
    if (write_loc->numel() != static_cast<size_t>(tokens) || positions->numel() != static_cast<size_t>(tokens)) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_reference metadata token count mismatch.");
    }
    if (ape->ndim() != 2 || ape->size(1) != static_cast<size_t>(head_dim)) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_reference expects ape [128, head_dim].");
    }

    auto output = Tensor::zeros({static_cast<size_t>(tokens), static_cast<size_t>(head_dim)}, kv_score_input->dtype(), kv_score_input->device());
    if (tokens == 0) {
        return output;
    }

    auto state_at = infinicore::adaptor::to_aten_tensor(compressor_state);
    if (!state_at.is_contiguous()) {
        throw std::runtime_error("deepseek_v4_c128_compress_stateful_reference expects contiguous compressor_state.");
    }
    auto kv_score_at = infinicore::adaptor::to_aten_tensor(kv_score_input).contiguous().to(state_at.scalar_type())
                           .reshape({tokens, 2 * head_dim});
    auto state_groups = state_at.view({static_cast<int64_t>(compressor_state->size(0)) / 128, 128, 2, head_dim});

    auto write_loc_at = infinicore::adaptor::to_aten_tensor(write_loc).reshape({tokens}).to(at::kLong);
    auto positions_at = infinicore::adaptor::to_aten_tensor(positions).reshape({tokens}).to(at::kLong);
    auto write_pos_at = positions_at.remainder(128);

    auto valid_write = write_loc_at.ge(0);
    auto valid_rows = at::nonzero(valid_write).reshape({-1});
    if (valid_rows.numel() > 0) {
        auto valid_groups = write_loc_at.index_select(0, valid_rows);
        auto valid_write_pos = write_pos_at.index_select(0, valid_rows);
        auto valid_values = kv_score_at.index_select(0, valid_rows).reshape({valid_rows.numel(), 2, head_dim});
        state_groups.index_put_({valid_groups, valid_write_pos}, valid_values);
    }

    auto boundary_mask = valid_write.logical_and((positions_at + 1).remainder(128).eq(0));
    auto boundary_rows = at::nonzero(boundary_mask).reshape({-1});
    if (boundary_rows.numel() > 0) {
        auto groups = write_loc_at.index_select(0, boundary_rows);
        auto state = state_groups.index_select(0, groups).to(at::kFloat);
        auto kv_window = state.select(2, 0);
        auto score_window = state.select(2, 1);
        auto ape_at = infinicore::adaptor::to_aten_tensor(ape).contiguous().to(at::kFloat);
        if (ape_at.size(0) < 128) {
            throw std::runtime_error("deepseek_v4_c128_compress_stateful_reference expects ape first dim >= 128.");
        }
        score_window = score_window + ape_at.slice(0, 0, 128).view({1, 128, head_dim});
        auto prob = at::softmax(score_window, 1);
        auto compressed = (kv_window * prob).sum(1);
        auto output_at = infinicore::adaptor::to_aten_tensor(output);
        output_at.index_copy_(0, boundary_rows, compressed.to(output_at.scalar_type()));
    }
    return output;
#else
    (void)kv_score_input;
    (void)ape;
    (void)compressor_state;
    (void)write_loc;
    (void)positions;
    throw std::runtime_error("deepseek_v4_c128_compress_stateful_reference requires an ATen-enabled HYGON build.");
#endif
}

Tensor deepseek_v4_c128_compress_stateful(const Tensor &kv_score_input,
                                          const Tensor &ape,
                                          Tensor compressor_state,
                                          const Tensor &write_loc,
                                          const Tensor &positions) {
    return deepseek_v4_c128_compress_stateful_reference(kv_score_input,
                                                        ape,
                                                        compressor_state,
                                                        write_loc,
                                                        positions);
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
        return;
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

    std::optional<at::Tensor> tile_scheduler_metadata = std::nullopt;
    std::optional<at::Tensor> num_splits = std::nullopt;
    static auto flash_mla_sparse_decode_fn = reinterpret_cast<FlashMlaSparseDecodeFn>(
        resolve_flashmla_sparse_decode("deepseek_v4_flashmla_sparse_attention_"));
    auto flash_out = flash_mla_sparse_decode_fn(q_flash,
                                                k_cache_fp8,
                                                indices_flash,
                                                topk_lengths_opt,
                                                attn_sink_opt,
                                                tile_scheduler_metadata,
                                                num_splits,
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
#else
    (void)q;
    (void)raw_cache;
    (void)indices;
    (void)topk_lengths;
    (void)attn_sink;
    (void)output;
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
} // namespace infinicore::op

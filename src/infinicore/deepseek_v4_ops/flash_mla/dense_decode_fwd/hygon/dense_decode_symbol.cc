#include "dense_decode_symbol.hpp"

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <elf.h>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::op::flash_mla::dense_decode_fwd_hygon {
namespace {

constexpr const char *kFlashMlaDenseDecodeInterfaceSymbol = "_ZL27dense_attn_decode_interfaceRN2at6TensorERKS0_iS3_S3_fbRSt8optionalIS0_ES6_";
constexpr const char *kDefaultFlashMlaSoPath = "/usr/local/lib/python3.10/dist-packages/flash_mla/cuda.cpython-310-x86_64-linux-gnu.so";
constexpr const char *kFlashMlaAnchorSymbol = "PyInit_cuda";

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

} // namespace

FlashMlaDenseDecodeFn flashmla_dense_decode_fn(const char *op_name) {
    static auto fn = reinterpret_cast<FlashMlaDenseDecodeFn>(
        resolve_flashmla_so_symbol(kFlashMlaDenseDecodeInterfaceSymbol, op_name));
    return fn;
}

} // namespace infinicore::op::flash_mla::dense_decode_fwd_hygon

#endif

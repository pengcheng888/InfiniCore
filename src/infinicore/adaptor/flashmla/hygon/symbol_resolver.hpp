#pragma once

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <elf.h>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

namespace infinicore::adaptor::flashmla::hygon::detail {

// Hygon ships several FlashMLA wheel variants. Prefer normal ELF symbols and
// use pybind11 registration records as the fallback for stripped wheels.

constexpr const char *kDefaultFlashMlaSoPath = "/usr/local/lib/python3.10/dist-packages/flash_mla/cuda.cpython-310-x86_64-linux-gnu.so";
constexpr const char *kFlashMlaAnchorSymbol = "PyInit_cuda";

// Returns an ELF symbol value (relative address), not a runtime address. The
// caller must add the shared object's load base.
inline uintptr_t find_elf_symbol_value(const std::string &path, const char *symbol) {
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
        if (symtab.sh_offset + symtab.sh_size > data.size()
            || strtab.sh_offset + strtab.sh_size > data.size()
            || symtab.sh_entsize != sizeof(Elf64_Sym)) {
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
    return 0;
}

// pybind11 ABI v1 stores a stateless function pointer at function_record::data[0].
// Both the legacy capsule and the current function_record Python object keep the
// function_record pointer immediately after PyObject_HEAD.
struct PyObjectPrefix {
    intptr_t ref_count;
    void *type;
    void *function_record;
};

struct FunctionRecordPrefix {
    const char *name;
    const char *doc;
    const char *signature;
    void *args[3];
    void *impl;
    void *data[3];
};

static_assert(offsetof(FunctionRecordPrefix, data) == 7 * sizeof(void *));

struct PythonApi {
    using GilEnsure = int (*)();
    using GilRelease = void (*)(int);
    using ImportModule = void *(*)(const char *);
    using GetAttrString = void *(*)(void *, const char *);
    using CFunctionGetSelf = void *(*)(void *);
    using DecRef = void (*)(void *);
    using ErrClear = void (*)();

    GilEnsure gil_ensure = load<GilEnsure>("PyGILState_Ensure");
    GilRelease gil_release = load<GilRelease>("PyGILState_Release");
    ImportModule import_module = load<ImportModule>("PyImport_ImportModule");
    GetAttrString get_attr_string = load<GetAttrString>("PyObject_GetAttrString");
    CFunctionGetSelf cfunction_get_self = load<CFunctionGetSelf>("PyCFunction_GetSelf");
    DecRef decref = load<DecRef>("_Py_DecRef");
    ErrClear err_clear = load<ErrClear>("PyErr_Clear");

private:
    template <typename Fn>
    static Fn load(const char *name) {
        auto fn = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, name));
        if (fn == nullptr) {
            throw std::runtime_error(std::string("missing CPython runtime symbol: ") + name);
        }
        return fn;
    }
};

// Uses the CPython C API through dlsym so this library does not need to link
// directly against libpython. The fallback requires the GIL; callers must not
// wait on another thread while holding it.
template <typename Fn>
void *resolve_pybind_function(void *so_base, const char *python_name, const char *op_name) {
    PythonApi py;
    const int gil_state = py.gil_ensure();
    void *module = nullptr;
    void *function = nullptr;

    try {
        module = py.import_module("flash_mla.cuda");
        if (module == nullptr) {
            py.err_clear();
            throw std::runtime_error(std::string(op_name) + " failed to import flash_mla.cuda");
        }
        function = py.get_attr_string(module, python_name);
        if (function == nullptr) {
            py.err_clear();
            throw std::runtime_error(std::string(op_name) + " missing flash_mla.cuda." + python_name);
        }

        void *self = py.cfunction_get_self(function);
        if (self == nullptr) {
            throw std::runtime_error(std::string(op_name) + " expected a pybind11 builtin for flash_mla.cuda." + python_name);
        }
        auto *record = static_cast<FunctionRecordPrefix *>(
            static_cast<PyObjectPrefix *>(self)->function_record);
        if (record == nullptr || record->name == nullptr || std::strcmp(record->name, python_name) != 0
            || record->data[0] == nullptr || record->data[1] == nullptr) {
            throw std::runtime_error(std::string(op_name) + " found an incompatible pybind11 function record for flash_mla.cuda." + python_name);
        }

        const auto *actual_type = static_cast<const std::type_info *>(record->data[1]);
        if (*actual_type != typeid(Fn)) {
            throw std::runtime_error(std::string(op_name) + " found an incompatible C++ signature for flash_mla.cuda." + python_name);
        }

        Dl_info function_info;
        if (dladdr(record->data[0], &function_info) == 0 || function_info.dli_fbase != so_base) {
            throw std::runtime_error(std::string(op_name) + " resolved flash_mla.cuda." + python_name + " from a different shared object");
        }

        void *result = record->data[0];
        py.decref(function);
        py.decref(module);
        py.gil_release(gil_state);
        return result;
    } catch (...) {
        if (function != nullptr) {
            py.decref(function);
        }
        if (module != nullptr) {
            py.decref(module);
        }
        py.gil_release(gil_state);
        throw;
    }
}

template <typename Fn>
Fn resolve_flashmla_decode_function(const char *symbol, const char *python_name, const char *op_name) {
    // Resolution order: global export, extension export, local ELF symbol,
    // then pybind11 function record for stripped extension builds.
    if (void *fn = dlsym(RTLD_DEFAULT, symbol)) {
        return reinterpret_cast<Fn>(fn);
    }

    const char *so_path = std::getenv("INFINICORE_DSV4_FLASHMLA_SO");
    if (so_path == nullptr || so_path[0] == '\0') {
        so_path = kDefaultFlashMlaSoPath;
    }

    void *handle = dlopen(so_path, RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        const char *err = dlerror();
        throw std::runtime_error(std::string(op_name) + " requires flash_mla.cuda.so; failed to dlopen "
                                 + so_path + (err == nullptr ? "" : std::string(": ") + err));
    }
    if (void *fn = dlsym(handle, symbol)) {
        return reinterpret_cast<Fn>(fn);
    }

    void *anchor = dlsym(handle, kFlashMlaAnchorSymbol);
    if (anchor == nullptr) {
        throw std::runtime_error(std::string(op_name) + " requires flash_mla.cuda.so anchor symbol: " + kFlashMlaAnchorSymbol);
    }

    Dl_info info;
    if (dladdr(anchor, &info) == 0 || info.dli_fbase == nullptr) {
        throw std::runtime_error(std::string(op_name) + " failed to resolve flash_mla SO load base");
    }
    const std::string loaded_path = info.dli_fname == nullptr ? std::string(so_path) : std::string(info.dli_fname);
    const uintptr_t symbol_value = find_elf_symbol_value(loaded_path, symbol);
    if (symbol_value != 0) {
        return reinterpret_cast<Fn>(reinterpret_cast<uintptr_t>(info.dli_fbase) + symbol_value);
    }

    return reinterpret_cast<Fn>(resolve_pybind_function<Fn>(info.dli_fbase, python_name, op_name));
}

} // namespace infinicore::adaptor::flashmla::hygon::detail

#endif

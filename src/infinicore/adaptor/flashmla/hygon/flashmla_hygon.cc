#include "flashmla_hygon.hpp"
#include "symbol_resolver.hpp"

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)

#include <mutex>
#include <stdexcept>
#include <string>

namespace infinicore::adaptor::flashmla::hygon {
namespace {

constexpr const char *kFlashMlaDenseDecodeInterfaceSymbol = "_ZL27dense_attn_decode_interfaceRN2at6TensorERKS0_iS3_S3_fbRSt8optionalIS0_ES6_";
constexpr const char *kFlashMlaDenseDecodePythonName = "dense_decode_fwd";

constexpr const char *kFlashMlaSparseDecodeInterfaceSymbol = "_ZL28sparse_attn_decode_interfaceRKN2at6TensorES2_S2_RKSt8optionalIS0_ES6_RS4_S7_S6_S6_S6_if";
constexpr const char *kFlashMlaSparseDecodePythonName = "sparse_decode_fwd";

struct Symbols {
    FlashMlaDenseDecodeFn dense_decode = nullptr;
    FlashMlaSparseDecodeFn sparse_decode = nullptr;
    std::string dense_decode_error;
    std::string sparse_decode_error;
};

Symbols &symbols() {
    static Symbols syms;
    static std::once_flag once;
    std::call_once(once, []() {
        try {
            syms.dense_decode = detail::resolve_flashmla_decode_function<FlashMlaDenseDecodeFn>(
                kFlashMlaDenseDecodeInterfaceSymbol,
                kFlashMlaDenseDecodePythonName,
                "flash_mla.cuda.dense_decode_fwd");
        } catch (const std::exception &e) {
            syms.dense_decode_error = e.what();
        }

        try {
            syms.sparse_decode = detail::resolve_flashmla_decode_function<FlashMlaSparseDecodeFn>(
                kFlashMlaSparseDecodeInterfaceSymbol,
                kFlashMlaSparseDecodePythonName,
                "flash_mla.cuda.sparse_decode_fwd");
        } catch (const std::exception &e) {
            syms.sparse_decode_error = e.what();
        }
    });
    return syms;
}

} // namespace

FlashMlaDenseDecodeFn flashmla_dense_decode_fn(const char *op_name) {
    auto &syms = symbols();
    if (syms.dense_decode == nullptr) {
        throw std::runtime_error(std::string(op_name) + " requires flash_mla.cuda."
                                 + kFlashMlaDenseDecodePythonName + ": " + syms.dense_decode_error);
    }
    return syms.dense_decode;
}

FlashMlaSparseDecodeFn flashmla_sparse_decode_fn(const char *op_name) {
    auto &syms = symbols();
    if (syms.sparse_decode == nullptr) {
        throw std::runtime_error(std::string(op_name) + " requires flash_mla.cuda."
                                 + kFlashMlaSparseDecodePythonName + ": " + syms.sparse_decode_error);
    }
    return syms.sparse_decode;
}

bool flashmla_dense_decode_available() {
    return symbols().dense_decode != nullptr;
}

bool flashmla_sparse_decode_available() {
    return symbols().sparse_decode != nullptr;
}

} // namespace infinicore::adaptor::flashmla::hygon

#endif

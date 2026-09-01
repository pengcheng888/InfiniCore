#include "dense_decode_symbol.hpp"

#if defined(ENABLE_ATEN) && defined(ENABLE_METAX_API)

#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>
#include <string>

namespace infinicore::op::flash_mla::dense_decode_fwd_metax {
namespace {

constexpr const char *kFlashMlaDenseDecodeSymbol = "_Z19mha_fwd_kvcache_mlaRN2at6TensorERKS0_RSt8optionalIS2_EiS3_S3_fbS3_S3_";
constexpr const char *kFlashMlaMetadataSymbol = "_Z16get_mla_metadataRN2at6TensorEii";
constexpr const char *kDefaultFlashMlaSoPath = "/opt/conda/lib/python3.10/site-packages/flash_mla_cuda.cpython-310-x86_64-linux-gnu.so";

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

} // namespace

FlashMlaDenseDecodeFn flashmla_dense_decode_fn(const char *op_name) {
    static auto fn = reinterpret_cast<FlashMlaDenseDecodeFn>(
        resolve_flashmla_so_symbol(kFlashMlaDenseDecodeSymbol, op_name));
    return fn;
}

FlashMlaMetadataFn flashmla_metadata_fn(const char *op_name) {
    static auto fn = reinterpret_cast<FlashMlaMetadataFn>(
        resolve_flashmla_so_symbol(kFlashMlaMetadataSymbol, op_name));
    return fn;
}

} // namespace infinicore::op::flash_mla::dense_decode_fwd_metax

#endif

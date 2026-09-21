#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
#include "infinicore/adaptor/hygon_vendor_adaptor.hpp"

#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/Exception.h>

#include <array>
#include <cstdlib>
#include <dlfcn.h>
#include <optional>
#include <stdexcept>
#include <string>

namespace infinicore::adaptor::hygon_vendor {
namespace {

using ConcatAndCacheMlaSignature =
    void(at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
         const std::string &, at::Tensor &);
using ConcatAndCacheMlaOperator =
    c10::TypedOperatorHandle<ConcatAndCacheMlaSignature>;

constexpr std::array<const char *, 2> kConcatAndCacheMlaOpNames = {
    "_C_cache_ops::concat_and_cache_mla",
    "hcu_ops::concat_and_cache_mla",
};
constexpr const char *kDefaultHcuOpsSoPath =
    "/usr/local/lib/python3.10/dist-packages/vllm_hcu/"
    "hcu_ops.cpython-310-x86_64-linux-gnu.so";

struct HcuOpsLoadResult {
    void *handle = nullptr;
    std::string error;
};

const HcuOpsLoadResult &load_hcu_ops() {
    static const HcuOpsLoadResult result = []() {
        HcuOpsLoadResult loaded;
        const char *path = std::getenv("INFINICORE_HCU_OPS_SO");
        if (path == nullptr || path[0] == '\0') {
            path = kDefaultHcuOpsSoPath;
        }

        loaded.handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
        if (loaded.handle == nullptr) {
            const char *error = dlerror();
            loaded.error = std::string("failed to load ") + path;
            if (error != nullptr) {
                loaded.error += ": ";
                loaded.error += error;
            }
        }
        return loaded;
    }();
    return result;
}

std::optional<ConcatAndCacheMlaOperator>
find_concat_and_cache_mla(std::string &signature_errors) {
    auto &dispatcher = c10::Dispatcher::singleton();
    for (const char *name : kConcatAndCacheMlaOpNames) {
        auto schema = dispatcher.findSchema({name, ""});
        if (!schema.has_value()) {
            continue;
        }

        try {
            return schema->typed<ConcatAndCacheMlaSignature>();
        } catch (const c10::Error &error) {
            if (!signature_errors.empty()) {
                signature_errors += "; ";
            }
            signature_errors += name;
            signature_errors += " has an incompatible schema: ";
            signature_errors += error.what_without_backtrace();
        }
    }
    return std::nullopt;
}

ConcatAndCacheMlaOperator resolve_concat_and_cache_mla() {
    std::string signature_errors;
    if (auto op = find_concat_and_cache_mla(signature_errors)) {
        return *op;
    }

    const auto &load_result = load_hcu_ops();
    if (auto op = find_concat_and_cache_mla(signature_errors)) {
        return *op;
    }

    std::string error =
        "concat_and_cache_mla could not find a compatible operator; tried "
        "_C_cache_ops::concat_and_cache_mla and "
        "hcu_ops::concat_and_cache_mla";
    if (!signature_errors.empty()) {
        error += "; ";
        error += signature_errors;
    }
    if (!load_result.error.empty()) {
        error += "; ";
        error += load_result.error;
    }
    throw std::runtime_error(error);
}

const ConcatAndCacheMlaOperator &concat_and_cache_mla_operator() {
    // A failed local-static initialization is retried after a vendor extension
    // is loaded later in process startup; a successful lookup is cached.
    static const auto op = resolve_concat_and_cache_mla();
    return op;
}

} // namespace

void concat_and_cache_mla(at::Tensor &kv_c,
                          at::Tensor &k_pe,
                          at::Tensor &kv_cache,
                          at::Tensor &slot_mapping,
                          const std::string &kv_cache_dtype,
                          at::Tensor &scale) {
    concat_and_cache_mla_operator().call(
        kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale);
}

} // namespace infinicore::adaptor::hygon_vendor
#endif // ENABLE_ATEN && ENABLE_HYGON_API

#pragma once

#include "infinicore/device.hpp"
#include "infinicore/graph/graph.hpp"
#include "infinicore/tensor.hpp"

#include <stdexcept>
#include <string>

#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_ILUVATAR_API)
#define INFINICORE_DSV4_ACCELERATOR_API 1
#endif

namespace infinicore::op::deepseek_v4::detail {

constexpr Device::Type build_device_type() noexcept {
#if defined(ENABLE_HYGON_API)
    return Device::Type::HYGON;
#elif defined(ENABLE_NVIDIA_API)
    return Device::Type::NVIDIA;
#elif defined(ENABLE_METAX_API)
    return Device::Type::METAX;
#elif defined(ENABLE_ILUVATAR_API)
    return Device::Type::ILUVATAR;
#else
    return Device::Type::CPU;
#endif
}

constexpr const char *build_device_name() noexcept {
#if defined(ENABLE_HYGON_API)
    return "HYGON";
#elif defined(ENABLE_NVIDIA_API)
    return "NVIDIA";
#elif defined(ENABLE_METAX_API)
    return "METAX";
#elif defined(ENABLE_ILUVATAR_API)
    return "ILUVATAR";
#else
    return "unsupported";
#endif
}

inline void check_build_device(const Tensor &tensor, const char *op_name) {
#if defined(INFINICORE_DSV4_ACCELERATOR_API)
    if (tensor->device().getType() != build_device_type()) {
        throw std::runtime_error(
            std::string(op_name) + " expects " + build_device_name()
            + " tensors in this build.");
    }
#else
    (void)tensor;
    throw std::runtime_error(
        std::string(op_name) + " requires a HYGON/NVIDIA/METAX/ILUVATAR build.");
#endif
}

} // namespace infinicore::op::deepseek_v4::detail

#if defined(INFINICORE_DSV4_ACCELERATOR_API)
#define INFINICORE_DSV4_GRAPH_OP_REGISTER_BUILD_DEVICE(                                 \
    __OP_NAME__, __PLAN_F__, __RUN_F__, __CLEANUP_F__)                                  \
    static bool registered = []() {                                                     \
        const auto device = ::infinicore::op::deepseek_v4::detail::build_device_type(); \
        __OP_NAME__::plan_dispatcher().registerDevice(device, __PLAN_F__, false);       \
        __OP_NAME__::run_dispatcher().registerDevice(device, __RUN_F__, false);         \
        __OP_NAME__::cleanup_dispatcher().registerDevice(                               \
            device, __CLEANUP_F__, false);                                              \
        return true;                                                                    \
    }();
#else
#define INFINICORE_DSV4_GRAPH_OP_REGISTER_BUILD_DEVICE( \
    __OP_NAME__, __PLAN_F__, __RUN_F__, __CLEANUP_F__)  \
    static bool registered = false;
#endif

#if defined(ENABLE_HYGON_API) || defined(ENABLE_NVIDIA_API)
#define INFINICORE_DSV4_NATIVE_GRAPH_OP_REGISTER_BUILD_DEVICE( \
    __OP_NAME__, __PLAN_F__, __RUN_F__, __CLEANUP_F__)         \
    INFINICORE_DSV4_GRAPH_OP_REGISTER_BUILD_DEVICE(            \
        __OP_NAME__, __PLAN_F__, __RUN_F__, __CLEANUP_F__)
#else
#define INFINICORE_DSV4_NATIVE_GRAPH_OP_REGISTER_BUILD_DEVICE( \
    __OP_NAME__, __PLAN_F__, __RUN_F__, __CLEANUP_F__)         \
    static bool registered = false;
#endif

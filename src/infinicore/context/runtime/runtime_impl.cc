#include "runtime_impl.hpp"

namespace infinicore {
namespace context {
void setDevice(Device device) {
}

Device getDevice() {
    return Device(Device::Type::cpu, 0);
}

void syncStream() {
}

void syncDevice() {
}

void memcpyH2D(void *dst, const void *src, size_t size) {
}

void memcpyD2H(void *dst, const void *src, size_t size) {
}

void memcpyD2D(void *dst, const void *src, size_t size) {
}

} // namespace context

} // namespace infinicore
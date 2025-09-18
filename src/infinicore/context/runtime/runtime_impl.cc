#include "runtime_impl.hpp"

namespace infinicore {
namespace context {
void setDevice(Device device) {
}

Device getDevice() {
    return Device(Device::Type::CPU, 0);
}

infinirtStream_t getStream() {
    return nullptr;
}

infiniopHandle_t getInfiniopHandle() {
    return nullptr;
}

void syncStream() {
}

void syncDevice() {
}

std::shared_ptr<Memory> allocateMemory(size_t size) {
    return nullptr;
}

void memcpyH2D(void *dst, const void *src, size_t size) {
}

void memcpyD2H(void *dst, const void *src, size_t size) {
}

void memcpyD2D(void *dst, const void *src, size_t size) {
}

} // namespace context

} // namespace infinicore
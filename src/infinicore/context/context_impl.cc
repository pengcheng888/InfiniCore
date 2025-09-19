#include "context_impl.hpp"

namespace infinicore {

Runtime *ContextImpl::getCurrentRuntime() {
    return current_runtime_;
}

ContextImpl &ContextImpl::singleton() {
    static ContextImpl instance;
    return instance;
}

ContextImpl::ContextImpl() {
    std::vector<int> device_counter(size_t(Device::Type::COUNT));
    infinirtGetAllDeviceCount(device_counter.data());

    // Reserve runtime slot for all devices.
    // Context will try to use the first non-cpu available device as the default runtime.
    for (int i = int(Device::Type::COUNT) - 1; i >= 0; i--) {
        if (device_counter[i] > 0) {
            runtime_table_[i].resize(device_counter[i]);
            if (current_runtime_ == nullptr) {
                runtime_table_[i][0] = std::make_unique<Runtime>(Device(Device::Type(i), 0));
                current_runtime_ = runtime_table_[i][0].get();
            } else if (i == 0) {
                runtime_table_[i][0] = std::make_unique<Runtime>(Device(Device::Type(i), 0));
            }
        }
    }
}

namespace context {
void setDevice(Device device) {
}

Device getDevice() {
    return Device(Device::Type::CPU, 0);
}

infinirtStream_t getStream() {
    // TODO: Implement this.
    return nullptr;
}

infiniopHandle_t getInfiniopHandle() {
    // TODO: Implement this.
    return nullptr;
}

void syncStream() {
}

void syncDevice() {
}

std::shared_ptr<Memory> allocateMemory(size_t size) {
    // TODO: Implement this.
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
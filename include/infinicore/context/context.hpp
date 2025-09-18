#pragma once

#include "../device.hpp"
#include "../memory.hpp"

#include <memory>

namespace infinicore {

namespace context {
void setDevice(Device device);
Device getDevice();

void syncStream();
void syncDevice();

std::shared_ptr<Memory> allocateMemory(size_t size);

void memcpyH2D(void *dst, const void *src, size_t size);
void memcpyD2H(void *dst, const void *src, size_t size);
void memcpyD2D(void *dst, const void *src, size_t size);

} // namespace context

} // namespace infinicore
#pragma once

#include "device.hpp"

#include <cstddef>

namespace infinicore {
class Memory {
private:
    std::byte *data_;
    size_t size_;
    Device device_;
    void (*_deleter)(void *);

public:
    Memory(std::byte *data, size_t size, Device device, void (*deleter)(void *));
    std::byte *data();
    Device device() const;
    size_t size() const;
};
} // namespace infinicore

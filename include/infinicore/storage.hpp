#pragma once

#include "device.hpp"

#include <cstddef>

namespace infinicore {
class Storage {
private:
    std::byte *_data;
    size_t _size;
    Device _device;

public:
    std::byte *data() { return _data; }
    const std::byte *data() const { return _data; }

    Device device() const { return _device; }
    size_t size() const { return _size; }
};
} // namespace infinicore

#pragma once

#include "../allocators/memory_allocator.hpp"
#include "infinicore/context/context.hpp"

#include <infinirt.h>

namespace infinicore {
class Runtime {
private:
    Device _device;
    infinirtStream_t _stream;
    std::unique_ptr<MemoryAllocator> _memory_allocator;

protected:
    Runtime(Device device);
    ~Runtime();
    friend class Context;

public:
    std::shared_ptr<Memory> allocateMemory(size_t size);
};
} // namespace infinicore

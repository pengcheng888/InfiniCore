#include "host_allocator.hpp"

namespace infinicore {
std::byte *HostAllocator::allocate(size_t size) {
    return nullptr;
}

void HostAllocator::deallocate(std::byte *ptr) {
}
} // namespace infinicore
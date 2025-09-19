#include "runtime.hpp"

namespace infinicore {

Runtime::Runtime(Device device) {}
Runtime::~Runtime() {}

infinirtStream_t Runtime::stream() const {}
std::shared_ptr<Memory> Runtime::allocateMemory(size_t size) {}

} // namespace infinicore

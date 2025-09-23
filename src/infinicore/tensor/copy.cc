#include "infinicore/context/context.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/tensor.hpp"

#include <spdlog/spdlog.h>

namespace infinicore {
Tensor TensorImpl::to(Device device) const {
    if (device == data_.memory->device()) {
        auto _t = std::make_shared<TensorImpl>(meta_.shape, meta_.strides, meta_.dtype);
        _t->data_ = data_;
        return Tensor(_t);
    } else {
        if (!is_contiguous()) {
            spdlog::error("Only contiguous tensors can be copied to another device.");
            std::abort();
        }
        std::shared_ptr<TensorImpl> _t = empty(meta_.shape, meta_.dtype, device, true);
        if (device == Device::Type::CPU) {
            context::memcpyD2H(_t->data(), data(), _t->data_.memory->size());
        } else if (this->device() == Device::Type::CPU) {
            context::memcpyH2D(_t->data(), data(), _t->data_.memory->size());
        } else {
            context::memcpyD2D(_t->data(), data(), _t->data_.memory->size());
        }
        return Tensor(_t);
    }
}
} // namespace infinicore

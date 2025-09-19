#include "infinicore/tensor.hpp"
#include "infinicore/dtype.hpp"

namespace infinicore {

TensorImpl *Tensor::operator->() { return impl_.get(); }

const TensorImpl *Tensor::operator->() const { return impl_.get(); }

Tensor Tensor::empty(const Shape &shape,
                     const DataType &dtype,
                     const Device &device) {
    return Tensor{TensorImpl::empty(shape, dtype, device)};
}

Tensor Tensor::zeros(const Shape &shape,
                     const DataType &dtype,
                     const Device &device) {
    return Tensor{TensorImpl::zeros(shape, dtype, device)};
}

Tensor Tensor::ones(const Shape &shape,
                    const DataType &dtype,
                    const Device &device) {
    return Tensor{TensorImpl::ones(shape, dtype, device)};
}

std::byte *TensorImpl::data() {
    return data_.memory->data();
}

const std::byte *TensorImpl::data() const {
    return data_.memory->data();
}

const Shape &TensorImpl::shape() const {
    return meta_.shape;
}

const Strides &TensorImpl::strides() const {
    return meta_.strides;
}

Size TensorImpl::size(size_t dim) const {
    return meta_.shape[dim];
}

Stride TensorImpl::stride(size_t dim) const {
    return meta_.strides[dim];
}

DataType TensorImpl::dtype() const {
    return meta_.dtype;
}

Device TensorImpl::device() const {
    return data_.memory->device();
}

infiniopTensorDescriptor_t TensorImpl::desc() const {
    // TODO: Implement this.
    return nullptr;
}

std::shared_ptr<TensorImpl> TensorImpl::empty(const Shape &shape,
                                              const DataType &dtype,
                                              const Device &device) {
    // TODO: Implement this.
    return nullptr;
}
std::shared_ptr<TensorImpl> TensorImpl::zeros(const Shape &shape,
                                              const DataType &dtype,
                                              const Device &device) {
    // TODO: Implement this.
    return nullptr;
}
std::shared_ptr<TensorImpl> TensorImpl::ones(const Shape &shape,
                                             const DataType &dtype,
                                             const Device &device) {
    // TODO: Implement this.
    return nullptr;
}

Tensor TensorImpl::narrow(const std::vector<SliceParams> &slices) const {
    // Create a new TensorImpl with narrowed view
    auto tensor_impl = std::make_shared<TensorImpl>();

    // Copy the metadata
    tensor_impl->meta_ = this->meta_;

    // Adjust shape and calculate offset
    size_t offset = data_.offset;

    for (const auto &slice : slices) {
        assert(slice.len > 0);
        assert(this->meta_.shape[slice.dim] >= slice.start + slice.len);
        tensor_impl->meta_.shape[slice.dim] = slice.len;
        offset += slice.start * this->meta_.strides[slice.dim] * dsize(this->meta_.dtype);
    }

    // Set the data with adjusted offset (same memory, different offset)
    tensor_impl->data_.offset = offset;
    tensor_impl->data_.memory = this->data_.memory;

    return Tensor(tensor_impl);
}

Tensor TensorImpl::permute(const std::vector<size_t> &order) const {
    // Validate input
    assert(meta_.shape.size() == order.size());

    // Check that order contains all indices from 0 to n-1 exactly once
    for (size_t i = 0; i < order.size(); i++) {
        assert(std::find(order.begin(), order.end(), i) != order.end());
    }

    auto tensor_impl = std::make_shared<TensorImpl>();

    // Copy the original metadata
    tensor_impl->meta_ = this->meta_;

    // Permute shape and strides
    Shape new_shape(order.size());
    Strides new_strides(order.size());

    for (size_t i = 0; i < order.size(); i++) {
        new_shape[i] = meta_.shape[order[i]];
        new_strides[i] = meta_.strides[order[i]];
    }

    tensor_impl->meta_.shape = new_shape;
    tensor_impl->meta_.strides = new_strides;

    tensor_impl->data_ = this->data_;

    return Tensor(tensor_impl);
}

Tensor TensorImpl::view(const std::vector<size_t> &new_shape) const {
    // Step 1: Validate total size
    size_t numel = 1;
    for (size_t dim : meta_.shape) {
        numel *= dim;
    }

    size_t new_numel = 1;
    for (size_t dim : new_shape) {
        new_numel *= dim;
    }

    assert(numel == new_numel);

    // Step 2: Get current shape and strides
    const Shape &old_shape = meta_.shape;
    const Strides &old_strides = meta_.strides;

    // Step 3: Create merged shape and strides
    std::vector<size_t> merged_shape;
    std::vector<ptrdiff_t> merged_strides;

    if (!old_shape.empty()) {
        merged_shape.push_back(old_shape[0]);
        merged_strides.push_back(old_strides[0]);

        for (size_t i = 1; i < old_shape.size(); ++i) {
            if (old_strides[i] * static_cast<ptrdiff_t>(old_shape[i]) == merged_strides.back()) {
                merged_shape.back() *= old_shape[i];
                merged_strides.back() = old_strides[i];
            } else {
                merged_shape.push_back(old_shape[i]);
                merged_strides.push_back(old_strides[i]);
            }
        }
    }

    // Step 4: Compute new strides by splitting merged dimensions
    std::vector<ptrdiff_t> new_strides(new_shape.size());
    size_t merged_idx = 0;
    ptrdiff_t current_stride = merged_strides[0];
    size_t remaining_size = merged_shape[0];

    for (size_t i = 0; i < new_shape.size(); ++i) {
        // Find which merged dimension contains this new dimension
        while (new_shape[i] > remaining_size) {
            assert(++merged_idx < merged_shape.size());
            current_stride = merged_strides[merged_idx];
            remaining_size = merged_shape[merged_idx];
        }

        assert(remaining_size % new_shape[i] == 0);

        new_strides[i] = current_stride * (remaining_size / new_shape[i]);
        remaining_size /= new_shape[i];
    }

    return this->view_as(new_shape, new_strides);
}

Tensor TensorImpl::view_as(const std::vector<size_t> &new_shape) const {
    auto tensor_impl = std::make_shared<TensorImpl>();
    tensor_impl->meta_.dtype = meta_.dtype;
    tensor_impl->meta_.shape = new_shape;

    // Compute contiguous strides for the new shape
    tensor_impl->meta_.strides.resize(new_shape.size());
    ptrdiff_t stride = 1;
    for (int i = new_shape.size() - 1; i >= 0; --i) {
        tensor_impl->meta_.strides[i] = stride;
        stride *= new_shape[i];
    }

    tensor_impl->data_ = data_;

    return Tensor(tensor_impl);
}

Tensor TensorImpl::view_as(const std::vector<size_t> &new_shape, const std::vector<Stride> &new_strides) const {
    auto tensor_impl = std::make_shared<TensorImpl>();
    tensor_impl->meta_.dtype = meta_.dtype;
    tensor_impl->meta_.shape = new_shape;
    tensor_impl->meta_.strides = new_strides;

    tensor_impl->data_ = data_;

    return Tensor(tensor_impl);
}

} // namespace infinicore

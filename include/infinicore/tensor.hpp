#pragma once

#include "device.hpp"
#include "dtype.hpp"
#include "storage.hpp"

#include <memory>
#include <vector>
namespace infinicore {

using Size = std::size_t;
using Stride = std::ptrdiff_t;
using Shape = std::vector<Size>;
using Strides = std::vector<Stride>;

class TensorImpl;

struct TensorMetaData {
    Shape shape;
    Strides strides;
    DataType dtype;
};

struct TensorData {
    size_t offset;
    std::shared_ptr<Storage> storage;
};

class Tensor {
public:
    static Tensor empty(const Shape &shape,
                        const DataType &dtype,
                        const Device &device);

    static Tensor zeros(const Shape &shape,
                        const DataType &dtype,
                        const Device &device);

    static Tensor ones(const Shape &shape,
                       const DataType &dtype,
                       const Device &device);
    Tensor(const Tensor &) = default;
    Tensor(Tensor &&) = default;
    Tensor &operator=(const Tensor &) = default;
    Tensor &operator=(Tensor &&) = default;

    TensorImpl *operator->() { return _impl.get(); }
    const TensorImpl *operator->() const { return _impl.get(); }

protected:
    explicit Tensor(std::shared_ptr<TensorImpl> impl) : _impl(std::move(impl)) {}
    std::shared_ptr<TensorImpl> _impl;
    friend class TensorImpl;
};

class TensorImpl {

public:
    const Shape &shape() const;

    DataType dtype() const;

    Device device() const;

protected:
    static Tensor empty(const Shape &shape, const DataType &dtype, const Device &device);
    static Tensor zeros(const Shape &shape, const DataType &dtype, const Device &device);
    static Tensor ones(const Shape &shape, const DataType &dtype, const Device &device);

    TensorImpl();
    friend class Tensor;

private:
    TensorMetaData _meta;
    TensorData _data;
};

} // namespace infinicore

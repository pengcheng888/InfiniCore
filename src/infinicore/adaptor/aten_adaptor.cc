#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"

#include <memory>
#include <stdexcept>
#include <string>


namespace {

infinicore::DataType from_at_dtype(at::ScalarType dtype) {
    switch (dtype) {
    case at::kFloat:
        return infinicore::DataType::F32;
    case at::kHalf:
        return infinicore::DataType::F16;
    case at::kBFloat16:
        return infinicore::DataType::BF16;
    case at::kChar:
        return infinicore::DataType::I8;
    case at::kInt:
        return infinicore::DataType::I32;
    case at::kLong:
        return infinicore::DataType::I64;
    case at::kByte:
        return infinicore::DataType::U8;
#if defined(ENABLE_HYGON_API)
    case at::kFloat8_e4m3fnuz:
        return infinicore::DataType::F8;
#else
    case at::kFloat8_e4m3fn:
        return infinicore::DataType::F8;
#endif
    default:
        throw std::runtime_error("Unsupported ATen dtype for InfiniCore tensor wrapping: " + std::to_string(static_cast<int>(dtype)));
    }
}

infinicore::Device from_at_device(const at::Device &device) {
    if (device.is_cpu()) {
        return infinicore::Device(infinicore::Device::Type::CPU, 0);
    }
    if (device.is_cuda()) {
#if defined(ENABLE_HYGON_API)
        return infinicore::Device(infinicore::Device::Type::HYGON, static_cast<infinicore::Device::Index>(device.index()));
#elif defined(ENABLE_NVIDIA_API)
        return infinicore::Device(infinicore::Device::Type::NVIDIA, static_cast<infinicore::Device::Index>(device.index()));
#elif defined(ENABLE_METAX_API)
        return infinicore::Device(infinicore::Device::Type::METAX, static_cast<infinicore::Device::Index>(device.index()));
#elif defined(ENABLE_QY_API)
        return infinicore::Device(infinicore::Device::Type::QY, static_cast<infinicore::Device::Index>(device.index()));
#else
        return infinicore::Device(infinicore::Device::Type::NVIDIA, static_cast<infinicore::Device::Index>(device.index()));
#endif
    }
    throw std::runtime_error("Unsupported ATen device for InfiniCore tensor wrapping: " + device.str());
}

} // namespace

namespace infinicore::adaptor {

at::Tensor to_aten_tensor(const infinicore::Tensor &t) {
    void *data_ptr = (void *)(t->data());

    auto sizes = std::vector<int64_t>(
        t->shape().begin(),
        t->shape().end());

    auto strides = t->strides();

    auto dtype = to_at_dtype(t->dtype());
    auto device = to_at_device(t->device());

    auto deleter_ = [](void * /*unused*/) mutable {

    };

    at::TensorOptions options = at::TensorOptions()
                                    .dtype(dtype)
                                    .device(device)
                                    .requires_grad(false);

    return at::from_blob(
        data_ptr,
        sizes,
        strides,
        deleter_,
        options);
}


Tensor from_aten_tensor(const at::Tensor &t) {
    if (!t.defined()) {
        throw std::runtime_error("from_aten_tensor expects a defined ATen tensor.");
    }
    auto contiguous = t.contiguous();
    auto holder = std::make_shared<at::Tensor>(contiguous);
    Shape shape;
    shape.reserve(static_cast<size_t>(contiguous.dim()));
    for (const auto dim : contiguous.sizes()) {
        shape.push_back(static_cast<size_t>(dim));
    }
    auto deleter = [holder](std::byte *) mutable {
        holder.reset();
    };
    return Tensor::from_blob(holder->data_ptr(), shape, from_at_dtype(holder->scalar_type()), from_at_device(holder->device()), std::move(deleter));
}

#if defined(ENABLE_HYGON_API)
c10::hip::HIPStream get_hip_stream() {
    return c10::hip::getStreamFromExternal(
        hipStream_t(infinicore::context::getStream()), infinicore::context::getDevice().getIndex());
}
#elif defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
c10::cuda::CUDAStream get_cuda_stream() {
    return c10::cuda::getStreamFromExternal(
        cudaStream_t(infinicore::context::getStream()), infinicore::context::getDevice().getIndex());
}
#endif

#if defined(ENABLE_MOORE_API)
c10::musa::MUSAStream get_musa_stream() {
    return c10::musa::getStreamFromExternal(
        musaStream_t(infinicore::context::getStream()), infinicore::context::getDevice().getIndex());
}
#endif

} // namespace infinicore::adaptor

#endif // ENABLE_ATEN

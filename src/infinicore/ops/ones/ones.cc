#include "infinicore/ops/ones.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Ones::schema> &Ones::dispatcher() {
    static common::OpDispatcher<Ones::schema> dispatcher_;
    return dispatcher_;
};

void Ones::execute(Tensor output) {
    context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Ones implementation found for device type: "
                                 + std::to_string(static_cast<int>(device_type)));
    }

    func(output);
}

void ones_(Tensor output) {
    Ones::execute(output);
}

} // namespace infinicore::op

#pragma once

#include "../../device.hpp"

namespace infinicore::op::common {
template <typename Fn>
class OpDispatcher {
public:
    void registerOp(Device::Type device_type, Fn fn) {
        table_[device_type] = fn;
    }

    Fn lookup(Device::Type device_type) const {
        return table_.at({device_type});
    }

private:
    struct Key {
        Device::Type device_type;
        bool operator<(const Key &o) const {
            return std::tie(device_type) < std::tie(o.device_type);
        }
    };
    std::map<Key, Fn> table_;
};
} // namespace infinicore::op::common

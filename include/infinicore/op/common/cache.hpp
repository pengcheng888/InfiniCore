#pragma once

#include "../../common/LRUCache.hpp"
#include "../../device.hpp"
#include <array>
#include <functional>
#include <memory>
#include <vector>

namespace infinicore::op::common {

template <typename Key, typename Value>
class OpCache {
private:
    using BaseCache = infinicore::common::LRUCache<Key, Value>;
    using CacheVector = std::vector<BaseCache>;
    using CacheArray = std::array<CacheVector, static_cast<size_t>(Device::Type::COUNT)>;

public:
    using Destructor = typename BaseCache::Destructor;

    explicit OpCache(size_t capacity = 100, Destructor destructor = nullptr)
        : capacity_(capacity), destructor_(destructor) {}

    ~OpCache() {
        clear();
    }

    BaseCache &getCache(Device::Type device_type, size_t device_index) {
        auto &cache_vector = caches_[static_cast<size_t>(device_type)];

        if (cache_vector.size() <= device_index) {
            // Resize and initialize with BaseCache objects with destructor
            cache_vector.resize(device_index + 1, BaseCache(capacity_, destructor_));
        } else {
            // Ensure the cache has the current destructor
            cache_vector[device_index].setDestructor(destructor_);
        }

        return cache_vector[device_index];
    }

    void setCapacity(size_t capacity) {
        capacity_ = capacity;
        for (auto &vec : caches_) {
            for (auto &cache : vec) {
                cache.setCapacity(capacity);
            }
        }
    }

    void clear() {
        for (auto &vec : caches_) {
            for (auto &cache : vec) {
                cache.clear();
            }
            vec.clear();
        }
    }

private:
    size_t capacity_;
    Destructor destructor_;

    thread_local static CacheArray caches_;
};

// Static member definition for thread-local caches
template <typename Key, typename Value>
thread_local typename OpCache<Key, Value>::CacheArray
    OpCache<Key, Value>::caches_;

} // namespace infinicore::op::common

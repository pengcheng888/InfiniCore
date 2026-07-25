#include "infinicore/ops/deepseek_v4_dcu_custom_allreduce.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/device.hpp"

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#endif

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <vector>
#endif

namespace infinicore::op {

namespace {

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
using AllocateSharedBufferAndHandleSchema = std::tuple<int64_t, at::Tensor>(int64_t);
using OpenMemHandleSchema = int64_t(at::Tensor &);
using InitCustomArSchema = int64_t(c10::List<int64_t>, const at::Tensor &, int64_t, bool);
using RegisterBufferSchema = void(int64_t, c10::List<int64_t>);
using AllReduceSchema = void(int64_t, const at::Tensor &, at::Tensor &, int64_t, int64_t);

const c10::TypedOperatorHandle<AllocateSharedBufferAndHandleSchema> &allocate_shared_buffer_and_handle_op() {
    static const auto op = c10::Dispatcher::singleton()
                               .findSchemaOrThrow("_C_custom_ar::allocate_shared_buffer_and_handle", "")
                               .typed<AllocateSharedBufferAndHandleSchema>();
    return op;
}

const c10::TypedOperatorHandle<OpenMemHandleSchema> &open_mem_handle_op() {
    static const auto op = c10::Dispatcher::singleton()
                               .findSchemaOrThrow("_C_custom_ar::open_mem_handle", "")
                               .typed<OpenMemHandleSchema>();
    return op;
}

const c10::TypedOperatorHandle<InitCustomArSchema> &init_custom_ar_op() {
    static const auto op = c10::Dispatcher::singleton()
                               .findSchemaOrThrow("_C_custom_ar::init_custom_ar", "")
                               .typed<InitCustomArSchema>();
    return op;
}

const c10::TypedOperatorHandle<RegisterBufferSchema> &register_buffer_op() {
    static const auto op = c10::Dispatcher::singleton()
                               .findSchemaOrThrow("_C_custom_ar::register_buffer", "")
                               .typed<RegisterBufferSchema>();
    return op;
}

const c10::TypedOperatorHandle<AllReduceSchema> &all_reduce_op() {
    static const auto op = c10::Dispatcher::singleton()
                               .findSchemaOrThrow("_C_custom_ar::all_reduce", "")
                               .typed<AllReduceSchema>();
    return op;
}

struct RankState {
    bool buffers_ready{false};
    bool initialized{false};
    int64_t meta_ptr{0};
    int64_t buffer_ptr{0};
    at::Tensor meta_handle;
    at::Tensor buffer_handle;
    at::Tensor rank_data;
    int64_t fa{0};
    std::vector<int64_t> meta_ptrs;
    std::vector<int64_t> buffer_ptrs;
};

struct CustomArGroup {
    std::mutex mutex;
    std::condition_variable cv;
    int world_size{0};
    int max_size_bytes{0};
    std::vector<RankState> ranks;
};

CustomArGroup &custom_ar_group() {
    static CustomArGroup group;
    return group;
}

std::mutex &ipc_open_mutex() {
    static std::mutex mutex;
    return mutex;
}

bool supported_world_size(int world_size) {
    return world_size == 2 || world_size == 4 || world_size == 6 || world_size == 8 || world_size == 16;
}

bool should_custom_ar(const Tensor &input, int tp_rank, int tp_size, int max_size_bytes) {
    if (tp_rank < 0 || tp_rank >= tp_size || !supported_world_size(tp_size)) {
        return false;
    }
    if (!input || !input->is_contiguous()) {
        return false;
    }
    const size_t bytes = input->nbytes();
    return bytes > 0 && bytes <= static_cast<size_t>(max_size_bytes) && bytes % 16 == 0;
}

void ensure_rank_initialized(const Tensor &input, int tp_rank, int tp_size, int max_size_bytes) {
    auto &group = custom_ar_group();
    bool need_allocate = false;
    {
        std::unique_lock<std::mutex> lock(group.mutex);
        if (group.world_size == 0) {
            group.world_size = tp_size;
            group.max_size_bytes = max_size_bytes;
            group.ranks.resize(static_cast<size_t>(tp_size));
        } else if (group.world_size != tp_size || group.max_size_bytes != max_size_bytes) {
            throw std::runtime_error("deepseek_v4_dcu_custom_allreduce_: custom AR group shape changed after initialization.");
        }
        need_allocate = !group.ranks[static_cast<size_t>(tp_rank)].buffers_ready;
    }

    if (need_allocate) {
        const int64_t meta_size_bytes = static_cast<int64_t>(max_size_bytes) + 8192 * 1024;
        auto [meta_ptr, meta_handle] = allocate_shared_buffer_and_handle_op().call(meta_size_bytes);
        auto [buffer_ptr, buffer_handle] = allocate_shared_buffer_and_handle_op().call(max_size_bytes);
        auto options = at::TensorOptions()
                           .dtype(at::kByte)
                           .device(infinicore::adaptor::to_at_device(input->device()))
                           .requires_grad(false);
        auto rank_data = at::empty({8 * 1024 * 1024}, options);

        std::unique_lock<std::mutex> lock(group.mutex);
        auto &state = group.ranks[static_cast<size_t>(tp_rank)];
        if (!state.buffers_ready) {
            state.meta_ptr = meta_ptr;
            state.buffer_ptr = buffer_ptr;
            state.meta_handle = meta_handle;
            state.buffer_handle = buffer_handle;
            state.rank_data = rank_data;
            state.buffers_ready = true;
        }
        group.cv.notify_all();
    }

    std::unique_lock<std::mutex> lock(group.mutex);
    group.cv.wait(lock, [&group] {
        if (group.world_size <= 0 || group.ranks.size() != static_cast<size_t>(group.world_size)) {
            return false;
        }
        for (const auto &rank : group.ranks) {
            if (!rank.buffers_ready) {
                return false;
            }
        }
        return true;
    });

    auto &state = group.ranks[static_cast<size_t>(tp_rank)];
    if (state.initialized) {
        return;
    }

    std::vector<int64_t> local_meta_ptrs;
    std::vector<int64_t> local_buffer_ptrs;
    std::vector<at::Tensor> meta_handles;
    std::vector<at::Tensor> buffer_handles;
    local_meta_ptrs.reserve(static_cast<size_t>(tp_size));
    local_buffer_ptrs.reserve(static_cast<size_t>(tp_size));
    meta_handles.reserve(static_cast<size_t>(tp_size));
    buffer_handles.reserve(static_cast<size_t>(tp_size));
    for (const auto &rank : group.ranks) {
        local_meta_ptrs.push_back(rank.meta_ptr);
        local_buffer_ptrs.push_back(rank.buffer_ptr);
        meta_handles.push_back(rank.meta_handle);
        buffer_handles.push_back(rank.buffer_handle);
    }
    auto rank_data = state.rank_data;
    lock.unlock();

    {
        std::lock_guard<std::mutex> open_lock(ipc_open_mutex());
        for (int i = 0; i < tp_size; ++i) {
            if (i == tp_rank) {
                continue;
            }
            auto meta_handle = meta_handles[static_cast<size_t>(i)];
            auto buffer_handle = buffer_handles[static_cast<size_t>(i)];
            local_meta_ptrs[static_cast<size_t>(i)] = open_mem_handle_op().call(meta_handle);
            local_buffer_ptrs[static_cast<size_t>(i)] = open_mem_handle_op().call(buffer_handle);
        }
    }
    auto fa = init_custom_ar_op().call(c10::List<int64_t>(local_meta_ptrs), rank_data, tp_rank, true);
    register_buffer_op().call(fa, c10::List<int64_t>(local_buffer_ptrs));

    lock.lock();
    state.fa = fa;
    state.meta_ptrs = std::move(local_meta_ptrs);
    state.buffer_ptrs = std::move(local_buffer_ptrs);
    state.initialized = true;
}
#endif

} // namespace

bool deepseek_v4_dcu_custom_allreduce_(Tensor output,
                                       const Tensor &input,
                                       int tp_rank,
                                       int tp_size,
                                       int max_size_bytes) {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    if (!should_custom_ar(input, tp_rank, tp_size, max_size_bytes)) {
        return false;
    }
    if (!output || output->shape() != input->shape() || output->dtype() != input->dtype() || output->device() != input->device() || !output->is_contiguous()) {
        return false;
    }
    try {
        c10::hip::HIPGuard device_guard(static_cast<c10::DeviceIndex>(input->device().getIndex()));
        c10::hip::HIPStreamGuard stream_guard(infinicore::adaptor::get_hip_stream());
        ensure_rank_initialized(input, tp_rank, tp_size, max_size_bytes);

        auto input_at = infinicore::adaptor::to_aten_tensor(input);
        auto output_at = infinicore::adaptor::to_aten_tensor(output);
        auto &state = custom_ar_group().ranks[static_cast<size_t>(tp_rank)];
        all_reduce_op().call(state.fa, input_at, output_at, state.buffer_ptrs[static_cast<size_t>(tp_rank)], max_size_bytes);
        return true;
    } catch (...) {
        return false;
    }
#else
    (void)output;
    (void)input;
    (void)tp_rank;
    (void)tp_size;
    (void)max_size_bytes;
    return false;
#endif
}

} // namespace infinicore::op

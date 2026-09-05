#pragma once

#include "infinicore/tensor.hpp"

#include <cstddef>
#include <optional>

namespace infinicore::op::flash_mla {

class FlashMLASchedMeta {
public:
    struct Config {
        size_t b;
        size_t s_q;
        size_t h_q;
        size_t page_block_size;
        size_t h_k;

        bool causal;
        bool is_fp8_kvcache;
        std::optional<size_t> topk;

        std::optional<size_t> extra_page_block_size;
        std::optional<size_t> extra_topk;
    };

    bool have_initialized{false};
    bool have_refreshed{false};
    std::optional<Config> config;
    Tensor tile_scheduler_metadata;
    Tensor num_splits;

    FlashMLASchedMeta() = default;

    bool has_sched_buffer() const {
        return tile_scheduler_metadata && num_splits;
    }

    bool has_valid_sched_meta() const {
        return has_sched_buffer() && have_refreshed;
    }

    bool has_sched_meta() const {
        return has_sched_buffer();
    }
};

} // namespace infinicore::op::flash_mla

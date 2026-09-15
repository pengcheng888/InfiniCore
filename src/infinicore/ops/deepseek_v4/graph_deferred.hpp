#pragma once

#include "infinicore/context/context.hpp"
#include "infinicore/graph/graph.hpp"

#include <functional>
#include <memory>
#include <utility>

namespace infinicore::op::deepseek_v4::detail {

class DeferredHostGraphOperator final : public graph::GraphOperator {
public:
    explicit DeferredHostGraphOperator(std::function<void()> runner)
        : runner_(std::move(runner)) {}

    void run() const override {
        runner_();
    }

    bool is_device_graph_capture_safe() const override {
        return false;
    }

private:
    std::function<void()> runner_;
};

template <typename Fn>
void record_or_run_host_graph_op(Fn &&fn) {
    auto op = std::make_shared<DeferredHostGraphOperator>(
        std::function<void()>(std::forward<Fn>(fn)));
    if (context::isGraphRecording()) {
        context::addGraphOperator(op);
    } else {
        op->run();
    }
}

} // namespace infinicore::op::deepseek_v4::detail

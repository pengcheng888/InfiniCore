#include "graph_manager.hpp"

#include "../utils.hpp"
#include "infinicore/context/context.hpp"
#include <infinirt.h>
#include <cstdlib>
#include <cstdio>

namespace infinicore::graph {

namespace {
bool graph_debug_enabled() {
    static const bool enabled = []() {
        const char *value = std::getenv("INFINICORE_GRAPH_DEBUG");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}
} // namespace

/* =========================
 * GraphTensor
 * ========================= */

GraphTensor::GraphTensor(const Tensor &tensor) : Tensor(tensor->to_blob_()) {
}

/* =========================
 * GraphOperator
 * ========================= */

void DispatchableGraphOperator::run() const {
    runner_(planned_meta_);
}

DispatchableGraphOperator::~DispatchableGraphOperator() {
    if (deleter_) {
        deleter_(&planned_meta_);
    }
}

/* =========================
 * Graph
 * ========================= */

struct Graph::DeviceGraph {
    infinirtGraph_t graph = nullptr;
    infinirtGraphExec_t exec = nullptr;
    infinirtGraphNode_t node = nullptr;
    std::vector<char> log_buffer;

    DeviceGraph() {
        log_buffer.resize(4 * 1024);
    }

    ~DeviceGraph() {
        if (exec) {
            infinirtGraphExecDestroy(exec);
        }
        if (graph) {
            infinirtGraphDestroy(graph);
        }
    }

    void launch() {
        INFINICORE_CHECK_ERROR(infinirtGraphLuanch(exec, context::getStream()));
    }
};

Graph::Graph() {
}

void Graph::run() const {
    if (device_graph_ != nullptr && device_graph_.get()->exec != nullptr) {
        if (graph_debug_enabled()) {
            std::fprintf(stderr, "[infinicore graph] run device_graph ops=%zu\n", op_list_.size());
        }
        device_graph_.get()->launch();
    } else {
        if (graph_debug_enabled()) {
            std::fprintf(stderr, "[infinicore graph] run op_list ops=%zu\n", op_list_.size());
        }
        for (auto &op : op_list_) {
            op->run();
        }
    }
}

void Graph::add_operator(std::shared_ptr<GraphOperator> op) {
    if (graph_debug_enabled()) {
        std::fprintf(stderr,
                     "[infinicore graph] add_operator supports_device_graph_capture=%d\n",
                     op->supports_device_graph_capture() ? 1 : 0);
    }
    op_list_.push_back(op);
}

void Graph::instantiate() {
    if (graph_debug_enabled()) {
        std::fprintf(stderr, "[infinicore graph] instantiate ops=%zu\n", op_list_.size());
    }
    if (op_list_.empty()) {
        device_graph_.reset();
        return;
    }

    for (auto &op : op_list_) {
        if (!op->supports_device_graph_capture()) {
            if (graph_debug_enabled()) {
                std::fprintf(stderr, "[infinicore graph] skip device graph capture\n");
            }
            device_graph_.reset();
            return;
        }
    }

    // Reset device graph
    device_graph_ = std::make_unique<DeviceGraph>();

    // warmup
    for (size_t iter = 0; iter < 5; ++iter) {
        this->run();
    }
    infinicore::context::syncStream();

    if (infinirtStreamBeginCapture(
            context::getStream(),
            INFINIRT_STREAM_CAPTURE_MODE_RELAXED)
        != INFINI_STATUS_SUCCESS) {
        device_graph_.reset();
        return;
    }

    // Run and record
    this->run();

    if (infinirtStreamEndCapture(
            context::getStream(),
            &device_graph_.get()->graph)
        != INFINI_STATUS_SUCCESS) {
        device_graph_.reset();
        return;
    }

    if (infinirtGraphInstantiate(
            &device_graph_.get()->exec,
            device_graph_.get()->graph,
            &device_graph_.get()->node,
            device_graph_.get()->log_buffer.data(),
            device_graph_.get()->log_buffer.size())
        != INFINI_STATUS_SUCCESS) {
        static bool warned_once = false;
        if (!warned_once) {
            warned_once = true;
            spdlog::warn("Fail to instantiate device graph: {}", std::string(device_graph_.get()->log_buffer.data()));
        }
        device_graph_.reset();
    }
}

Graph::~Graph() = default;

/* =========================
 * GraphManager
 * ========================= */

bool GraphManager::is_recording() const {
    return recording_;
}

void GraphManager::start_recording() {
    if (is_recording()) {
        spdlog::warn("Graph is already recording. Previous recording will be dropped.");
    }
    recording_ = true;
    graph_ = std::make_shared<Graph>();
}

void GraphManager::add_operator(std::shared_ptr<GraphOperator> op) {
    INFINICORE_ASSERT(is_recording());

    graph_->add_operator(op);
}

std::shared_ptr<Graph> GraphManager::stop_recording() {
    if (!is_recording()) {
        spdlog::warn("Graph is not recording. Please start recording first.");
        return nullptr;
    }
    recording_ = false;
#ifdef USE_INFINIRT_GRAPH
    graph_->instantiate();
#endif
    return std::exchange(graph_, nullptr);
}

} // namespace infinicore::graph

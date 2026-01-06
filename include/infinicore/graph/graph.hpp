#pragma once

#include <memory>
#include <vector>

#include "../tensor.hpp"

namespace infinicore::graph {
// Forward declarations
class GraphManager;

class GraphTensor : public Tensor {
public:
    GraphTensor(const Tensor &);
};

class GraphOperator {

public:
    void run() const;
    ~GraphOperator();

protected:
    using run_schema = void (*)(void *);
    using cleanup_schema = void (*)(void **);
    void *planned_meta_;
    run_schema runner_;
    cleanup_schema deleter_;
};

class Graph {
public:
    Graph() = default;
    ~Graph() = default;

    void run() const;

protected:
    void add_operator(std::shared_ptr<GraphOperator> op);

    std::vector<std::shared_ptr<GraphOperator>> op_list_;

    friend class GraphManager;
};
} // namespace infinicore::graph

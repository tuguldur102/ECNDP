#ifndef IMPROVEMENT_CONDITION_HPP
#define IMPROVEMENT_CONDITION_HPP

#include <unordered_set>
#include <vector>

#include "exact.hpp"

bool improvement_condition(
    const GraphData &graph_data,
    const std::unordered_set<int> &K,
    int best_pc,
    const std::vector<int> &terminals);

#endif
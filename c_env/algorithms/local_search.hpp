#ifndef LOCAL_SEARCH_HPP
#define LOCAL_SEARCH_HPP

#include <functional>
#include <unordered_set>
#include <vector>

#include "exact.hpp"

std::unordered_set<int> local_search_procedure(
    const GraphData &graph_data,
    const std::unordered_set<int> &S,
    const std::vector<int> &terminals,
    int case_value,
    const std::function<bool(
        const GraphData &,
        const std::unordered_set<int> &,
        int,
        const std::vector<int> &)> &improvement_condition);

#endif
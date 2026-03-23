#ifndef GREEDY_EMPTY_SET_HPP
#define GREEDY_EMPTY_SET_HPP

#include <unordered_set>
#include <utility>
#include <vector>

#include "exact.hpp"

std::unordered_set<int> greedy_empty_set(
    const GraphData &graph_data,
    const std::vector<int> &terminals,
    int k,
    int case_value);

std::pair<std::unordered_set<int>, int> extended_critical_node_empty_set(
    const GraphData &graph_data,
    const std::vector<int> &terminals,
    int k,
    int case_value,
    int max_iter,
    bool use_local_search);

#endif
#ifndef UTILS_HPP
#define UTILS_HPP

#include <unordered_set>
#include <utility>
#include <vector>

#include "exact.hpp"

std::vector<std::vector<int>> build_adjacency_list(
    const GraphData &graph_data);

std::vector<std::vector<int>> connected_components(
    const GraphData &graph_data);

std::vector<std::vector<int>> connected_components_after_deletion(
    const GraphData &graph_data,
    const std::unordered_set<int> &deleted_nodes);

int comb_of_two(
    const std::vector<std::vector<int>> &excluded_components);

std::pair<GraphData, std::vector<int>> assign_terminals_randomly(
    const GraphData &graph_data,
    double budget_percent,
    int seed);

int compute_pc(
    const GraphData &graph_data,
    const std::unordered_set<int> &deleted_nodes,
    const std::vector<int> &terminal_nodes);

int compute_pc_no_S(
    const GraphData &graph_data,
    const std::vector<int> &terminal_nodes);

int obj(
    const GraphData &graph_data,
    const std::unordered_set<int> &kept_nodes,
    const std::vector<int> &terminal_nodes);

#endif
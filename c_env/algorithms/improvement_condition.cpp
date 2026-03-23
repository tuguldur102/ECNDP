#include "improvement_condition.hpp"

#include "utils.hpp"

bool improvement_condition(
    const GraphData &graph_data,
    const std::unordered_set<int> &K,
    int best_pc,
    const std::vector<int> &terminals)
{
  return obj(graph_data, K, terminals) < best_pc;
}
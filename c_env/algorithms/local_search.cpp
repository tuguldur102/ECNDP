#include "local_search.hpp"

#include "utils.hpp"

std::unordered_set<int> local_search_procedure(
    const GraphData &graph_data,
    const std::unordered_set<int> &S,
    const std::vector<int> &terminals,
    int case_value,
    const std::function<bool(
        const GraphData &,
        const std::unordered_set<int> &,
        int,
        const std::vector<int> &)> &improvement_condition)
{
  std::unordered_set<int> V(
      graph_data.nodes.begin(),
      graph_data.nodes.end());

  std::unordered_set<int> T(
      terminals.begin(),
      terminals.end());

  std::unordered_set<int> K;
  for (int node : V)
  {
    if (!S.count(node))
    {
      K.insert(node);
    }
  }

  int current_pc = obj(graph_data, K, terminals);
  bool local_improvement = true;

  std::vector<int> current_nodes;
  if (case_value == 1)
  {
    current_nodes = graph_data.nodes;
  }
  else
  {
    for (int node : graph_data.nodes)
    {
      if (!T.count(node))
      {
        current_nodes.push_back(node);
      }
    }
  }

  while (local_improvement)
  {
    local_improvement = false;
    std::unordered_set<int> best_K = K;

    for (int i : current_nodes)
    {
      for (int j : current_nodes)
      {
        if (K.count(i) && !K.count(j))
        {
          K.erase(i);
          K.insert(j);

          if (obj(graph_data, K, terminals) <
              obj(graph_data, best_K, terminals))
          {
            best_K = K;
          }
          else
          {
            K.erase(j);
            K.insert(i);
          }
        }
      }
    }

    if (improvement_condition(graph_data, best_K, current_pc, terminals))
    {
      K = best_K;
      local_improvement = true;
    }

    std::unordered_set<int> result;
    for (int node : V)
    {
      if (!K.count(node))
      {
        result.insert(node);
      }
    }
    return result;
  }

  std::unordered_set<int> result;
  for (int node : V)
  {
    if (!K.count(node))
    {
      result.insert(node);
    }
  }
  return result;
}
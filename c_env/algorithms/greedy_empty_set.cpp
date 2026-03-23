#include "greedy_empty_set.hpp"

#include <limits>

#include "improvement_condition.hpp"
#include "local_search.hpp"
#include "utils.hpp"

std::unordered_set<int> greedy_empty_set(
    const GraphData &graph_data,
    const std::vector<int> &terminals,
    int k,
    int case_value)
{
  (void)terminals;

  std::unordered_set<int> T(
      terminals.begin(),
      terminals.end());

  std::unordered_set<int> V(
      graph_data.nodes.begin(),
      graph_data.nodes.end());

  std::unordered_set<int> S;
  std::unordered_set<int> K;

  if (case_value == 1)
  {
    K.clear();
  }
  else if (case_value == 2)
  {
    K = T;
  }

  while (static_cast<int>(K.size()) != static_cast<int>(V.size()) - k)
  {
    int best_j = -1;
    int best_pc = std::numeric_limits<int>::max();

    for (int j : V)
    {
      if (K.count(j))
      {
        continue;
      }

      std::unordered_set<int> candidate_kept = K;
      candidate_kept.insert(j);

      std::unordered_set<int> S_j;
      for (int node : V)
      {
        if (!candidate_kept.count(node))
        {
          S_j.insert(node);
        }
      }

      int pc_j = compute_pc(graph_data, S_j, terminals);

      if (pc_j < best_pc)
      {
        best_pc = pc_j;
        best_j = j;
      }
    }

    if (best_j == -1)
    {
      break;
    }

    K.insert(best_j);
  }

  for (int node : V)
  {
    if (!K.count(node))
    {
      S.insert(node);
    }
  }

  return S;
}

std::pair<std::unordered_set<int>, int> extended_critical_node_empty_set(
    const GraphData &graph_data,
    const std::vector<int> &terminals,
    int k,
    int case_value,
    int max_iter,
    bool use_local_search)
{
  (void)max_iter;

  std::unordered_set<int> S;
  std::unordered_set<int> current_S;
  int best_pc = std::numeric_limits<int>::max();

  S = greedy_empty_set(graph_data, terminals, k, case_value);
  current_S = S;

  if (use_local_search)
  {
    std::unordered_set<int> S_ls = local_search_procedure(
        graph_data,
        S,
        terminals,
        case_value,
        improvement_condition);
    current_S = S_ls;
  }

  int current_pc = compute_pc(graph_data, current_S, terminals);
  return {current_S, current_pc};
}
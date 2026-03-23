#include "greedy_mis.hpp"

#include <iostream>
#include <limits>
#include <unordered_map>

#include "improvement_condition.hpp"
#include "local_search.hpp"
#include "utils.hpp"

namespace
{

  std::unordered_map<int, std::unordered_set<int>> build_adjacency_map(
      const GraphData &graph_data)
  {
    std::unordered_map<int, std::unordered_set<int>> adjacency_map;

    for (int node : graph_data.nodes)
    {
      adjacency_map[node] = {};
    }

    for (const auto &edge : graph_data.edges)
    {
      int u = edge.first;
      int v = edge.second;
      adjacency_map[u].insert(v);
      adjacency_map[v].insert(u);
    }

    return adjacency_map;
  }

  GraphData induced_subgraph_without_nodes(
      const GraphData &graph_data,
      const std::unordered_set<int> &removed_nodes)
  {
    GraphData subgraph;

    for (int node : graph_data.nodes)
    {
      if (!removed_nodes.count(node))
      {
        subgraph.nodes.push_back(node);
      }
    }

    for (const auto &edge : graph_data.edges)
    {
      int u = edge.first;
      int v = edge.second;
      if (!removed_nodes.count(u) && !removed_nodes.count(v))
      {
        subgraph.edges.push_back(edge);
      }
    }

    return subgraph;
  }

  std::unordered_set<int> maximal_independent_set_greedy(
      const GraphData &graph_data)
  {
    auto adjacency_map = build_adjacency_map(graph_data);
    std::unordered_set<int> independent_set;
    std::unordered_set<int> blocked_nodes;

    for (int node : graph_data.nodes)
    {
      if (blocked_nodes.count(node))
      {
        continue;
      }

      independent_set.insert(node);
      blocked_nodes.insert(node);

      for (int neighbor : adjacency_map[node])
      {
        blocked_nodes.insert(neighbor);
      }
    }

    return independent_set;
  }

} // namespace

std::unordered_set<int> greedy_mis(
    const GraphData &graph_data,
    const std::vector<int> &terminals,
    int k,
    int case_value)
{
  std::unordered_set<int> T(
      terminals.begin(),
      terminals.end());

  std::unordered_set<int> V(
      graph_data.nodes.begin(),
      graph_data.nodes.end());

  std::unordered_set<int> S;

  if (case_value == 1)
  {
    std::unordered_set<int> MIS =
        maximal_independent_set_greedy(graph_data);

    while (static_cast<int>(MIS.size()) < static_cast<int>(V.size()) - k)
    {
      int best_j = -1;
      int best_pc = std::numeric_limits<int>::max();

      for (int j : V)
      {
        if (MIS.count(j))
        {
          continue;
        }

        std::unordered_set<int> candidate_mis = MIS;
        candidate_mis.insert(j);

        std::unordered_set<int> S_j;
        for (int node : V)
        {
          if (!candidate_mis.count(node))
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

      MIS.insert(best_j);
    }

    for (int node : V)
    {
      if (!MIS.count(node))
      {
        S.insert(node);
      }
    }

    return S;
  }

  if (case_value == 2)
  {
    std::unordered_set<int> U;
    for (int node : V)
    {
      if (!T.count(node))
      {
        U.insert(node);
      }
    }

    GraphData H = induced_subgraph_without_nodes(graph_data, T);
    std::unordered_set<int> MIS = maximal_independent_set_greedy(H);

    int r = static_cast<int>(V.size()) - static_cast<int>(T.size()) - k;

    while (static_cast<int>(MIS.size()) > r)
    {
      int best_remove = -1;
      int best_pc = std::numeric_limits<int>::max();

      for (int j : MIS)
      {
        std::unordered_set<int> kept = T;
        for (int node : MIS)
        {
          if (node != j)
          {
            kept.insert(node);
          }
        }

        std::unordered_set<int> S_j;
        for (int node : V)
        {
          if (!kept.count(node))
          {
            S_j.insert(node);
          }
        }

        int pc_j = compute_pc(graph_data, S_j, terminals);

        if (pc_j < best_pc)
        {
          best_pc = pc_j;
          best_remove = j;
        }
      }

      if (best_remove == -1)
      {
        break;
      }

      MIS.erase(best_remove);
    }

    while (static_cast<int>(MIS.size()) < r)
    {
      int best_add = -1;
      int best_pc = std::numeric_limits<int>::max();

      for (int j : U)
      {
        if (MIS.count(j))
        {
          continue;
        }

        std::unordered_set<int> kept = T;
        for (int node : MIS)
        {
          kept.insert(node);
        }
        kept.insert(j);

        std::unordered_set<int> S_j;
        for (int node : V)
        {
          if (!kept.count(node))
          {
            S_j.insert(node);
          }
        }

        int pc_j = compute_pc(graph_data, S_j, terminals);

        if (pc_j < best_pc)
        {
          best_pc = pc_j;
          best_add = j;
        }
      }

      if (best_add == -1)
      {
        break;
      }

      MIS.insert(best_add);
    }

    for (int node : U)
    {
      if (!MIS.count(node))
      {
        S.insert(node);
      }
    }

    return S;
  }

  return S;
}

std::pair<std::unordered_set<int>, int> extended_critical_node_mis(
    const GraphData &graph_data,
    const std::vector<int> &terminals,
    int k,
    int case_value,
    int max_iter,
    bool use_local_search)
{
  std::unordered_set<int> best_S;
  int best_pc = std::numeric_limits<int>::max();

  for (int iteration_idx = 0; iteration_idx < max_iter; ++iteration_idx)
  {
    std::unordered_set<int> current_S = greedy_mis(
        graph_data,
        terminals,
        k,
        case_value);

    if (use_local_search)
    {
      std::unordered_set<int> S_ls = local_search_procedure(
          graph_data,
          current_S,
          terminals,
          case_value,
          improvement_condition);
      current_S = S_ls;
    }

    int current_pc = compute_pc(graph_data, current_S, terminals);

    if (current_pc < best_pc)
    {
      best_pc = current_pc;
      best_S = current_S;
      std::cout << "Refined PC: " << best_pc << "\n";
    }
  }

  return {best_S, best_pc};
}
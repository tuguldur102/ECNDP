#include "utils.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <unordered_map>

std::vector<std::vector<int>> build_adjacency_list(
    const GraphData &graph_data)
{
  std::unordered_map<int, int> node_to_position;
  int total_nodes = static_cast<int>(graph_data.nodes.size());

  for (int idx = 0; idx < total_nodes; ++idx)
  {
    node_to_position[graph_data.nodes[idx]] = idx;
  }

  std::vector<std::vector<int>> adjacency_list(total_nodes);

  for (const auto &edge : graph_data.edges)
  {
    int source = edge.first;
    int target = edge.second;

    int source_position = node_to_position.at(source);
    int target_position = node_to_position.at(target);

    adjacency_list[source_position].push_back(target);
    adjacency_list[target_position].push_back(source);
  }

  return adjacency_list;
}

std::vector<std::vector<int>> connected_components(
    const GraphData &graph_data)
{
  std::vector<std::vector<int>> adjacency_list =
      build_adjacency_list(graph_data);

  std::unordered_map<int, int> node_to_position;
  int total_nodes = static_cast<int>(graph_data.nodes.size());

  for (int idx = 0; idx < total_nodes; ++idx)
  {
    node_to_position[graph_data.nodes[idx]] = idx;
  }

  std::vector<bool> visited(total_nodes, false);
  std::vector<std::vector<int>> all_components;

  for (int idx = 0; idx < total_nodes; ++idx)
  {
    if (visited[idx])
    {
      continue;
    }

    int start_node = graph_data.nodes[idx];
    std::queue<int> bfs_queue;
    std::vector<int> current_component;

    visited[idx] = true;
    bfs_queue.push(start_node);

    while (!bfs_queue.empty())
    {
      int current_node = bfs_queue.front();
      bfs_queue.pop();

      current_component.push_back(current_node);

      for (int neighbor : adjacency_list[node_to_position[current_node]])
      {
        int neighbor_position = node_to_position[neighbor];
        if (!visited[neighbor_position])
        {
          visited[neighbor_position] = true;
          bfs_queue.push(neighbor);
        }
      }
    }

    all_components.push_back(current_component);
  }

  return all_components;
}

std::vector<std::vector<int>> connected_components_after_deletion(
    const GraphData &graph_data,
    const std::unordered_set<int> &deleted_nodes)
{
  std::vector<int> remaining_nodes;
  remaining_nodes.reserve(graph_data.nodes.size());

  for (int node : graph_data.nodes)
  {
    if (!deleted_nodes.count(node))
    {
      remaining_nodes.push_back(node);
    }
  }

  std::vector<std::pair<int, int>> remaining_edges;
  remaining_edges.reserve(graph_data.edges.size());

  for (const auto &edge : graph_data.edges)
  {
    int source = edge.first;
    int target = edge.second;

    if (!deleted_nodes.count(source) && !deleted_nodes.count(target))
    {
      remaining_edges.push_back(edge);
    }
  }

  GraphData reduced_graph_data{remaining_nodes, remaining_edges};

  return connected_components(reduced_graph_data);
}

int comb_of_two(
    const std::vector<std::vector<int>> &excluded_components)
{
  int total_pairwise_connectivity = 0;

  for (const auto &component : excluded_components)
  {
    int component_size = static_cast<int>(component.size());
    total_pairwise_connectivity +=
        component_size * (component_size - 1) / 2;
  }

  return total_pairwise_connectivity;
}

std::pair<GraphData, std::vector<int>> assign_terminals_randomly(
    const GraphData &graph_data,
    double budget_percent,
    int seed)
{
  std::vector<int> nodes = graph_data.nodes;
  int budget = static_cast<int>(nodes.size() * budget_percent);

  std::cout << "budget " << budget << "\n";

  std::mt19937 random_generator(seed);
  std::shuffle(nodes.begin(), nodes.end(), random_generator);

  std::vector<int> terminal_nodes;
  terminal_nodes.reserve(budget);

  for (int idx = 0; idx < budget; ++idx)
  {
    terminal_nodes.push_back(nodes[idx]);
  }

  std::cout << terminal_nodes.size() << "\n";

  return {graph_data, terminal_nodes};
}

int compute_pc(
    const GraphData &graph_data,
    const std::unordered_set<int> &deleted_nodes,
    const std::vector<int> &terminal_nodes)
{
  std::unordered_set<int> terminal_set(
      terminal_nodes.begin(),
      terminal_nodes.end());

  std::vector<std::vector<int>> components =
      connected_components_after_deletion(graph_data, deleted_nodes);

  std::vector<std::vector<int>> excluded_components;

  for (const auto &component : components)
  {
    std::vector<int> excluded_component;

    for (int node : component)
    {
      if (terminal_set.count(node))
      {
        excluded_component.push_back(node);
      }
    }

    excluded_components.push_back(excluded_component);
  }

  return comb_of_two(excluded_components);
}

int compute_pc_no_S(
    const GraphData &graph_data,
    const std::vector<int> &terminal_nodes)
{
  std::unordered_set<int> terminal_set(
      terminal_nodes.begin(),
      terminal_nodes.end());

  std::vector<std::vector<int>> components =
      connected_components(graph_data);

  std::vector<std::vector<int>> excluded_components;

  for (const auto &component : components)
  {
    std::vector<int> excluded_component;

    for (int node : component)
    {
      if (terminal_set.count(node))
      {
        excluded_component.push_back(node);
      }
    }

    excluded_components.push_back(excluded_component);
  }

  return comb_of_two(excluded_components);
}

int obj(
    const GraphData &graph_data,
    const std::unordered_set<int> &kept_nodes,
    const std::vector<int> &terminal_nodes)
{
  std::unordered_set<int> deleted_nodes;

  for (int node : graph_data.nodes)
  {
    if (!kept_nodes.count(node))
    {
      deleted_nodes.insert(node);
    }
  }

  return compute_pc(graph_data, deleted_nodes, terminal_nodes);
}
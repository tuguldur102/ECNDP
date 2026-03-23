#include "exact.hpp"
#include "absl/time/time.h"

using operations_research::MPConstraint;
using operations_research::MPObjective;
using operations_research::MPSolver;
using operations_research::MPVariable;

std::string to_lower_copy(std::string text)
{
  std::transform(
      text.begin(),
      text.end(),
      text.begin(),
      [](unsigned char character)
      { return static_cast<char>(std::tolower(character)); });
  return text;
}

std::vector<int> ordered_unique_nodes(std::vector<int> nodes)
{
  std::sort(nodes.begin(), nodes.end());
  nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
  return nodes;
}

std::vector<std::pair<int, int>> ordered_unique_edges(
    const std::vector<std::pair<int, int>> &edges,
    const std::unordered_set<int> &node_set)
{
  std::set<std::pair<int, int>> unique_edges;

  for (const auto &edge : edges)
  {
    int left_node = edge.first;
    int right_node = edge.second;

    if (left_node == right_node)
    {
      throw std::runtime_error("The formulation assumes a simple graph (no self-loops).");
    }

    if (!node_set.count(left_node) || !node_set.count(right_node))
    {
      throw std::runtime_error("Every edge endpoint must belong to graph_data.nodes.");
    }

    if (right_node < left_node)
    {
      std::swap(left_node, right_node);
    }

    unique_edges.insert({left_node, right_node});
  }

  return std::vector<std::pair<int, int>>(unique_edges.begin(), unique_edges.end());
}

std::unique_ptr<MPSolver> create_mip_solver_or_throw(const std::string &solver_name)
{
  const std::string normalized_solver_name = to_lower_copy(solver_name);

  std::string backend_name;
  if (normalized_solver_name == "cbc")
  {
    backend_name = "CBC";
  }
  else if (normalized_solver_name == "glpk")
  {
    backend_name = "GLPK";
  }
  else
  {
    throw std::runtime_error("solver_name must be one of: 'cbc', 'glpk'");
  }

  std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver(backend_name));
  if (!solver)
  {
    throw std::runtime_error(
        "Could not create solver '" + backend_name +
        "'. The backend may not be linked in your OR-Tools build.");
  }

  return solver;
}

std::pair<int, int> ordered_pair_by_position(
    const std::unordered_map<int, int> &node_to_position,
    int node_i,
    int node_j)
{
  if (node_to_position.at(node_i) < node_to_position.at(node_j))
  {
    return {node_i, node_j};
  }

  return {node_j, node_i};
}

const MPVariable *pairwise_connectivity_variable(
    const EcndpModelData &model_data,
    int node_i,
    int node_j)
{
  if (node_i == node_j)
  {
    throw std::runtime_error("Diagonal terms do not have explicit x_ii variables.");
  }

  const auto ordered_pair =
      ordered_pair_by_position(model_data.node_to_position, node_i, node_j);

  return model_data.pairwise_connectivity_variables.at(ordered_pair);
}

double pairwise_connectivity_solution_value(
    const EcndpModelData &model_data,
    int node_i,
    int node_j)
{
  if (node_i == node_j)
  {
    return 1.0 - model_data.deleted_node_variables.at(node_i)->solution_value();
  }

  return pairwise_connectivity_variable(model_data, node_i, node_j)->solution_value();
}

std::vector<std::vector<int>> connected_components_after_deletion_exact(
    const GraphData &graph_data,
    const std::unordered_set<int> &deleted_node_set)
{
  std::unordered_map<int, std::vector<int>> adjacency;

  for (int node : graph_data.nodes)
  {
    if (!deleted_node_set.count(node))
    {
      adjacency[node] = {};
    }
  }

  for (const auto &edge : graph_data.edges)
  {
    const int left_node = edge.first;
    const int right_node = edge.second;

    if (deleted_node_set.count(left_node) || deleted_node_set.count(right_node))
    {
      continue;
    }

    adjacency[left_node].push_back(right_node);
    adjacency[right_node].push_back(left_node);
  }

  std::vector<std::vector<int>> components;
  std::unordered_set<int> visited_nodes;

  for (int start_node : graph_data.nodes)
  {
    if (deleted_node_set.count(start_node) || visited_nodes.count(start_node))
    {
      continue;
    }

    std::vector<int> current_component;
    std::queue<int> bfs_queue;
    bfs_queue.push(start_node);
    visited_nodes.insert(start_node);

    while (!bfs_queue.empty())
    {
      const int current_node = bfs_queue.front();
      bfs_queue.pop();
      current_component.push_back(current_node);

      for (int next_node : adjacency[current_node])
      {
        if (!visited_nodes.count(next_node))
        {
          visited_nodes.insert(next_node);
          bfs_queue.push(next_node);
        }
      }
    }

    std::sort(current_component.begin(), current_component.end());
    components.push_back(current_component);
  }

  return components;
}

std::string solver_status_to_string(MPSolver::ResultStatus status)
{
  switch (status)
  {
  case MPSolver::OPTIMAL:
    return "Optimal";
  case MPSolver::FEASIBLE:
    return "Feasible";
  case MPSolver::INFEASIBLE:
    return "Infeasible";
  case MPSolver::UNBOUNDED:
    return "Unbounded";
  case MPSolver::ABNORMAL:
    return "Abnormal";
  case MPSolver::MODEL_INVALID:
    return "ModelInvalid";
  case MPSolver::NOT_SOLVED:
    return "NotSolved";
  default:
    return "Unknown";
  }
}

EcndpModelData build_ecndp_model_ortools(
    const GraphData &graph_data,
    const std::vector<int> &terminals,
    int K,
    bool allow_terminal_deletion,
    bool include_diagonal_in_objective,
    const std::string &solver_name)
{
  if (K < 0)
  {
    throw std::runtime_error("K must be nonnegative.");
  }

  EcndpModelData model_data;
  model_data.nodes = ordered_unique_nodes(graph_data.nodes);
  model_data.terminals = ordered_unique_nodes(terminals);
  model_data.terminal_set = std::unordered_set<int>(
      model_data.terminals.begin(),
      model_data.terminals.end());
  model_data.include_diagonal_in_objective = include_diagonal_in_objective;

  const std::unordered_set<int> node_set(
      model_data.nodes.begin(),
      model_data.nodes.end());

  for (int terminal : model_data.terminals)
  {
    if (!node_set.count(terminal))
    {
      throw std::runtime_error("All terminals must belong to the graph.");
    }
  }

  const auto unique_edges = ordered_unique_edges(graph_data.edges, node_set);

  model_data.solver = create_mip_solver_or_throw(solver_name);
  const double infinity = MPSolver::infinity();

  for (int position = 0; position < static_cast<int>(model_data.nodes.size()); ++position)
  {
    model_data.node_to_position[model_data.nodes[position]] = position;
  }

  for (int node : model_data.nodes)
  {
    const int position = model_data.node_to_position.at(node);
    model_data.deleted_node_variables[node] = model_data.solver->MakeIntVar(
        0.0,
        1.0,
        "s_" + std::to_string(position));
  }

  for (int left_position = 0; left_position < static_cast<int>(model_data.nodes.size()); ++left_position)
  {
    const int left_node = model_data.nodes[left_position];

    for (int right_position = left_position + 1;
         right_position < static_cast<int>(model_data.nodes.size());
         ++right_position)
    {
      const int right_node = model_data.nodes[right_position];

      model_data.pairwise_connectivity_variables[{left_node, right_node}] =
          model_data.solver->MakeIntVar(
              0.0,
              1.0,
              "x_" + std::to_string(left_position) + "_" + std::to_string(right_position));
    }
  }

  MPObjective *objective = model_data.solver->MutableObjective();
  objective->SetMinimization();

  if (include_diagonal_in_objective)
  {
    objective->SetOffset(static_cast<double>(model_data.terminals.size()));
    for (int terminal : model_data.terminals)
    {
      objective->SetCoefficient(model_data.deleted_node_variables.at(terminal), -1.0);
    }
  }

  for (int left_position = 0; left_position < static_cast<int>(model_data.terminals.size()); ++left_position)
  {
    for (int right_position = left_position + 1;
         right_position < static_cast<int>(model_data.terminals.size());
         ++right_position)
    {
      const int left_terminal = model_data.terminals[left_position];
      const int right_terminal = model_data.terminals[right_position];

      objective->SetCoefficient(
          pairwise_connectivity_variable(model_data, left_terminal, right_terminal),
          1.0);
    }
  }

  {
    MPConstraint *budget_constraint =
        model_data.solver->MakeRowConstraint(-infinity, static_cast<double>(K), "budget");

    if (allow_terminal_deletion)
    {
      for (int node : model_data.nodes)
      {
        budget_constraint->SetCoefficient(model_data.deleted_node_variables.at(node), 1.0);
      }
    }
    else
    {
      for (int node : model_data.nodes)
      {
        if (!model_data.terminal_set.count(node))
        {
          budget_constraint->SetCoefficient(model_data.deleted_node_variables.at(node), 1.0);
        }
      }

      for (int terminal : model_data.terminals)
      {
        MPConstraint *protect_terminal_constraint =
            model_data.solver->MakeRowConstraint(0.0, 0.0);
        protect_terminal_constraint->SetCoefficient(
            model_data.deleted_node_variables.at(terminal),
            1.0);
      }
    }
  }

  for (const auto &edge : unique_edges)
  {
    const int left_node = edge.first;
    const int right_node = edge.second;

    MPConstraint *edge_constraint =
        model_data.solver->MakeRowConstraint(1.0, infinity);

    edge_constraint->SetCoefficient(
        pairwise_connectivity_variable(model_data, left_node, right_node),
        1.0);
    edge_constraint->SetCoefficient(model_data.deleted_node_variables.at(left_node), 1.0);
    edge_constraint->SetCoefficient(model_data.deleted_node_variables.at(right_node), 1.0);
  }

  for (int left_position = 0; left_position < static_cast<int>(model_data.nodes.size()); ++left_position)
  {
    const int left_node = model_data.nodes[left_position];

    for (int right_position = left_position + 1;
         right_position < static_cast<int>(model_data.nodes.size());
         ++right_position)
    {
      const int right_node = model_data.nodes[right_position];
      const MPVariable *pairwise_variable =
          pairwise_connectivity_variable(model_data, left_node, right_node);

      MPConstraint *upper_bound_left_constraint =
          model_data.solver->MakeRowConstraint(-infinity, 1.0);
      upper_bound_left_constraint->SetCoefficient(pairwise_variable, 1.0);
      upper_bound_left_constraint->SetCoefficient(
          model_data.deleted_node_variables.at(left_node),
          1.0);

      MPConstraint *upper_bound_right_constraint =
          model_data.solver->MakeRowConstraint(-infinity, 1.0);
      upper_bound_right_constraint->SetCoefficient(pairwise_variable, 1.0);
      upper_bound_right_constraint->SetCoefficient(
          model_data.deleted_node_variables.at(right_node),
          1.0);
    }
  }

  for (int first_position = 0; first_position < static_cast<int>(model_data.nodes.size()); ++first_position)
  {
    const int first_node = model_data.nodes[first_position];

    for (int second_position = first_position + 1;
         second_position < static_cast<int>(model_data.nodes.size());
         ++second_position)
    {
      const int second_node = model_data.nodes[second_position];

      for (int third_position = second_position + 1;
           third_position < static_cast<int>(model_data.nodes.size());
           ++third_position)
      {
        const int third_node = model_data.nodes[third_position];

        const MPVariable *first_second_variable =
            pairwise_connectivity_variable(model_data, first_node, second_node);
        const MPVariable *first_third_variable =
            pairwise_connectivity_variable(model_data, first_node, third_node);
        const MPVariable *second_third_variable =
            pairwise_connectivity_variable(model_data, second_node, third_node);

        MPConstraint *transitivity_constraint_1 =
            model_data.solver->MakeRowConstraint(-infinity, 1.0);
        transitivity_constraint_1->SetCoefficient(first_second_variable, 1.0);
        transitivity_constraint_1->SetCoefficient(second_third_variable, 1.0);
        transitivity_constraint_1->SetCoefficient(first_third_variable, -1.0);

        MPConstraint *transitivity_constraint_2 =
            model_data.solver->MakeRowConstraint(-infinity, 1.0);
        transitivity_constraint_2->SetCoefficient(first_third_variable, 1.0);
        transitivity_constraint_2->SetCoefficient(second_third_variable, 1.0);
        transitivity_constraint_2->SetCoefficient(first_second_variable, -1.0);

        MPConstraint *transitivity_constraint_3 =
            model_data.solver->MakeRowConstraint(-infinity, 1.0);
        transitivity_constraint_3->SetCoefficient(first_second_variable, 1.0);
        transitivity_constraint_3->SetCoefficient(first_third_variable, 1.0);
        transitivity_constraint_3->SetCoefficient(second_third_variable, -1.0);
      }
    }
  }

  return model_data;
}

EcndpSolveResult solve_ecndp_ortools(
    const GraphData &graph_data,
    const std::vector<int> &terminals,
    int K,
    bool allow_terminal_deletion,
    bool include_diagonal_in_objective,
    const std::string &solver_name,
    std::optional<double> time_limit_seconds,
    bool verbose)
{
  EcndpModelData model_data = build_ecndp_model_ortools(
      graph_data,
      terminals,
      K,
      allow_terminal_deletion,
      include_diagonal_in_objective,
      solver_name);

  if (verbose)
  {
    model_data.solver->EnableOutput();
  }
  else
  {
    model_data.solver->SuppressOutput();
  }

  if (time_limit_seconds.has_value())
  {
    model_data.solver->SetTimeLimit(absl::Seconds(*time_limit_seconds));
  }

  const MPSolver::ResultStatus solve_status = model_data.solver->Solve();
  const std::string status_string = solver_status_to_string(solve_status);

  EcndpSolveResult result;
  result.status = status_string;
  result.solver = std::move(model_data.solver);

  if (solve_status != MPSolver::OPTIMAL && solve_status != MPSolver::FEASIBLE)
  {
    return result;
  }

  for (int node : model_data.nodes)
  {
    const double deleted_value =
        model_data.deleted_node_variables.at(node)->solution_value();

    if (deleted_value >= 0.99)
    {
      result.deleted_nodes.push_back(node);
    }
  }

  for (int left_position = 0; left_position < static_cast<int>(model_data.terminals.size()); ++left_position)
  {
    const int left_terminal = model_data.terminals[left_position];

    if (include_diagonal_in_objective)
    {
      result.pairwise_connectivity_solution[{left_terminal, left_terminal}] =
          static_cast<int>(std::llround(
              pairwise_connectivity_solution_value(model_data, left_terminal, left_terminal)));
    }

    for (int right_position = left_position + 1;
         right_position < static_cast<int>(model_data.terminals.size());
         ++right_position)
    {
      const int right_terminal = model_data.terminals[right_position];

      result.pairwise_connectivity_solution[{left_terminal, right_terminal}] =
          static_cast<int>(std::llround(
              pairwise_connectivity_solution_value(model_data, left_terminal, right_terminal)));
    }
  }

  result.objective = result.solver->Objective().Value();

  const std::unordered_set<int> deleted_node_set(
      result.deleted_nodes.begin(),
      result.deleted_nodes.end());
  result.components_after_deletion =
      connected_components_after_deletion_exact(graph_data, deleted_node_set);

  return result;
}
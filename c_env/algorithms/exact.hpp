#ifndef EXACT_HPP
#define EXACT_HPP

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ortools/linear_solver/linear_solver.h"

struct GraphData
{
  std::vector<int> nodes;
  std::vector<std::pair<int, int>> edges;
};

struct EcndpModelData
{
  std::unique_ptr<operations_research::MPSolver> solver;
  std::vector<int> nodes;
  std::vector<int> terminals;
  std::unordered_set<int> terminal_set;
  std::unordered_map<int, int> node_to_position;
  std::unordered_map<int, const operations_research::MPVariable *> deleted_node_variables;
  std::map<std::pair<int, int>, const operations_research::MPVariable *> pairwise_connectivity_variables;
  bool include_diagonal_in_objective = false;
};

struct EcndpSolveResult
{
  std::string status;
  std::optional<double> objective;
  std::vector<int> deleted_nodes;
  std::map<std::pair<int, int>, int> pairwise_connectivity_solution;
  std::vector<std::vector<int>> components_after_deletion;
  std::unique_ptr<operations_research::MPSolver> solver;
};

EcndpModelData build_ecndp_model_ortools(
    const GraphData &graph_data,
    const std::vector<int> &terminals,
    int K,
    bool allow_terminal_deletion = true,
    bool include_diagonal_in_objective = false,
    const std::string &solver_name = "cbc");

EcndpSolveResult solve_ecndp_ortools(
    const GraphData &graph_data,
    const std::vector<int> &terminals,
    int K,
    bool allow_terminal_deletion = true,
    bool include_diagonal_in_objective = false,
    const std::string &solver_name = "cbc",
    std::optional<double> time_limit_seconds = std::nullopt,
    bool verbose = false);

#endif
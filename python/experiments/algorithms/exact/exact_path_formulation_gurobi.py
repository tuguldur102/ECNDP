import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from itertools import combinations
import os
from algorithms.compute_PC import compute_pc
import time

def gurobi_status_to_string(status_code):
  status_map = {
    GRB.OPTIMAL: "OPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.INTERRUPTED: "INTERRUPTED",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.INF_OR_UNBD: "INF_OR_UNBD"
  }
  return status_map.get(status_code, f"STATUS_{status_code}")


def build_remaining_graph(graph, deleted_nodes):
  remaining_nodes = [node for node in graph.nodes() if node not in deleted_nodes]
  return graph.subgraph(remaining_nodes).copy()


def extract_solution_data(
  model,
  graph,
  terminals,
  node_removed,
  pairwise_connectivity
):
  solution = {
    "status_code": model.Status,
    "status": gurobi_status_to_string(model.Status),
    "objective": None,
    "deleted_nodes": [],
    "pairwise_connectivity": {},
    "connected_terminal_pairs": [],
    "components_after_deletion": []
  }

  if model.SolCount == 0:
    return solution

  solution["objective"] = model.ObjVal

  deleted_nodes = [
    node for node in graph.nodes()
    if node_removed[node].X > 0.5
  ]
  solution["deleted_nodes"] = deleted_nodes

  terminal_pairs = [(i, j) for i, j in combinations(sorted(terminals), 2)]

  pairwise_connectivity_values = {}
  connected_terminal_pairs = []

  for i, j in terminal_pairs:
    x_value = int(round(pairwise_connectivity[i, j].X))
    pairwise_connectivity_values[(i, j)] = x_value
    if x_value == 1:
      connected_terminal_pairs.append((i, j))

  solution["pairwise_connectivity"] = pairwise_connectivity_values
  solution["connected_terminal_pairs"] = connected_terminal_pairs

  remaining_graph = build_remaining_graph(graph, deleted_nodes)
  components = [sorted(component) for component in nx.connected_components(remaining_graph)]
  components.sort(key=lambda component_nodes: (len(component_nodes), component_nodes))

  solution["components_after_deletion"] = components

  return solution


def solve_ecndp_path_formulation(
  graph,
  terminals,
  budget,
  protect_terminals=False,
  license_path=None,
  time_limit=None,
  mip_gap=None,
  verbose=True
):
  """
  Solve ECNDP with path-based lazy constraints.

  Parameters
  ----------
  graph : networkx.Graph
    Undirected graph.
  terminals : list
    Terminal nodes.
  budget : int
    Maximum number of deleted nodes.
  protect_terminals : bool
    False -> Case 1 (terminals deletable)
    True  -> Case 2 (terminals protected)
  license_path : str or None
    Optional full path to gurobi.lic.
    Must be set before model creation.
  time_limit : float or None
    Time limit in seconds.
  mip_gap : float or None
    Relative MIP gap.
  verbose : bool
    Show Gurobi log or not.
  """

  start = time.perf_counter()

  if license_path is not None:
    os.environ["GRB_LICENSE_FILE"] = license_path

  node_list = list(graph.nodes())
  terminals = sorted(list(terminals))
  terminal_set = set(terminals)

  if not terminal_set.issubset(set(node_list)):
    raise ValueError("All terminals must belong to graph nodes.")

  if budget < 0:
    raise ValueError("Budget must be nonnegative.")

  terminal_pairs = [(i, j) for i, j in combinations(terminals, 2)]

  model = gp.Model("ECNDP_path_lazy")

  if not verbose:
    model.Params.OutputFlag = 0
  if time_limit is not None:
    model.Params.TimeLimit = time_limit
  if mip_gap is not None:
    model.Params.MIPGap = mip_gap

  model.Params.LazyConstraints = 1

  node_removed = model.addVars(
    node_list,
    vtype=GRB.BINARY,
    name="s"
  )

  pairwise_connectivity = model.addVars(
    terminal_pairs,
    vtype=GRB.BINARY,
    name="x"
  )

  model.setObjective(
    gp.quicksum(pairwise_connectivity[i, j] for i, j in terminal_pairs),
    GRB.MINIMIZE
  )

  model.addConstr(
    gp.quicksum(node_removed[node] for node in node_list) <= budget,
    name="budget"
  )

  if protect_terminals:
    for terminal_node in terminals:
      model.addConstr(
        node_removed[terminal_node] == 0,
        name=f"protect_terminal[{terminal_node}]"
      )

  for i, j in terminal_pairs:
    model.addConstr(
      pairwise_connectivity[i, j] <= 1 - node_removed[i],
      name=f"endpoint_i_removed[{i},{j}]"
    )
    model.addConstr(
      pairwise_connectivity[i, j] <= 1 - node_removed[j],
      name=f"endpoint_j_removed[{i},{j}]"
    )

  model._graph = graph
  model._terminals = terminals
  model._terminal_pairs = terminal_pairs
  model._node_removed = node_removed
  model._pairwise_connectivity = pairwise_connectivity

  def lazy_separator(callback_model, where):
    if where != GRB.Callback.MIPSOL:
      return

    graph_data = callback_model._graph
    terminal_pairs_data = callback_model._terminal_pairs
    node_removed_data = callback_model._node_removed
    pairwise_connectivity_data = callback_model._pairwise_connectivity

    node_removed_values = {
      node: callback_model.cbGetSolution(node_removed_data[node])
      for node in graph_data.nodes()
    }

    surviving_nodes = [
      node for node in graph_data.nodes()
      if node_removed_values[node] < 0.5
    ]
    surviving_graph = graph_data.subgraph(surviving_nodes)

    for i, j in terminal_pairs_data:
      x_value = callback_model.cbGetSolution(pairwise_connectivity_data[i, j])

      if x_value > 0.5:
        continue

      if node_removed_values[i] > 0.5 or node_removed_values[j] > 0.5:
        continue

      if nx.has_path(surviving_graph, i, j):
        violating_path = nx.shortest_path(surviving_graph, source=i, target=j)

        callback_model.cbLazy(
          pairwise_connectivity_data[i, j] +
          gp.quicksum(node_removed_data[node] for node in violating_path) >= 1
        )

  model.optimize(lazy_separator)

  end = time.perf_counter()
  total_time = end - start

  solution = extract_solution_data(
    model=model,
    graph=graph,
    terminals=terminals,
    node_removed=node_removed,
    pairwise_connectivity=pairwise_connectivity
  )

  return model, solution, total_time

def solve_exact_path(graph, terminals, budget, case, license_path):

  if case == 1:
    model_case_1, solution_case_1, total_time_1 = solve_ecndp_path_formulation(
      graph=graph,
      terminals=terminals,
      budget=budget,
      license_path=license_path,
      protect_terminals=False,
      verbose=False
    )

    if solution_case_1["status"] == "OPTIMAL":
      # S = solution_case_1["deleted_nodes"]
      optimal_pc = solution_case_1["objective"]

      return optimal_pc, total_time_1
    else:
      return None
    
  elif case == 2:
    # Case 2: terminals are protected
    model_case_2, solution_case_2, total_time_2 = solve_ecndp_path_formulation(
      graph=graph,
      terminals=terminals,
      budget=budget,
      license_path=license_path,
      protect_terminals=True,
      verbose=False
    )

    if solution_case_2["status"] == "OPTIMAL":
      S = solution_case_2["deleted_nodes"]
      optimal_pc = S = solution_case_2["objective"]

      return optimal_pc, total_time_2
    else:
      return None

    


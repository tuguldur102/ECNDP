import math
import time
import networkx as nx
import pulp as pl
from tqdm import tqdm


def _ordered_nodes(nodes):
  try:
    return sorted(nodes)
  except TypeError:
    return list(nodes)


def _all_simple_terminal_paths(G: nx.Graph, terminals):
  terminal_pairs = []
  terminal_paths = {}

  ordered_terminals = _ordered_nodes(set(terminals))

  for left_position in range(len(ordered_terminals)):
    source_node = ordered_terminals[left_position]
    for right_position in range(left_position + 1, len(ordered_terminals)):
      target_node = ordered_terminals[right_position]
      pair = (source_node, target_node)

      simple_paths = [
        list(path_nodes)
        for path_nodes in nx.all_simple_paths(G, source=source_node, target=target_node)
      ]

      terminal_pairs.append(pair)
      terminal_paths[pair] = simple_paths

  return terminal_pairs, terminal_paths


def build_ecndp_path_model_pulp(
  G: nx.Graph,
  terminals,
  K: int,
  allow_terminal_deletion: bool = True,
  include_diagonal_in_objective: bool = False,
  model_name: str = "ECNDP_Path",
  show_progress: bool = False
):
  if nx.number_of_selfloops(G) > 0:
    raise ValueError("The formulation assumes a simple graph (no self-loops).")

  V = _ordered_nodes(G.nodes())
  T = _ordered_nodes(set(terminals))
  V_set = set(V)
  T_set = set(T)

  if not T_set.issubset(V_set):
    raise ValueError("All terminals must belong to the graph.")
  if K < 0:
    raise ValueError("K must be nonnegative.")

  model = pl.LpProblem(model_name, pl.LpMinimize)

  node_to_position = {node: position for position, node in enumerate(V)}
  terminal_pairs, terminal_paths = _all_simple_terminal_paths(G, T)

  num_pair_variables = len(terminal_pairs)
  num_path_constraints = sum(
    len(terminal_paths[pair])
    for pair in terminal_pairs
  )

  total_progress_steps = (
    num_pair_variables +      # x variable creation
    num_path_constraints      # path constraints
  )

  progress_bar = None
  if show_progress:
    print(f"Building path model with {total_progress_steps} progress steps...")
    progress_bar = tqdm(
      total=total_progress_steps,
      desc="Building ECNDP path model",
      unit="step",
      leave=True,
      dynamic_ncols=True,
      miniters=1
    )

  # Deletion variables: s_i = 1 if vertex i is removed
  s = {
    node: pl.LpVariable(
      f"s_{node_to_position[node]}",
      lowBound=0,
      upBound=1,
      cat=pl.LpBinary
    )
    for node in V
  }

  # Connectivity variables only for terminal pairs i < j
  x = {}
  for source_node, target_node in terminal_pairs:
    source_position = node_to_position[source_node]
    target_position = node_to_position[target_node]

    x[(source_node, target_node)] = pl.LpVariable(
      f"x_{source_position}_{target_position}",
      lowBound=0,
      upBound=1,
      cat=pl.LpBinary
    )

    if progress_bar is not None:
      progress_bar.update(1)

  def X(node_i, node_j):
    if node_i == node_j:
      return 1 - s[node_i]

    position_i = node_to_position[node_i]
    position_j = node_to_position[node_j]

    if position_i < position_j:
      return x[(node_i, node_j)]
    return x[(node_j, node_i)]

  # Objective: minimize pairwise connectivity among terminals
  if include_diagonal_in_objective:
    model += (
      pl.lpSum(X(terminal, terminal) for terminal in T) +
      pl.lpSum(X(source_node, target_node) for source_node, target_node in terminal_pairs)
    )
  else:
    model += pl.lpSum(
      X(source_node, target_node)
      for source_node, target_node in terminal_pairs
    )

  # Budget / terminal protection
  if allow_terminal_deletion:
    model += pl.lpSum(s[node] for node in V) <= K, "budget"
  else:
    model += pl.lpSum(s[node] for node in V if node not in T_set) <= K, "budget"
    for terminal in T:
      model += s[terminal] == 0, f"protect_terminal_{node_to_position[terminal]}"

  # Path constraints:
  # x_ij + sum_{k in P} s_k >= 1, for every terminal pair (i, j) and every i-j path P
  #
  # Since i and j are in P, deleting an endpoint also satisfies the inequality.
  for source_node, target_node in terminal_pairs:
    source_position = node_to_position[source_node]
    target_position = node_to_position[target_node]

    for path_index, path_nodes in enumerate(terminal_paths[(source_node, target_node)]):
      model += (
        X(source_node, target_node) + pl.lpSum(s[node] for node in path_nodes) >= 1,
        f"path_{source_position}_{target_position}_{path_index}"
      )

      if progress_bar is not None:
        progress_bar.update(1)

  if progress_bar is not None:
    progress_bar.close()

  return model, X, s, terminal_paths


def get_pulp_solver(
  solver_name: str = "cbc",
  time_limit: float | None = None,
  msg: bool = False
):
  solver_name = solver_name.lower()

  if solver_name == "cbc":
    solver_kwargs = {"msg": msg}
    if time_limit is not None:
      solver_kwargs["timeLimit"] = time_limit
    return pl.PULP_CBC_CMD(**solver_kwargs)

  if solver_name == "highs":
    solver_kwargs = {"msg": msg}
    if time_limit is not None:
      solver_kwargs["timeLimit"] = time_limit
    return pl.HiGHS(**solver_kwargs)

  if solver_name == "highs_cmd":
    solver_kwargs = {"msg": msg}
    if time_limit is not None:
      solver_kwargs["timeLimit"] = time_limit
    return pl.HiGHS_CMD(**solver_kwargs)

  raise ValueError("solver_name must be one of: 'cbc', 'highs', 'highs_cmd'")


def solve_ecndp_path_pulp(
  G: nx.Graph,
  terminals,
  K: int,
  allow_terminal_deletion: bool = True,
  include_diagonal_in_objective: bool = False,
  solver_name: str = "cbc",
  time_limit: float | None = None,
  verbose: bool = False,
  show_progress: bool = False
):
  build_start_time = time.perf_counter()

  model, X, s, terminal_paths = build_ecndp_path_model_pulp(
    G=G,
    terminals=terminals,
    K=K,
    allow_terminal_deletion=allow_terminal_deletion,
    include_diagonal_in_objective=include_diagonal_in_objective,
    show_progress=show_progress
  )

  build_seconds = time.perf_counter() - build_start_time

  solver = get_pulp_solver(
    solver_name=solver_name,
    time_limit=time_limit,
    msg=verbose
  )

  solve_start_time = time.perf_counter()
  status_code = model.solve(solver)
  solve_seconds = time.perf_counter() - solve_start_time

  status_string = pl.LpStatus[status_code]

  if status_string not in {"Optimal"}:
    return {
      "status": status_string,
      "objective": None,
      "deleted_nodes": None,
      "x": None,
      "components_after_deletion": None,
      "build_seconds": build_seconds,
      "solve_seconds": solve_seconds,
      "model": model,
      "terminal_paths": terminal_paths
    }

  V = _ordered_nodes(G.nodes())
  T = _ordered_nodes(set(terminals))

  deleted_nodes = [
    node for node in V
    if pl.value(s[node]) is not None and pl.value(s[node]) >= 0.99
  ]

  if include_diagonal_in_objective:
    x_solution = {
      (T[left_position], T[right_position]): int(round(pl.value(X(T[left_position], T[right_position]))))
      for left_position in range(len(T))
      for right_position in range(left_position, len(T))
    }
  else:
    x_solution = {
      (T[left_position], T[right_position]): int(round(pl.value(X(T[left_position], T[right_position]))))
      for left_position in range(len(T))
      for right_position in range(left_position + 1, len(T))
    }

  residual_graph = G.copy()
  residual_graph.remove_nodes_from(deleted_nodes)
  components_after_deletion = list(nx.connected_components(residual_graph))

  return {
    "status": status_string,
    "objective": pl.value(model.objective),
    "deleted_nodes": deleted_nodes,
    "x": x_solution,
    "components_after_deletion": components_after_deletion,
    "build_seconds": build_seconds,
    "solve_seconds": solve_seconds,
    "model": model,
    "terminal_paths": terminal_paths
  }
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


def build_ecndp_model_pulp(
  G: nx.Graph,
  terminals,
  K: int,
  allow_terminal_deletion: bool = True,
  include_diagonal_in_objective: bool = False,
  model_name: str = "ECNDP",
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

  num_nodes = len(V)
  num_edges = G.number_of_edges()
  num_pairs = math.comb(num_nodes, 2) if num_nodes >= 2 else 0
  num_triples = math.comb(num_nodes, 3) if num_nodes >= 3 else 0

  total_progress_steps = (
    num_pairs +               # x variable creation
    num_edges +               # edge constraints
    2 * num_pairs +           # upper bound constraints
    3 * num_triples           # transitivity constraints
  )

  progress_bar = None
  if show_progress:
    print(f"Building model with {total_progress_steps} progress steps...")
    progress_bar = tqdm(
      total=total_progress_steps,
      desc="Building ECNDP model",
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

  # Connectivity variables: x_ij for unordered pairs i < j only
  x = {}
  for left_position in range(len(V)):
    left_node = V[left_position]
    for right_position in range(left_position + 1, len(V)):
      right_node = V[right_position]
      x[(left_node, right_node)] = pl.LpVariable(
        f"x_{left_position}_{right_position}",
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
      pl.lpSum(
        X(T[left_position], T[right_position])
        for left_position in range(len(T))
        for right_position in range(left_position + 1, len(T))
      )
    )
  else:
    model += pl.lpSum(
      X(T[left_position], T[right_position])
      for left_position in range(len(T))
      for right_position in range(left_position + 1, len(T))
    )

  # Budget / terminal protection
  if allow_terminal_deletion:
    model += pl.lpSum(s[node] for node in V) <= K, "budget"
  else:
    model += pl.lpSum(s[node] for node in V if node not in T_set) <= K, "budget"
    for terminal in T:
      model += s[terminal] == 0, f"protect_terminal_{node_to_position[terminal]}"

  # Edge constraints: x_ij >= 1 - s_i - s_j for every edge {i, j} in E
  for node_i, node_j in G.edges():
    if node_i == node_j:
      continue

    left_position = min(node_to_position[node_i], node_to_position[node_j])
    right_position = max(node_to_position[node_i], node_to_position[node_j])

    model += (
      X(node_i, node_j) >= 1 - s[node_i] - s[node_j],
      f"edge_lb_{left_position}_{right_position}"
    )

    if progress_bar is not None:
      progress_bar.update(1)

  # Upper bounds: x_ij <= 1 - s_i and x_ij <= 1 - s_j for all i < j
  for left_position in range(len(V)):
    left_node = V[left_position]
    for right_position in range(left_position + 1, len(V)):
      right_node = V[right_position]

      model += (
        X(left_node, right_node) <= 1 - s[left_node],
        f"upper_i_{left_position}_{right_position}"
      )
      model += (
        X(left_node, right_node) <= 1 - s[right_node],
        f"upper_j_{left_position}_{right_position}"
      )

      if progress_bar is not None:
        progress_bar.update(2)

  # Transitivity constraints
  for first_position in range(len(V)):
    for second_position in range(first_position + 1, len(V)):
      for third_position in range(second_position + 1, len(V)):
        first_node = V[first_position]
        second_node = V[second_position]
        third_node = V[third_position]

        model += (
          X(first_node, third_node) >=
          X(first_node, second_node) + X(second_node, third_node) - 1,
          f"transitivity_1_{first_position}_{second_position}_{third_position}"
        )
        model += (
          X(first_node, second_node) >=
          X(first_node, third_node) + X(second_node, third_node) - 1,
          f"transitivity_2_{first_position}_{second_position}_{third_position}"
        )
        model += (
          X(second_node, third_node) >=
          X(first_node, second_node) + X(first_node, third_node) - 1,
          f"transitivity_3_{first_position}_{second_position}_{third_position}"
        )

        if progress_bar is not None:
          progress_bar.update(3)

  if progress_bar is not None:
    progress_bar.close()

  return model, X, s


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


def solve_ecndp_pulp(
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

  model, X, s = build_ecndp_model_pulp(
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

  if status_string not in {"Optimal", "Feasible"}:
    return {
      "status": status_string,
      "objective": None,
      "deleted_nodes": None,
      "x": None,
      "components_after_deletion": None,
      "build_seconds": build_seconds,
      "solve_seconds": solve_seconds,
      "model": model
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
    "model": model
  }
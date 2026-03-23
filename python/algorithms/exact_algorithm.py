import networkx as nx
import pulp as pl


def build_ecndp_model_pulp(
  G: nx.Graph,
  terminals,
  K: int,
  allow_terminal_deletion: bool = True,
  include_diagonal_in_objective: bool = False,
  model_name: str = "ECNDP"
):
  V = sorted(G.nodes())
  T = sorted(terminals)
  T_set = set(T)
  V_set = set(V)

  if not T_set.issubset(V_set):
    raise ValueError("All terminals must belong to the graph.")
  if K < 0:
    raise ValueError("K must be nonnegative.")

  m = pl.LpProblem(model_name, pl.LpMinimize)

  # Deletion variables
  v = {
    i: pl.LpVariable(f"v_{i}", lowBound=0, upBound=1, cat=pl.LpBinary)
    for i in V
  }

  # Connectivity variables for unordered pairs only: i <= j
  u = {}
  for a in range(len(V)):
    i = V[a]
    u[(i, i)] = pl.LpVariable(f"u_{i}_{i}", lowBound=0, upBound=1, cat=pl.LpBinary)
    for b in range(a + 1, len(V)):
      j = V[b]
      u[(i, j)] = pl.LpVariable(f"u_{i}_{j}", lowBound=0, upBound=1, cat=pl.LpBinary)

  def U(i, j):
    return u[(i, j)] if i <= j else u[(j, i)]

  # Objective
  if include_diagonal_in_objective:
    m += (
      pl.lpSum(U(i, i) for i in T) +
      pl.lpSum(
        U(T[a], T[b])
        for a in range(len(T))
        for b in range(a + 1, len(T))
      )
    )
  else:
    m += pl.lpSum(
      U(T[a], T[b])
      for a in range(len(T))
      for b in range(a + 1, len(T))
    )

  # Budget / terminal protection
  if allow_terminal_deletion:
    m += pl.lpSum(v[i] for i in V) <= K, "budget"
  else:
    m += pl.lpSum(v[i] for i in V if i not in T_set) <= K, "budget"
    for t in T:
      m += v[t] == 0, f"protect_terminal_{t}"

  # Diagonal constraints: u_ii = 1 - v_i
  for i in V:
    m += U(i, i) + v[i] == 1, f"diag_{i}"

  # Edge constraints
  for i, j in G.edges():
    name = f"edge_{min(i, j)}_{max(i, j)}"
    m += U(i, j) + v[i] + v[j] >= 1, name

  # Triangle constraints for unordered triples only
  for a in range(len(V)):
    for b in range(a + 1, len(V)):
      for c in range(b + 1, len(V)):
        i = V[a]
        j = V[b]
        k = V[c]

        m += U(i, j) + U(j, k) - U(i, k) <= 1, f"tri1_{i}_{j}_{k}"
        m += U(i, j) - U(j, k) + U(i, k) <= 1, f"tri2_{i}_{j}_{k}"
        m += -U(i, j) + U(j, k) + U(i, k) <= 1, f"tri3_{i}_{j}_{k}"

  return m, U, v


def get_pulp_solver(
  solver_name: str = "cbc",
  time_limit: float | None = None,
  msg: bool = False
):
  name = solver_name.lower()

  if name == "cbc":
    kwargs = {"msg": msg}
    if time_limit is not None:
      kwargs["timeLimit"] = time_limit
    return pl.PULP_CBC_CMD(**kwargs)

  if name == "highs":
    kwargs = {"msg": msg}
    if time_limit is not None:
      kwargs["timeLimit"] = time_limit
    return pl.HiGHS(**kwargs)

  if name == "highs_cmd":
    kwargs = {"msg": msg}
    if time_limit is not None:
      kwargs["timeLimit"] = time_limit
    return pl.HiGHS_CMD(**kwargs)

  raise ValueError("solver_name must be one of: 'cbc', 'highs', 'highs_cmd'")


def solve_ecndp_pulp(
  G: nx.Graph,
  terminals,
  K: int,
  allow_terminal_deletion: bool = True,
  include_diagonal_in_objective: bool = False,
  solver_name: str = "cbc",
  time_limit: float | None = None,
  verbose: bool = False
):
  m, U, v = build_ecndp_model_pulp(
    G=G,
    terminals=terminals,
    K=K,
    allow_terminal_deletion=allow_terminal_deletion,
    include_diagonal_in_objective=include_diagonal_in_objective
  )

  solver = get_pulp_solver(
    solver_name=solver_name,
    time_limit=time_limit,
    msg=verbose
  )

  status_code = m.solve(solver)
  status_str = pl.LpStatus[status_code]

  if status_str not in {"Optimal", "Feasible"}:
    return {
      "status": status_str,
      "objective": None,
      "deleted_nodes": None,
      "u": None,
      "components_after_deletion": None,
      "model": m
    }

  V = sorted(G.nodes())
  T = sorted(terminals)

  deleted_nodes = [
    i for i in V
    if pl.value(v[i]) is not None and pl.value(v[i]) >= 0.99
  ]

  if include_diagonal_in_objective:
    u_sol = {
      (T[a], T[b]): int(round(pl.value(U(T[a], T[b]))))
      for a in range(len(T))
      for b in range(a, len(T))
    }
  else:
    u_sol = {
      (T[a], T[b]): int(round(pl.value(U(T[a], T[b]))))
      for a in range(len(T))
      for b in range(a + 1, len(T))
    }

  H = G.copy()
  H.remove_nodes_from(deleted_nodes)
  components = list(nx.connected_components(H))

  return {
    "status": status_str,
    "objective": pl.value(m.objective),
    "deleted_nodes": deleted_nodes,
    "u": u_sol,
    "components_after_deletion": components,
    "model": m
  }
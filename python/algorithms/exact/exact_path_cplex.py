import cplex
from cplex.callbacks import LazyConstraintCallback
import networkx as nx
from itertools import combinations
import random


def normalize_pair(i, j):
  return (i, j) if i < j else (j, i)


def path_signature(path):
  """
  The paper says: 'We denote a path P as the set of nodes in that path.'
  So we store a path by its node set, not by edge order.
  """
  return frozenset(path)


class PathSeparationCallback(LazyConstraintCallback):
  """
  Lazy callback implementing equation (12):

    x_ij + sum_{k in P} s_k >= 1

  for terminal pairs (i, j) and paths P that were not yet considered.
  """

  def __call__(self):
    # Current candidate solution values
    s_val = {
      i: self.get_values(self.s_idx[i])
      for i in self.nodes
    }
    x_val = {
      pair: self.get_values(self.x_idx[pair])
      for pair in self.term_pairs
    }

    # Build residual graph after removing selected nodes.
    # case = 1: terminals may be removed
    # case = 2: terminals are protected (and also forced to s_t = 0 in model)
    residual = self.G.copy()

    if self.case == 1:
      removed_nodes = [i for i in self.nodes if s_val[i] > 0.5]
    elif self.case == 2:
      removed_nodes = [
        i for i in self.nodes
        if s_val[i] > 0.5 and i not in self.terminals
      ]
    else:
      raise ValueError("case must be 1 or 2")

    residual.remove_nodes_from(removed_nodes)

    # Look for one violated path that has not been considered before.
    for pair in self.term_pairs:
      i, j = pair

      # If x_ij = 1, the pair is already paid for in the objective.
      if x_val[pair] > 0.5:
        continue

      # If one endpoint is not present, there is no surviving path.
      if i not in residual or j not in residual:
        continue

      try:
        # Search simple paths from shortest upward.
        for path in nx.shortest_simple_paths(residual, i, j):
          sig = path_signature(path)

          if sig in self.known_paths[pair]:
            continue

          # Mark this path as considered.
          self.known_paths[pair].add(sig)

          # Equation (12), literally as written:
          # x_ij + sum_{k in P} s_k >= 1
          ind = [self.x_idx[pair]] + [self.s_idx[k] for k in path]
          val = [1.0] + [1.0] * len(path)

          self.add(
            constraint=cplex.SparsePair(ind=ind, val=val),
            sense="G",
            rhs=1.0
          )

          # Add one violated path, then return control to CPLEX.
          return

      except (nx.NetworkXNoPath, nx.NodeNotFound):
        continue


def add_path_constraint(cpx, s_idx, x_idx, pair, path):
  """
  Add one constraint of the form:

    x_ij + sum_{k in P} s_k >= 1
  """
  ind = [x_idx[pair]] + [s_idx[k] for k in path]
  val = [1.0] + [1.0] * len(path)

  cpx.linear_constraints.add(
    lin_expr=[cplex.SparsePair(ind=ind, val=val)],
    senses=["G"],
    rhs=[1.0]
  )


def build_model_exact_from_excerpt(G, terminals, K, case, initial_paths=None):
  """
  Build the model from the paper excerpt:

    min  sum x_ij
    s.t. sum s_i <= K
         x_ij + sum_{k in P} s_k >= 1   for selected initial paths only
         s_i in {0,1}
         x_ij in {0,1}

  Missing path constraints are added lazily by callback.

  Parameters
  ----------
  G : networkx.Graph
    Undirected graph.
  terminals : iterable
    Terminal set T.
  K : int
    Budget.
  case : int
    case = 1 -> terminals are deletable
    case = 2 -> terminals are protected
  initial_paths : dict or None
    Maps terminal pair (i, j) to a list of node-paths.
    If None, one shortest path per terminal pair is used.
  """
  if case not in [1, 2]:
    raise ValueError("case must be 1 or 2")

  cpx = cplex.Cplex()
  cpx.objective.set_sense(cpx.objective.sense.minimize)

  nodes = list(G.nodes())
  terminals = sorted(terminals)
  term_pairs = list(combinations(terminals, 2))

  # Create variables s_i for all i in V
  s_names = [f"s_{i}" for i in nodes]
  cpx.variables.add(
    names=s_names,
    obj=[0.0] * len(nodes),
    lb=[0.0] * len(nodes),
    ub=[1.0] * len(nodes),
    types=["B"] * len(nodes)
  )

  # Create variables x_ij for all i<j, i,j in T
  x_names = [f"x_{i}_{j}" for (i, j) in term_pairs]
  cpx.variables.add(
    names=x_names,
    obj=[1.0] * len(term_pairs),
    lb=[0.0] * len(term_pairs),
    ub=[1.0] * len(term_pairs),
    types=["B"] * len(term_pairs)
  )

  s_idx = {i: cpx.variables.get_indices(f"s_{i}") for i in nodes}
  x_idx = {(i, j): cpx.variables.get_indices(f"x_{i}_{j}") for (i, j) in term_pairs}

  # Budget constraint: sum s_i <= K
  cpx.linear_constraints.add(
    lin_expr=[cplex.SparsePair(
      ind=[s_idx[i] for i in nodes],
      val=[1.0] * len(nodes)
    )],
    senses=["L"],
    rhs=[float(K)]
  )

  # case = 2: terminals are protected, so force s_t = 0
  if case == 2:
    for t in terminals:
      cpx.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=[s_idx[t]], val=[1.0])],
        senses=["E"],
        rhs=[0.0]
      )

  # Track all paths already considered in the formulation
  known_paths = {pair: set() for pair in term_pairs}

  # Default initial path set: one shortest path per terminal pair
  if initial_paths is None:
    initial_paths = {}
    for pair in term_pairs:
      i, j = pair
      try:
        initial_paths[pair] = [nx.shortest_path(G, i, j)]
      except (nx.NetworkXNoPath, nx.NodeNotFound):
        initial_paths[pair] = []

  # Add initial path constraints
  for raw_pair, paths in initial_paths.items():
    pair = normalize_pair(*raw_pair)
    if pair not in known_paths:
      continue

    for path in paths:
      sig = path_signature(path)
      if sig in known_paths[pair]:
        continue

      known_paths[pair].add(sig)
      add_path_constraint(cpx, s_idx, x_idx, pair, path)

  # Register lazy callback
  cb = cpx.register_callback(PathSeparationCallback)
  cb.G = G
  cb.nodes = nodes
  cb.terminals = set(terminals)
  cb.term_pairs = term_pairs
  cb.s_idx = s_idx
  cb.x_idx = x_idx
  cb.known_paths = known_paths
  cb.case = case

  return cpx, s_idx, x_idx, known_paths


def solve_exact_from_excerpt(G, terminals, K, case, initial_paths=None):
  """
  Build and solve the model.
  """
  cpx, s_idx, x_idx, known_paths = build_model_exact_from_excerpt(
    G=G,
    terminals=terminals,
    K=K,
    case=case,
    initial_paths=initial_paths
  )

  cpx.solve()

  s_sol = {
    i: cpx.solution.get_values(idx)
    for i, idx in s_idx.items()
  }
  x_sol = {
    pair: cpx.solution.get_values(idx)
    for pair, idx in x_idx.items()
  }

  return cpx, s_sol, x_sol, known_paths


def assign_terminals(G, terminal_number, seed):
  nodes = list(G.nodes())
  random.seed(seed)

  terminals = random.sample(nodes, terminal_number)

  nx.set_node_attributes(G, False, "terminal")
  for node in terminals:
    G.nodes[node]["terminal"] = True

  terminal_nodes = [n for n, d in G.nodes(data=True) if d["terminal"]]
  return G, terminal_nodes


if __name__ == "__main__":
  NODES = 100
  graph_model = nx.erdos_renyi_graph(NODES, 0.046, seed=1)

  print("graph:", len(graph_model.nodes()), "nodes,", len(graph_model.edges()), "edges")

  terminal_count = 40
  G_assigned, terminals_assigned = assign_terminals(graph_model, terminal_count, 1)

  K = 20

  # Choose case:
  # 1 -> terminals deletable
  # 2 -> terminals protected
  case = 1

  # You can leave this as None.
  # Then the code uses one shortest path per terminal pair initially.
  initial_paths = None

  cpx, s_sol, x_sol, known_paths = solve_exact_from_excerpt(
    G=G_assigned,
    terminals=terminals_assigned,
    K=K,
    case=case,
    initial_paths=initial_paths
  )

  print("Status:", cpx.solution.get_status_string())
  print("Objective:", cpx.solution.get_objective_value())

  # print("\ns solution")
  # for i in sorted(s_sol):
  #   print(f"s[{i}] = {s_sol[i]}")

  # print("\nx solution")
  # for pair in sorted(x_sol):
  #   print(f"x{pair} = {x_sol[pair]}")

  # print("\nNumber of considered paths")
  # for pair in sorted(known_paths):
  #   print(f"{pair}: {len(known_paths[pair])}")
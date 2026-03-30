from algorithms.compute_PC import compute_pc
from algorithms.local_search import local_search_procedure
from utils.improvement_condition import improvement_condition
import networkx as nx
from tqdm import tqdm


def _compute_terminal_affinity_scores(
  G: nx.Graph,
  terminals: list[int]
) -> dict[int, float]:
  """
  For each non-terminal node u, compute:
    affinity(u) = d1(u) / d2(u)

  where:
    d1(u) = distance to closest terminal
    d2(u) = distance to second-closest terminal

  Smaller score => node is strongly associated with one terminal region.
  Score near 1 => node is similarly close to multiple terminals.

  If |T| < 2, all non-terminals get score 0.0.
  """
  T = set(terminals)
  affinity: dict[int, float] = {}

  if len(T) < 2:
    for u in G.nodes:
      if u not in T:
        affinity[u] = 0.0
    return affinity

  # shortest path distances from each terminal to all reachable nodes
  dist_from_terminal: dict[int, dict[int, int]] = {}
  for t in terminals:
    dist_from_terminal[t] = nx.single_source_shortest_path_length(G, t)

  for u in G.nodes:
    if u in T:
      continue

    dists = []
    for t in terminals:
      d = dist_from_terminal[t].get(u, float("inf"))
      dists.append(d)

    dists.sort()
    d1 = dists[0]
    d2 = dists[1]

    if d1 == float("inf") and d2 == float("inf"):
      affinity[u] = float("inf")
    elif d2 == 0:
      affinity[u] = 0.0
    elif d2 == float("inf"):
      # reachable from only one terminal => strongly terminal-specific
      affinity[u] = 0.0
    else:
      affinity[u] = d1 / d2

  return affinity


def greedy_terminal_affinity_cand(
  G: nx.Graph,
  terminals: list[int],
  k: int,
  case: int
) -> set[int]:
  """
  Case 1:
    Keep your original MIS-based construction.

  Case 2:
    Start from X = T and greedily augment the kept set until |X| = |V| - k.
    Primary criterion: minimize compute_pc(...)
    Secondary criterion: minimize terminal-affinity score.
  """
  T: set[int] = set(terminals)
  V: set[int] = set(G.nodes)

  if case == 1:
    MIS = set(nx.maximal_independent_set(G))

    while len(MIS) < len(V) - k:
      best_j = None
      best_pc = float("inf")

      for j in V - MIS:
        S_j = V - (MIS | {j})
        pc_j = compute_pc(G, S_j, terminal_nodes=terminals)

        if pc_j < best_pc:
          best_pc = pc_j
          best_j = j

      if best_j is None:
        break

      MIS.add(best_j)

    return V - MIS

  elif case == 2:
    # terminals are undeletable in this construction
    target_keep = len(V) - k

    if len(T) > target_keep:
      raise ValueError(
        "Infeasible instance for case=2: "
        "cannot keep all terminals with the given budget."
      )

    X = set(T)
    affinity = _compute_terminal_affinity_scores(G, terminals)

    while len(X) < target_keep:
      best_j = None
      best_pc = float("inf")
      best_aff = float("inf")

      for j in V - X:
        S_j = V - (X | {j})
        pc_j = compute_pc(G, S_j, terminal_nodes=terminals)

        # terminals are already in X for case 2, so j is non-terminal
        aff_j = affinity.get(j, float("inf"))

        # lexicographic minimization:
        #  1) smaller pairwise connectivity
        #  2) smaller affinity score
        if pc_j < best_pc:
          best_pc = pc_j
          best_aff = aff_j
          best_j = j
        elif pc_j == best_pc and aff_j < best_aff:
          best_aff = aff_j
          best_j = j

      if best_j is None:
        break

      X.add(best_j)

    return V - X

  else:
    raise ValueError("case must be either 1 or 2")


def extended_critical_node_terminal_affinity_candidate(
  G,
  terminals,
  k,
  case,
  maxIter,
  use_ls: bool = False
):
  """
  Multi-start wrapper matching the structure of your current code.
  """
  S = None
  best_S = None
  best_pc = float("inf")
  curr_S = None

  S = greedy_terminal_affinity_cand(
    G,
    terminals=terminals,
    k=k,
    case=case
  )

  curr_S = set(S)

  if use_ls:
    S_ls = local_search_procedure(
      G,
      S,
      terminals=terminals,
      case=case,
      improvement_condition=improvement_condition
    )
    curr_S = set(S_ls)

  best_pc = compute_pc(G, curr_S, terminal_nodes=terminals)

  return curr_S, best_pc
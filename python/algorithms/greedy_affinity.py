from algorithms.compute_PC import compute_pc
from algorithms.local_search import local_search_procedure
from utils.improvement_condition import improvement_condition
import networkx as nx
from tqdm import tqdm


def _terminal_affinity_vector(
  G: nx.Graph,
  u: int,
  terminals: list[int],
  dist_from_terminal: dict[int, dict[int, int]]
) -> dict[int, float]:
  """
  For a non-terminal node u, compute the affinity vector:
    affinity_i(u) = d(u, i) / min_{j in T \ {i}} d(u, j)

  Returns a dictionary:
    {terminal_i: affinity_i(u)}
  """
  affinity_u: dict[int, float] = {}

  for i in terminals:
    d_ui = dist_from_terminal[i].get(u, float("inf"))

    best_other = float("inf")
    for j in terminals:
      if j == i:
        continue
      d_uj = dist_from_terminal[j].get(u, float("inf"))
      if d_uj < best_other:
        best_other = d_uj

    if best_other == float("inf"):
      affinity_u[i] = float("inf")
    elif best_other == 0:
      affinity_u[i] = float("inf")
    else:
      affinity_u[i] = d_ui / best_other

  return affinity_u


def _compute_simple_affinity_scores(
  G: nx.Graph,
  terminals: list[int]
) -> dict[int, float]:
  """
  Compute the simple scalar score from the screenshot idea:

    score(u) = min_{i in T} affinity_i(u)

  Smaller score means u is strongly associated with one terminal
  and relatively far from the others.
  """
  T = set(terminals)
  scores: dict[int, float] = {}

  # shortest-path distances from each terminal
  dist_from_terminal: dict[int, dict[int, int]] = {}
  for t in terminals:
    dist_from_terminal[t] = nx.single_source_shortest_path_length(G, t)

  for u in G.nodes:
    if u in T:
      continue

    affinity_u = _terminal_affinity_vector(
      G,
      u,
      terminals,
      dist_from_terminal
    )

    if len(affinity_u) == 0:
      scores[u] = float("inf")
    else:
      scores[u] = min(affinity_u.values())

  return scores


def greedy_terminal_affinity_simple(
  G: nx.Graph,
  terminals: list[int],
  k: int,
  case: int
) -> set[int]:
  """
  Simple implementation exactly matching the screenshot idea.

  Case 1:
    Keep your original MIS-based construction.

  Case 2:
    Start from X = T
    Compute score(u) = min_i affinity_i(u)
    Greedily add the node with smallest score
    until |X| = |V| - k
    Return S = V - X
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
    target_keep = len(V) - k

    if len(T) > target_keep:
      raise ValueError(
        "Infeasible case=2 instance: number of terminals exceeds |V| - k."
      )

    X = set(T)
    affinity_scores = _compute_simple_affinity_scores(G, terminals)

    while len(X) < target_keep:
      best_j = None
      best_score = float("inf")

      for j in V - X:
        score_j = affinity_scores.get(j, float("inf"))

        if score_j < best_score:
          best_score = score_j
          best_j = j

      if best_j is None:
        break

      X.add(best_j)

    S = V - X
    return S

  else:
    raise ValueError("case must be either 1 or 2")


def extended_critical_node_terminal_affinity_simple(
  G,
  terminals,
  k,
  case,
  maxIter,
  use_ls: bool = False
):
  """
  Wrapper in the same style as your current code.
  """
  S = None
  best_S = None
  best_pc = float("inf")

  S = greedy_terminal_affinity_simple(
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
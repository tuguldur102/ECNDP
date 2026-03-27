from algorithms.compute_PC import compute_pc
from algorithms.local_search import local_search_procedure
from utils.improvement_condition import improvement_condition
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Set, Any

def compute_affinity_score(
  G, terminals
  ):

  V = set(G.nodes())
  T = set(terminals)
  U = set(V - T)

  # scores : dict[int, List[float]] = defaultdict(list)
  scores : dict[int, float] = {}

  # shortest path distances from each terminal
  # dist_from_terminal : dict[int, dict[int, int]] = {}
  # for t in terminals:
  #   dist_from_terminal[t] = nx.single_target_shortest_path_length(G, t)

  # For a non-terminal node u, compute the affinity score:
  # affinity_i(u) = d(u, i) / min_{j in T \ {i}} d(u, j)
  for u in U:

    terminal_scores_u = []
    for t in T:

      # d(u, i)
      d_u_i = nx.shortest_path_length(G=G, source=u, target=t)

      # min_{j in T \ {i}} d(u, j)
      best_u_j = float("inf")

      for j in T - {t}:
        # j in T \ {i} d(u, j)
        d_u_j = nx.shortest_path_length(G=G, source=u, target=j)

        # argmin
        if d_u_j < best_u_j:
          best_u_j = d_u_j

      # append terminal scores
      aff_i_u = d_u_i / best_u_j
      terminal_scores_u.append(aff_i_u)

    scores[u] = min(terminal_scores_u)

  return scores 

def greedy_terminal_affinity(
  G: nx.Graph,
  terminals: list[int],
  k: int,
  case: int
) -> set[int]:
  T: set[int] = set(terminals)
  V: set[int] = set(G.nodes())

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

      MIS.add(best_j)

    return V - MIS

  elif case == 2:

    X = set(T)
    affinity_scores = compute_affinity_score(G, terminals)

    while len(X) != len(V) - k:
      best_j = None
      best_score = float("inf")

      for j in V - X:
        score_j = affinity_scores[j]

        if score_j < best_score:
          best_score = score_j
          best_j = j

      if best_j is None:
        break

      X.add(best_j)

    S = V - X

    return S

def extended_critical_node_terminal_affinity(
  G,
  terminals,
  k,
  case,
  maxIter = None,
  use_ls: bool = False
):
  """
  Wrapper in the same style as your current code.
  """
  S = None
  best_S = None
  best_pc = float("inf")

  S = greedy_terminal_affinity(
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
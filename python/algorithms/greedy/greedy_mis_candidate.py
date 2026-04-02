from algorithms.compute_PC import compute_pc
from algorithms.local_search.local_search import local_search_procedure
from utils.improvement_condition import improvement_condition
import networkx as nx
from tqdm import tqdm

def greedy_mis_cand(
  G: nx.Graph, terminals: list[int], 
  k: int, case: 1 | 2) -> tuple[set, list[int]]:
  
  # If graph is too big, use sparse representation
  # for better memory usage

  T : set = set(terminals)
  V : set = set(G.nodes())

  S : set = set()
  # pc_deltas : list[int] = []

  if case == 1:
    MIS = set(nx.maximal_independent_set(G))

    # print(f"mis: {MIS}")
  
    while len(MIS) < len(V) - k:

      best_j = None
      best_pc = float("inf")

      for j in V - MIS:
        
        S_j = V - (MIS | {j})
        pc_j = compute_pc(G, S_j, terminal_nodes=terminals)

        # argmin
        if pc_j < best_pc:
          best_pc = pc_j
          best_j = j

      MIS.add(best_j)
      # pc_deltas.append(best_pc)

    S = V - MIS

    return S

  # elif case == 2:
  #   U = V - T
  #   H = G.copy()
  #   H.remove_nodes_from(T)

  #   K = set(nx.maximal_independent_set(H))
  #   target = len(U) - k

  #   while len(K) < target:
  #     best_j = None
  #     best_pc = float("inf")

  #     for j in U - K:
  #       S_j = U - (K | {j})
  #       pc_j = compute_pc(G, S_j, terminal_nodes=terminals)

  #       if pc_j < best_pc:
  #         best_pc = pc_j
  #         best_j = j

  #     if best_j is None:
  #       break

  #     K.add(best_j)

  #   # print(f"K : {K}")
  #   S = U - K

  #   return S
  elif case == 2:

    # deletable nonterminals
    MIS = set(nx.maximal_independent_set(G))

    MIS = MIS | T

    while len(MIS) < len(V) - k:

      best_j = None
      best_pc = float("inf")

      for j in V - MIS:
        
        S_j = V - (MIS | {j})
        pc_j = compute_pc(G, S_j, terminal_nodes=terminals)

        # argmin
        if pc_j < best_pc:
          best_pc = pc_j
          best_j = j

      MIS.add(best_j)
      # pc_deltas.append(best_pc)

    S = V - MIS
      # pc_deltas.append(best_pc)

    return S

def extended_critical_node_mis_candidate(
    G, terminals, k, case, maxIter, use_ls: False):
  
  S = None
  best_S = None
  best_pc = float("inf")

  for _ in range(maxIter):

    curr_S = None

    S = greedy_mis_cand(
      G, terminals=terminals, k=k, case=case)

    curr_S = set(S)

    if use_ls:
      S_ls = local_search_procedure(
        G, S, 
        terminals=terminals, 
        case=case,
        improvement_condition=improvement_condition,
        max_iter=2)
      
      curr_S = set(S_ls)
    
    curr_pc = compute_pc(G, curr_S, terminal_nodes=terminals)

    if curr_pc < best_pc:
      best_pc = curr_pc
      best_S = set(curr_S)
      # print(f"Refined PC: {best_pc}\n")

  return best_S, best_pc
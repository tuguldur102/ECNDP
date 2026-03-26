from algorithms.compute_PC import compute_pc
from algorithms.local_search import local_search_procedure
from utils.improvement_condition import improvement_condition
import networkx as nx
from tqdm import tqdm

def greedy_mis(
  G: nx.Graph, terminals: list[int], 
  k: int, case: 1 | 2) -> tuple[set, list[int]]:
  
  # If graph is too big, use sparse representation
  # for better memory usage

  T : set = set(terminals)
  V : set = set(G.nodes)

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
    U = set(V - T)

    # H = G[V \ T]
    H = G.copy()
    H.remove_nodes_from(T)

    MIS = set(nx.maximal_independent_set(H))

    # number of nonterminal nodes that can be kept
    r = len(V) - len(T) - k

    # prune if too large
    while len(MIS) > r:

      best_remove = None
      best_pc = float("inf")

      for j in MIS:
        # remove j from the current MIS
        kept = T | (MIS - {j})
        S_j = V - kept
        pc_j = compute_pc(G=G, S=S_j, terminal_nodes=terminals)

        if pc_j < best_pc:
          best_pc = pc_j
          best_remove = j

      MIS.remove(best_remove)
      # pc_deltas.append(best_pc)

    # extend if too small
    while len(MIS) < r:
      
      best_add = None
      best_pc = float("inf")

      for j in U - MIS:
        # add j from remaining nonterminals
        kept = T | (MIS | {j})
        S_j = V - kept
        pc_j = compute_pc(G=G, S=S_j, terminal_nodes=terminals)

        if pc_j < best_pc:
          best_pc = pc_j
          best_add = j

      MIS.add(best_add)
      # pc_deltas.append(best_pc)

    S = U - MIS

    return S

def extended_critical_node_mis(
    G, terminals, k, case, maxIter, use_ls: False):
  
  S = None
  best_S = None
  best_pc = float("inf")

  for _ in tqdm(range(maxIter), desc="greedy MIS", total=maxIter):

    curr_S = None

    S = greedy_mis(
      G, terminals=terminals, k=k, case=case)

    curr_S = set(S)

    if use_ls:
      S_ls = local_search_procedure(
        G, S, 
        terminals=terminals, 
        case=case,
        improvement_condition=improvement_condition)
      
      curr_S = set(S_ls)
    
    curr_pc = compute_pc(G, curr_S, terminal_nodes=terminals)

    if curr_pc < best_pc:
      best_pc = curr_pc
      best_S = set(curr_S)
      # print(f"Refined PC: {best_pc}\n")

  return best_S, best_pc
import networkx as nx
from algorithms.compute_PC import compute_pc
from algorithms.local_search import local_search_procedure
from utils.improvement_condition import improvement_condition
from tqdm import tqdm

def greedy_empty_set(
    G: nx.Graph, terminals: list[int], 
    k: int, case: 1 | 2) -> tuple[set, list[int]]:
  
  # If graph is too big, use sparse representation
  # for better memory usage

  T : set = set(terminals)
  V : set = set(G.nodes)

  S : set = set()
  # pc_deltas : list[int] = []

  K : set = set()
  
  if case == 1:
    K = set()

  elif case == 2:
    K = set(T)

  while len(K) != len(V) - k:

    best_j = None
    best_pc = float("inf")

    for j in V - K:
      
      S_j = V - (K | {j})
      pc_j = compute_pc(G, S_j, terminal_nodes=terminals)

      # print(pc_j)
      
      # argmin
      if pc_j < best_pc:
        best_pc = pc_j
        best_j = j

    K.add(best_j)
    # pc_deltas.append(best_pc)

  S = V - K

  return S

def extended_critical_node_empty_set(
    G, terminals, k, case, maxIter, use_ls: False):
  
  S = None
  best_S = None
  best_pc = float("inf")
  
  # for _ in tqdm(range(maxIter), desc="greedy ES", total=maxIter):

  curr_S = None

  S = greedy_empty_set(G, terminals=terminals, k=k, case=case)

  curr_S = set(S)

  if use_ls:
    S_ls = local_search_procedure(
      G, S, 
      terminals=terminals, 
      case=case, 
      improvement_condition=improvement_condition)
    
    curr_S = set(S_ls)
  
  curr_pc = compute_pc(G, curr_S, terminal_nodes=terminals)

    # if curr_pc < best_pc:
    #   best_pc = curr_pc
    #   best_S = set(curr_S)
    #   print(f"Refined PC: {best_pc}\n")

  return curr_S, curr_pc
  # return best_S, best_pc
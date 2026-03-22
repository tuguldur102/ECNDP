import networkx as nx
from compute_PC import compute_pc
from typing import Optional, Callable, List, Set, Literal

def local_search(
    G, S, terminals, 
    case: Literal[1, 2], 
    improvement_condition: Callable[[nx.Graph, Set, int, List], bool], 
    ):

  V = set(G.nodes())
  T = set(terminals)

  # Kept set
  # if the algorithm is greedy MIS
  # K is MIS
  K = set(V - S)

  curr_pc = obj(G, K, terminals)
  local_improvement = True

  if case == 1:
    current_nodes = list(V)
  elif case == 2: 
    # terminals cannot be removed
    current_nodes = list(V - T)
  
  while local_improvement:
    local_improvement = False
    best_K = set(K)

    for i in current_nodes:
      for j in current_nodes:

        if i in K and j not in K:
          K.remove(i)
          K.add(j)

          if obj(G, K, terminals) < obj(G, best_K, terminals):
            best_K = set(K)
          else:
            K.remove(j)
            K.add(i)
    
    if improvement_condition(G, best_K, curr_pc, terminals):
      K = set(best_K)
      local_improvement = True

    return V - K


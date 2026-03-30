import networkx as nx
from algorithms.compute_PC import compute_pc

def greedy_mis_algorithm(
    G: nx.Graph, terminals: list[int], 
    k: int, case: int, mis_trails: int = 30) -> tuple[set, list[int]]:
  
  # If graph is too big, sparse representation
  # for better memory usage
  
  S : set = set()
  V : set = set(G.nodes)
  T : set = set(terminals)

  # Loop for selecting best warm-up MIS
  # which has resulted in lower total PC
  best_pc = float("inf")
  MIS : set = set()

  for _ in range(mis_trails):
    mis = set(nx.maximal_independent_set(G))

    # If case 2, ensure terminal nodes are not deleted
    # keep terminal nodes in MIS
    if case == 2:
      mis = mis | T  
    
    # deletions correspond to V \ MIS
    S0 = V - mis

    # feasibility for case 2, do not delete terminal nodes
    if case == 2 and (S0 & T):
      continue

    curr_pc = compute_pc(G=G, S=S0, terminal_nodes=terminals)

    if curr_pc < best_pc:
      best_pc = curr_pc
      MIS = mis

  pc_deltas : list[int] = []

  while len(MIS) != len(V) - k:

    best_node = None
    best_deg = -1
    best_pc = float("inf")

    for node in V - MIS:

      if case == 1:
        # case 1: S in V
        S_j = V - (MIS | {node})
        pc_Sj = compute_pc(G=G, S=S_j, terminal_nodes=terminals)
        
      if case == 2:
        # case 2: S in V \ T
        if node not in T:
          S_j = V - (MIS | {node})
          pc_Sj = compute_pc(G=G, S=S_j, terminal_nodes=terminals)

      # tie-breaker - if two nodes give the same best_pc
      # get the one with higher degree
      deg = G.degree(node)

      # argmin
      if pc_Sj < best_pc or (pc_Sj == best_pc and deg > best_deg):
        best_pc = pc_Sj
        best_node = node
        best_deg = deg
        # print(f"best node: {best_node} and best gain: {best_pc}")
    
    if best_node is None:
      # No feasible solution
      break

    MIS.add(best_node)
    pc_deltas.append(best_pc)
  
  S = V - MIS

  return S, pc_deltas
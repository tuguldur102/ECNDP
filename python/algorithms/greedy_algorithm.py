import networkx as nx
from compute_PC import compute_pc

def greedy_algorithm(
    G: nx.Graph, terminals: list[int], 
    k: int, case: 1 | 2) -> tuple[set, list[int]]:
  
  # If graph is too big, use sparse representation
  # for better memory usage

  T : set = set(terminals)
  V : set = set(G.nodes)

  S : set = set()
  pc_deltas : list[int] = []

  for _ in range(k):

    best_node = None
    best_deg = -1
    best_pc = float("inf")

    # print(f"\nK: iteration\n")
    for node in V - S:
      
      pc_Sj = None

      if case == 1:
        # case 1: S in V
        S_j = S | {node}
        pc_Sj = compute_pc(G=G, S=S_j, terminal_nodes=terminals)

        print(f"node: {node} and pc Sj : {pc_Sj}")

      if case == 2:
        # case 2: S in V \ T
        if node not in T:
          S_j = S | {node}
          pc_Sj = compute_pc(G=G, S=S_j, terminal_nodes=terminals)

      # tie-breaker - if two nodes give the same best_pc
      # get the one with higher degree
      # deg = G.degree(node)

      # argmin
      if pc_Sj is not None and pc_Sj < best_pc:
        best_pc = pc_Sj
        best_node = node
        # best_deg = deg
        # print(f"best node: {best_node} and best gain: {best_pc}")

    if best_node is None:
      # No feasible solution
      break 
    
    S.add(best_node)
    pc_deltas.append(best_pc)

  return S, pc_deltas




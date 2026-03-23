import networkx as nx
from utils.helpers_for_PC import bfs, connected_components, comb_of_two

def compute_pc(
    G: nx.Graph, S: set, 
    terminal_nodes: list[int]) -> int:

  # Remaining vertices after deletion
  # V_remaining = set(G.nodes) - S
  # num_rem = len(V_remaining)

  T = set(terminal_nodes)

  G_copy = G.copy()
  G_copy.remove_nodes_from(S)

  # 1. Get connected components
  components = connected_components(G_copy)

  # 2. Exclude non-terminal nodes from the components
  excluded_components = []

  for comp in components:
    excluded_comp = []
    for v in comp:
      if v in T:
        excluded_comp.append(v)

    excluded_components.append(excluded_comp) 

  # print(f"Excluded components: {excluded_components}")
  
  # 3. Compute pairwise connectivity across terminal nodes over components
  total_pc = comb_of_two(excluded_components)

  ## More simpler usage:
  # total_pc = 0
  # for comp in connected_components(G_copy):
  #   comp_count = sum(1 for v in comp if v in T)
  #   total_pc += comp_count * (comp_count - 1) // 2

  return total_pc

def compute_pc_no_S(
    G: nx.Graph, 
    terminal_nodes: list[int]) -> int:

  # Remaining vertices after deletion
  # V_remaining = set(G.nodes) - S
  # num_rem = len(V_remaining)

  T = set(terminal_nodes)

  G_copy = G.copy()

  # 1. Get connected components
  components = connected_components(G_copy)

  # 2. Exclude non-terminal nodes from the components
  excluded_components = []

  for comp in components:
    excluded_comp = []
    for v in comp:
      if v in T:
        excluded_comp.append(v)

    excluded_components.append(excluded_comp) 

  # print(f"Excluded components: {excluded_components}")
  
  # 3. Compute pairwise connectivity across terminal nodes over components
  total_pc = comb_of_two(excluded_components)

  ## More simpler usage:
  # total_pc = 0
  # for comp in connected_components(G_copy):
  #   comp_count = sum(1 for v in comp if v in T)
  #   total_pc += comp_count * (comp_count - 1) // 2

  return total_pc

def obj(G, K, terminals):
  V = set(G.nodes())
  S = set(V - K)
  return compute_pc(G, S, terminal_nodes=terminals)



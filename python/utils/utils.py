import random
import networkx as nx
from algorithms.compute_PC import compute_pc
import time

def create_custom_graph_with_2_comps(terminal_nodes: list[int]) -> nx.Graph:

  G = nx.Graph()

  C1 = [(1, 2), (2, 3), (3, 4)]
  C2 = [(5, 6), (6, 7)]

  G.add_edges_from(C1)
  G.add_edges_from(C2)

  # Add terminal attribute to nodes
  for v in G.nodes:
    if v in terminal_nodes:
      G.nodes[v]["terminal"] = True
    else:
      G.nodes[v]["terminal"] = False
  
  return G

def create_custom_graph_extreme(terminal_nodes, edge_arrangement):
  
  G = nx.Graph()

  if edge_arrangement == 1:
    C = [(1, 2), (1, 4), (2, 5), (2, 3), (3, 6), (5, 6), (4, 5)]
  if edge_arrangement == 2:
    C = [(1, 2), (1, 3), (2, 5), (2, 4), (3, 5), (5, 6), (4, 6)]
  if edge_arrangement == 3:
    C = [(1, 2), (1, 3), (2, 5), (2, 4), (3, 4), (4, 6), (5, 6)]

  G.add_edges_from(C)

  # Add terminal attribute to nodes
  for v in G.nodes:
    if v in terminal_nodes:
      G.nodes[v]["terminal"] = True
    else:
      G.nodes[v]["terminal"] = False
  
  return G

def assign_terminals(G, terminal_number, seed):

  nodes = list(G.nodes())
  budget = int(terminal_number)
  print(f"budget {budget}")

  random.seed(seed)

  terminals = random.sample(nodes, budget)

  nx.set_node_attributes(G, False, "terminal")

  for node in terminals:
    G.nodes[node]["terminal"] = True

  terminal_nodes = [n for n, d in G.nodes(data=True) if d["terminal"]]
  print(len(terminal_nodes))

  return G, terminal_nodes

def assign_terminals_randomly(G, budget_percent, seed):

  nodes = list(G.nodes())
  budget = int(len(nodes) * budget_percent)
  print(f"budget {budget}")

  random.seed(seed)


  terminals = random.sample(nodes, budget)

  nx.set_node_attributes(G, False, "terminal")

  for node in terminals:
    G.nodes[node]["terminal"] = True

  terminal_nodes = [n for n, d in G.nodes(data=True) if d["terminal"]]
  print(len(terminal_nodes))

  return G, terminal_nodes

def create_mammals_graph():
  G = nx.Graph()

  edges = [
    (17,25),(17,22),(17,26),
    (18,4),
    (27,28),(27,26),(27,29),
    (30,13),(30,31),(30,32),(30,33),(30,11),(30,1),
    (34,35),(34,36),(34,24),(34,37),(34,38),(34,39),
    (40,22),
    (33,41),(33,0),(33,10),
    (42,23),(42,9),(42,43),(42,20),
    (28,41),(28,26),(28,4),(28,29),
    (4,41),
    (19,35),(19,2),
    (37,35),(37,36),(37,24),(37,38),(37,39),
    (9,23),(9,0),(9,20),(9,43),
    (0,21),(0,10),
    (21,23),(21,15),(21,10),
    (39,35),(39,7),(39,36),(39,24),(39,2),(39,38),
    (14,7),(14,12),(14,24),(14,2),(14,8),
    (31,13),
    (7,12),(7,2),(7,8),
    (32,11),
    (26,44),(26,29),(26,22),
    (10,41),
    (8,12),(8,24),(8,2),(8,45),
    (46,5),(46,25),(46,47),
    (41,44),(41,5),
    (44,22),(44,5),
    (48,15),
    (38,35),(38,36),(38,24),
    (25,5),(25,47),
    (49,1),
    (35,36),(35,24),(35,2),
    (13,1),
    (45,6),(45,12),
    (36,24),
    (5,47),
    (12,6),(12,50),(12,2),(12,3),
    (3,50),
    (15,16),
    (6,16),
    (20,23),(20,43),
    (24,2),
    (23,43)
  ]

  G.add_edges_from(edges)

  return G

def draw_graph(G, terminals):
  import matplotlib.pyplot as plt

  color_map = ["red" if G.nodes[n]["terminal"] else "lightblue" for n in G.nodes()]

  nx.draw(G, node_color=color_map, with_labels=True)
  plt.show()

def solve(G, k, terminals, maxIter, case, algorithm, use_tqdm, use_ls, ls_iter):

  # curr_pc = compute_pc(G, set(), terminal_nodes=terminals)

  # algorithm
  maxIter = maxIter

  start = time.perf_counter()
  S, best_pc = algorithm(
    G, terminals, k, case, maxIter, 
    use_tqdm=use_tqdm, use_ls=use_ls, max_iter=ls_iter)
  
  end = time.perf_counter()
  total_time = end - start

  # print(f"Current PC: {curr_pc}")
  # print(f"After PC: {best_pc}")
  # print(f"Running time: {total_time}")

  return S, best_pc, total_time

def solve_exact(G, k, terminals, case, algorithm, show_progress = False):

    if case == 1:
      allow_terminal_deletion = True
    elif case == 2:
      allow_terminal_deletion = False

    curr_pc = compute_pc(G, set(), terminal_nodes=terminals)
    start = time.perf_counter()

    sol1 = algorithm(
      G,
      terminals=terminals,
      K=k,
      allow_terminal_deletion=allow_terminal_deletion,
      include_diagonal_in_objective=False,
      solver_name="cbc",
      show_progress=show_progress
    )
    end = time.perf_counter()

    print(f"Case {case}")
    print("Current PC:", curr_pc)
    print("Status:", sol1["status"])
    print("Objective:", sol1["objective"])
    print("Deleted:", sol1["deleted_nodes"])
    print("Terminal u-values:", sol1["x"])
    print("Components:", sol1["components_after_deletion"])
    dt = end - start

    print(f"case {case}: time: {dt:.6f}")

    return sol1["deleted_nodes"], sol1["objective"], dt
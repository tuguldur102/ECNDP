import networkx as nx
import matplotlib.pyplot as plt
from algorithms.compute_PC import compute_pc
from algorithms.exact_algorithm_new import solve_ecndp_pulp
from algorithms.greedy_empty_set import extended_critical_node_empty_set
from algorithms.greedy_mis_candidate import extended_critical_node_mis_candidate
from utils.utils import assign_terminals_randomly, assign_terminals, create_custom_graph_extreme, create_custom_graph_with_2_comps, create_mammals_graph, solve, solve_exact
from tqdm import tqdm
import time
import pandas as pd

# Constants
SEED = 1
NODES = 25
TERMINAL_BUDGETS = {
  15: [10, 8, 5], 
  25: [15, 10, 7], 
  38: [20, 15, 10],
}

NON_TERMINAL_BUDGETS = {
  15: [20, 15, 10],
  25: [15, 10, 7], 
  38: [10, 8, 5], 
}

K_BUDGETS = []

CASE_1 = 1
CASE_2 = 2

CASES = [2, 1]
# Random graph creations

graph_models = {
  'ER': nx.erdos_renyi_graph(NODES, 0.073, seed=SEED),
  'BA': nx.barabasi_albert_graph(NODES, 2,seed=SEED),
  'SW': nx.watts_strogatz_graph(NODES, 4, 0.3, seed=SEED)
}

for name in ["ER", "BA", "SW"]:
  print(len(graph_models[name].nodes()), len(graph_models[name].edges()))


records = []
for case_number in tqdm(CASES, desc="Cases", total=2):

  if case_number == 1:
    ter_budget_iter = TERMINAL_BUDGETS
  if case_number == 2:
    ter_budget_iter = NON_TERMINAL_BUDGETS

  for terminal, k_budgets in tqdm(
    ter_budget_iter.items(), desc="Terminal loop", total=len(ter_budget_iter)):

    for k_budget in tqdm(
      k_budgets, desc="deletion set loop", total=len(k_budgets)):
    

      Graph_sample = create_mammals_graph()

      G_assigned, terminals_assigned = assign_terminals(
        Graph_sample, terminal, SEED)

      nodes = list(G_assigned.nodes())

      len_non_terminals = int(len(nodes) - len(terminals_assigned))

      K = int(k_budget)

      print(f"Curr pc: {compute_pc(G_assigned, set(), terminals_assigned)} \n")
      print("~~~~~~~~~~~~~~~~~~~~~~~~")
      print(f"nodes: {len(nodes)} edges {len(G_assigned.edges())} and terminal: {len(terminals_assigned)} K: {K}")
      print("~~~~~~~~~~~~~~~~~~~~~~~~")
      
      # Greedy MIS
      S_mis_no_ls, best_pc_mis_no_ls, total_time_mis_no_ls = solve(
        G_assigned, K, terminals_assigned, case=case_number, 
        algorithm=extended_critical_node_mis_candidate, use_ls=False)
      
      S_mis_ls, best_pc_mis_ls, total_time_mis_ls = solve(
        G_assigned, K, terminals_assigned, case=case_number, 
        algorithm=extended_critical_node_mis_candidate, use_ls=True)
      for algo, best_pc, total_time in [
        (f'Greedy MIS - no ls', best_pc_mis_no_ls, total_time_mis_no_ls),
        (f'Greedy MIS - ls', best_pc_mis_ls, total_time_mis_ls),
      ]:
        
        records.append({
          'case': int(case_number),
          'algo': str(algo),
          # 'terminals_budget': float(terminal),
          # 'K_budget': float(k_budget),
          'terminals': int(len(terminals_assigned)),
          'non_terminals': int(len_non_terminals), 
          'K': int(K),
          'obj': float(best_pc),
          'time': f"{total_time:.5f}",
        })

      SAVE_PATH_ROOT = "/home/tuguldur/Development/Research/Dev/ECNDP/Extended-Critical-Node-Detection-Problem/python/results/csv"

      df = pd.DataFrame(records)
      df.to_csv(f"{SAVE_PATH_ROOT}/Result_ECNDP_mammals_MIS_cand.csv", index=False)


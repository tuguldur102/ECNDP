import networkx as nx
import matplotlib.pyplot as plt
from algorithms.compute_PC import compute_pc
from algorithms.greedy_affinity import extended_critical_node_terminal_affinity_simple
from algorithms.exact_algorithm_new import solve_ecndp_pulp
from algorithms.greedy_empty_set import extended_critical_node_empty_set
from algorithms.greedy_mis_candidate import extended_critical_node_mis_candidate
from algorithms.greedy_aff_candidate import extended_critical_node_terminal_affinity_candidate
from utils.utils import assign_terminals_randomly, assign_terminals, create_custom_graph_extreme, create_custom_graph_with_2_comps, create_mammals_graph, solve, solve_exact
from tqdm import tqdm
import time
import pandas as pd

# Constants
SEED = 1
NODES = 100

TERMINAL_BUDGETS = {
  18: [10, 8, 5], 
  37: [18, 10, 7], 
  56: [25, 15, 10],
}

NON_TERMINAL_BUDGETS = {
  int(NODES*(1 - 0.9)): [int(0.1*i*NODES*(1 - 0.1)) for i in range(1, 6)],
  int(NODES*(1 - 0.8)): [int(0.1*i*NODES*(1 - 0.2)) for i in range(1, 6)],
  int(NODES*(1 - 0.7)): [int(0.1*i*NODES*(1 - 0.3)) for i in range(1, 6)],
  int(NODES*(1 - 0.6)): [int(0.1*i*NODES*(1 - 0.4)) for i in range(1, 6)],
  int(NODES*(1 - 0.5)): [int(0.1*i*NODES*(1 - 0.5)) for i in range(1, 6)],
}

print(NON_TERMINAL_BUDGETS)

K_BUDGETS = []

CASE_1 = 1
CASE_2 = 2

CASES = [2]
TAG = "affinity"
# Random graph creations

graph_models = {
  'ER': nx.erdos_renyi_graph(NODES, 0.05, seed=SEED),
  'BA': nx.barabasi_albert_graph(NODES, 2,seed=SEED),
  'SW': nx.watts_strogatz_graph(NODES, 4, 0.3, seed=SEED)
}

for name in ["ER"]:
  print(len(graph_models[name].nodes()), len(graph_models[name].edges()))


records = []
for case_number in tqdm(CASES, desc="Cases", total=len(CASES)):

  if case_number == 1:
    ter_budget_iter = TERMINAL_BUDGETS
  if case_number == 2:
    ter_budget_iter = NON_TERMINAL_BUDGETS

  for terminal, k_budgets in tqdm(
    ter_budget_iter.items(), desc="Terminal loop", total=len(ter_budget_iter)):

    for k_budget in tqdm(
      k_budgets, desc="deletion set loop", total=len(k_budgets)):
    

      # Graph_sample = create_mammals_graph()
      Graph_sample = graph_models["ER"].copy()

      G_assigned, terminals_assigned = assign_terminals(
        Graph_sample, terminal, SEED)

      nodes = list(G_assigned.nodes())

      len_non_terminals = int(len(nodes) - len(terminals_assigned))

      K = int(k_budget)

      # print(f"Curr pc: {compute_pc(G_assigned, set(), terminals_assigned)} \n")
      # print("~~~~~~~~~~~~~~~~~~~~~~~~")
      # print(f"nodes: {len(nodes)} edges {len(G_assigned.edges())} and terminal: {len(terminals_assigned)} K: {K}")
      # print("~~~~~~~~~~~~~~~~~~~~~~~~")

      # Greedy Empty Set
      S_es_no_ls, best_pc_es_no_ls, total_time_es_no_ls = solve(
        G_assigned, K, terminals_assigned, case=case_number, 
        algorithm=extended_critical_node_empty_set, use_ls=False)
      
      S_es_ls, best_pc_es_ls, total_time_es_ls = solve(
        G_assigned, K, terminals_assigned, case=case_number, 
        algorithm=extended_critical_node_empty_set, use_ls=True)
      
      # Greedy MIS
      S_mis_no_ls, best_pc_mis_no_ls, total_time_mis_no_ls = solve(
        G_assigned, K, terminals_assigned, case=case_number, 
        algorithm=extended_critical_node_mis_candidate, use_ls=False)
      
      S_mis_ls, best_pc_mis_ls, total_time_mis_ls = solve(
        G_assigned, K, terminals_assigned, case=case_number, 
        algorithm=extended_critical_node_mis_candidate, use_ls=True)
      
      # Affinity
      S_aff_no_ls, best_pc_aff_no_ls, total_time_aff_no_ls = solve(
        G_assigned, K, terminals_assigned, case=case_number, 
        algorithm=extended_critical_node_terminal_affinity_simple, use_ls=False)
      
      S_aff_ls, best_pc_aff_ls, total_time_aff_ls = solve(
        G_assigned, K, terminals_assigned, case=case_number, 
        algorithm=extended_critical_node_terminal_affinity_simple, use_ls=True)
      
      # Affinity cand
      _, best_pc_aff_cand_no_ls, total_time_aff_cand_no_ls = solve(
        G_assigned, K, terminals_assigned, case=case_number, 
        algorithm=extended_critical_node_terminal_affinity_candidate, use_ls=False)
      
      _, best_pc_aff_cand_ls, total_time_aff_cand_ls = solve(
        G_assigned, K, terminals_assigned, case=case_number, 
        algorithm=extended_critical_node_terminal_affinity_candidate, use_ls=True)
      
      for algo, best_pc, total_time in [
        (f'Greedy ES - no ls', best_pc_es_no_ls, total_time_es_no_ls),
        (f'Greedy ES - ls', best_pc_es_ls, total_time_es_ls),

        (f'Greedy MIS - no ls', best_pc_mis_no_ls, total_time_mis_no_ls),
        (f'Greedy MIS - ls', best_pc_mis_ls, total_time_mis_ls),

        (f'Greedy affinity - no ls', best_pc_aff_no_ls, total_time_aff_no_ls),
        (f'Greedy affinity - ls', best_pc_aff_ls, total_time_aff_ls),

        (f'Greedy affinity cand - no ls', best_pc_aff_cand_no_ls, total_time_aff_cand_no_ls),
        (f'Greedy affinity cand - ls', best_pc_aff_cand_ls, total_time_aff_cand_ls),
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

      SAVE_PATH_ROOT = r"C:\Users\tuguldur\Documents\research\ECNDP (1)\Extended-Critical-Node-Detection-Problem\python\results\csv"

      df = pd.DataFrame(records)
      df.to_csv(f"{SAVE_PATH_ROOT}/Result_ECNDP_random_ER_all_{TAG}.csv", index=False)


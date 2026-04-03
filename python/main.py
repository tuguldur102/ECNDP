import networkx as nx
import matplotlib.pyplot as plt
from algorithms.compute_PC import compute_pc
from algorithms.exact.exact_path_cplex import solve_exact_from_excerpt
from algorithms.exact.exact_formulation import solve_ecndp_cplex
from algorithms.greedy.greedy_empty_set import extended_critical_node_empty_set
from algorithms.greedy.greedy_mis_candidate import extended_critical_node_mis_candidate
from utils.utils import assign_terminals_randomly, assign_terminals, create_mammals_graph, solve
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

CASES = [1, 2]
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
      
      # Exact - classics
      # start = time.perf_counter()
      # exact_res = solve_ecndp(
      #     G=G_assigned,
      #     terminals=terminals_assigned,
      #     budget=K,
      #     case=case_number,
      #     time_limit=None,
      #     log_output=False
      # )
      # end = time.perf_counter()
      # total_time_exact = end - start

      # exact_obj = - 1
      # if exact_res["solve_status"] == "integer optimal solution":
      #   exact_obj = exact_res["objective_value"]

      # Exact - path formulation
      start = time.perf_counter()
      cpx, s_sol, x_sol, known_paths = solve_exact_from_excerpt(
        G=G_assigned,
        terminals=terminals_assigned,
        K=K,
        case=case_number
      )
      end = time.perf_counter()
      total_time_exact_path = end - start

      exact_obj_path = -1

      if cpx.solution.get_status_string() == "integer optimal solution":
        exact_obj_path = cpx.solution.get_objective_value()

      for algo, best_pc, total_time in [
        (f'Greedy ES - no ls', best_pc_es_no_ls, total_time_es_no_ls),
        (f'Greedy ES - ls', best_pc_es_ls, total_time_es_ls),

        (f'Greedy MIS - no ls', best_pc_mis_no_ls, total_time_mis_no_ls),
        (f'Greedy MIS - ls', best_pc_mis_ls, total_time_mis_ls),

        # (f'Exact', exact_obj, total_time_exact),
        (f'Exact - path', exact_obj_path, total_time_exact_path),
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

      SAVE_PATH_ROOT = "/home/tuguldur/Development/Research/Dev/ECNDP/ECNDP/python/results/csv/on_real_graph"

      df = pd.DataFrame(records)
      df.to_csv(f"{SAVE_PATH_ROOT}/Result_ECNDP_mammals_all_with_cplex_max_iter_2.csv", index=False)


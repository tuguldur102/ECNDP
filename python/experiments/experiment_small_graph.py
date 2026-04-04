import time
from pathlib import Path
from statistics import mean

import networkx as nx
import pandas as pd
from tqdm import tqdm

from algorithms.compute_PC import compute_pc
from algorithms.exact.exact_formulation import solve_ecndp_cplex
from algorithms.exact.exact_path_cplex import solve_exact_from_excerpt
from algorithms.greedy.greedy_empty_set import extended_critical_node_empty_set
from algorithms.greedy.greedy_mis_candidate import extended_critical_node_mis_candidate
from utils.assign_terminals import generate_terminals_with_fallback
from utils.utils import solve

SEED = 1
NODE_SIZES = [75]
MAX_REPETITION = 5
CONSTANT_EXACT = True

TEST = "2_reverse"
TIME_LIMIT = 1000
TERMINAL_BUDGETS = [5, 10, 15, 20, 30, 40, 50]
K_BUDGETS = [30, 20, 10]
CASES = [2, 1]

SAVE_ROOT = Path(
  "/home/tuguldur/Development/Research/Dev/ECNDP/ECNDP/python/results/csv/on_real_graph"
)


def build_budget_map(total_nodes, terminal_budgets, k_budgets, use_terminal_base):
  budget_map = {}

  for terminal_budget in terminal_budgets:
    terminal_count = round(total_nodes * (terminal_budget / 100))
    base_count = terminal_count if use_terminal_base else total_nodes - terminal_count

    unique_k_values = []
    for k_budget in k_budgets:
      k_value = max(round(base_count * (k_budget / 100)), 1)
      if k_value not in unique_k_values:
        unique_k_values.append(k_value)

    budget_map[terminal_count] = unique_k_values

  return budget_map


def run_greedy(graph, budget_k, terminals_assigned, case_number, algorithm):
  _, best_pc, total_time = solve(
    graph,
    budget_k,
    terminals_assigned,
    case=case_number,
    algorithm=algorithm,
    use_ls=True
  )
  return best_pc, total_time


def run_exact(graph, budget_k, terminals_assigned, case_number):
  start_time = time.perf_counter()

  exact_result = solve_ecndp_cplex(
    G=graph,
    terminals=terminals_assigned,
    budget=budget_k,
    case=case_number,
    time_limit=TIME_LIMIT,
    log_output=False
  )

  total_time = time.perf_counter() - start_time

  # objective_value = -1
  # if exact_result["solve_status"] == "integer optimal solution":
  exact_result_status = exact_result.solution.get_status_string()
  objective_value = exact_result.solution.get_objective_value()

  return exact_result_status, objective_value, total_time


def run_exact_path(graph, budget_k, terminals_assigned, case_number):
  start_time = time.perf_counter()

  result_path, _, _, _ = solve_exact_from_excerpt(
    G=graph,
    terminals=terminals_assigned,
    K=budget_k,
    case=case_number,
    time_limit=TIME_LIMIT,
    log_output=False
  )

  total_time = time.perf_counter() - start_time

  # objective_value = -1
  # if result_path.solution.get_status_string() == "integer optimal solution":
  exact_result_status = result_path.solution.get_status_string()
  objective_value = result_path.solution.get_objective_value()

  return exact_result_status, objective_value, total_time


for total_nodes in NODE_SIZES:
  save_path = SAVE_ROOT / f"Result_ECNDP_all_with_{total_nodes}_{MAX_REPETITION}_reps_{TEST}.csv"

  terminal_node_budgets = build_budget_map(
    total_nodes,
    TERMINAL_BUDGETS,
    K_BUDGETS,
    use_terminal_base=True
  )
  non_terminal_node_budgets = build_budget_map(
    total_nodes,
    TERMINAL_BUDGETS,
    K_BUDGETS,
    use_terminal_base=False
  )

  print()
  print(f"===== NODES = {total_nodes} =====")
  print("terminal budgets:", terminal_node_budgets)
  print("non-terminal budgets:", non_terminal_node_budgets)
  print()

  if total_nodes == 75:
    edge_prob = 0.05
  elif total_nodes == 100:
    edge_prob = 0.042
  elif total_nodes == 150:
    edge_prob = 0.028

  graph_models = {
    "ER": nx.erdos_renyi_graph(total_nodes, edge_prob, seed=SEED),
    "BA": nx.barabasi_albert_graph(total_nodes, 2, seed=SEED),
    "SW": nx.watts_strogatz_graph(total_nodes, 4, 0.3, seed=SEED),
  }

  for name in ["ER", "BA", "SW"]:
    print(
      name,
      len(graph_models[name].nodes()),
      len(graph_models[name].edges())
    )

  records = []

  for case_number in tqdm(CASES, desc=f"Cases ({total_nodes} nodes)"):
    budget_map = terminal_node_budgets if case_number == 1 else non_terminal_node_budgets

    for terminal_count, k_values in tqdm(budget_map.items(), desc="Terminal loop"):
      for budget_k in tqdm(k_values, desc="Deletion set loop"):
        graph_assigned = graph_models["ER"].copy()

        results = {
          "Greedy ES - ls": {"sol_status": [], "obj": [], "time": []},
          "Greedy MIS - ls": {"sol_status": [], "obj": [], "time": []},
        }

        results["Exact - path"] = {"sol_status": [], "obj": [], "time": []}
        if CONSTANT_EXACT:
          results["Exact"] = {"sol_status": [], "obj": [], "time": []}

        success_count = 0

        for repetition in tqdm(range(MAX_REPETITION), desc="max_repetition"):
          try:
            terminal_result = generate_terminals_with_fallback(
              G=graph_assigned.copy(),
              terminal_count=terminal_count,
              distribution="uniform",
              random_seed=repetition,
              max_assignment_retries=1000
            )

            terminals_assigned = terminal_result["terminals"]

            if repetition == 1:
              curr_pc = compute_pc(graph_assigned, set(), terminals_assigned)
              print(f"\n~~~~~Curr pc: {curr_pc}~~~~~\n")

            objective_value, total_time = run_greedy(
              graph_assigned,
              budget_k,
              terminals_assigned,
              case_number,
              extended_critical_node_empty_set
            )
            results["Greedy ES - ls"]["obj"].append(objective_value)
            results["Greedy ES - ls"]["time"].append(total_time)

            objective_value, total_time = run_greedy(
              graph_assigned,
              budget_k,
              terminals_assigned,
              case_number,
              extended_critical_node_mis_candidate
            )
            results["Greedy MIS - ls"]["obj"].append(objective_value)
            results["Greedy MIS - ls"]["time"].append(total_time)

            if CONSTANT_EXACT:
              sol_status, objective_value, total_time = run_exact(
                graph_assigned,
                budget_k,
                terminals_assigned,
                case_number
              )
              results["Exact"]["sol_status"].append(sol_status)
              results["Exact"]["obj"].append(objective_value)
              results["Exact"]["time"].append(total_time)

            sol_status, objective_value, total_time = run_exact_path(
              graph_assigned,
              budget_k,
              terminals_assigned,
              case_number
            )
            results["Exact - path"]["sol_status"].append(sol_status)
            results["Exact - path"]["obj"].append(objective_value)
            results["Exact - path"]["time"].append(total_time)

            success_count += 1

          except RuntimeError as error:
            print(f"repetition {repetition} failed: {error}")

        if success_count == 0:
          continue

        for algorithm_name, algorithm_result in results.items():
          records.append({
            "nodes": total_nodes,
            "case": case_number,
            "algo": algorithm_name,
            "terminals": terminal_count,
            "non_terminals": total_nodes - terminal_count,
            "K": budget_k,
            "ave_obj": sum(algorithm_result["obj"]) / len(algorithm_result["obj"]) if not len(algorithm_result["obj"]) == 0 else 0,
            "ave_time": f"{sum(algorithm_result["time"]) / len(algorithm_result["time"]):.5f}" if not len(algorithm_result["time"]) == 0 else 0,
            "sol_status": algorithm_result["sol_status"],
            "obj": algorithm_result["obj"],
            "time": algorithm_result["time"],
          })

        pd.DataFrame(records).to_csv(save_path, index=False)

  print(f"Saved: {save_path}")
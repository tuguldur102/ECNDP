from itertools import combinations
import networkx as nx
from docplex.mp.model import Model

def solve_ecndp(
  G,
  terminals,
  budget,
  case=1,
  time_limit=None,
  log_output=True
):

  vertex_list = list(G.nodes())
  vertex_set = set(vertex_list)
  terminal_set = set(terminals)

  if not terminal_set.issubset(vertex_set):
    raise ValueError("terminals must be a subset of G.nodes")

  vertex_position = {vertex: idx for idx, vertex in enumerate(vertex_list)}

  def unordered_pair(first_vertex, second_vertex):
    if first_vertex == second_vertex:
      raise ValueError("pair must contain two distinct vertices")
    if vertex_position[first_vertex] < vertex_position[second_vertex]:
      return (first_vertex, second_vertex)
    return (second_vertex, first_vertex)

  normalized_edges = set()
  for first_vertex, second_vertex in G.edges():
    if first_vertex == second_vertex:
      continue
    normalized_edges.add(unordered_pair(first_vertex, second_vertex))

  unordered_vertex_pairs = [
    (vertex_list[first_idx], vertex_list[second_idx])
    for first_idx in range(len(vertex_list))
    for second_idx in range(first_idx + 1, len(vertex_list))
  ]

  terminal_pairs = [
    unordered_pair(first_terminal, second_terminal)
    for first_terminal, second_terminal in combinations(terminal_set, 2)
  ]

  model = Model(name="ECNDP")

  if time_limit is not None:
    model.parameters.timelimit = time_limit

  remove_vertex = model.binary_var_dict(
    vertex_list,
    name="remove_vertex"
  )

  same_component = model.binary_var_dict(
    unordered_vertex_pairs,
    name="same_component"
  )

  # Objective - minimize sum of x_ij over terminal pairs
  model.minimize(
    model.sum(
      same_component[pairwise_connectivity]
      for pairwise_connectivity in terminal_pairs
    )
  )

  # Budget constraint
  if case == 1:
    model.add_constraint(
      model.sum(remove_vertex[vertex] for vertex in vertex_list) <= budget,
      ctname="budget"
    )
  else:
    non_terminal_vertices = [vertex for vertex in vertex_list if vertex not in terminal_set]
    model.add_constraint(
      model.sum(remove_vertex[vertex] for vertex in non_terminal_vertices) <= budget,
      ctname="budget_non_terminals"
    )
    for terminal_vertex in terminal_set:
      model.add_constraint(
        remove_vertex[terminal_vertex] == 0,
        ctname=f"terminal_not_removed_{terminal_vertex}"
      )

  # Edge constraints:
  for first_vertex, second_vertex in normalized_edges:
    model.add_constraint(
      same_component[(first_vertex, second_vertex)] >=
      1 - remove_vertex[first_vertex] - remove_vertex[second_vertex]
    )

  # Deleted-vertex constraints:
  for first_vertex, second_vertex in unordered_vertex_pairs:
    model.add_constraint(
      same_component[(first_vertex, second_vertex)] <= 1 - remove_vertex[first_vertex]
    )
    model.add_constraint(
      same_component[(first_vertex, second_vertex)] <= 1 - remove_vertex[second_vertex]
    )

  # Transitivity constraints
  for first_vertex, second_vertex, third_vertex in combinations(vertex_list, 3):
    pair_ab = unordered_pair(first_vertex, second_vertex)
    pair_ac = unordered_pair(first_vertex, third_vertex)
    pair_bc = unordered_pair(second_vertex, third_vertex)

    model.add_constraint(
      same_component[pair_ac] >=
      same_component[pair_ab] + same_component[pair_bc] - 1
    )
    model.add_constraint(
      same_component[pair_ab] >=
      same_component[pair_ac] + same_component[pair_bc] - 1
    )
    model.add_constraint(
      same_component[pair_bc] >=
      same_component[pair_ab] + same_component[pair_ac] - 1
    )

  solution = model.solve(log_output=log_output)

  if solution is None:
    return {
      "objective_value": None,
      "removed_vertices": [],
      "same_component_pairs": [],
      "solve_status": str(model.solve_details.status)
    }

  # removed_vertices = [
  #   vertex
  #   for vertex in vertex_list
  #   if solution.get_value(remove_vertex[vertex]) > 0.5
  # ]

  # same_component_pairs = [
  #   pairwise_connectivity
  #   for pairwise_connectivity in unordered_vertex_pairs
  #   if solution.get_value(same_component[pairwise_connectivity]) > 0.5
  # ]

  return {
    "objective_value": solution.objective_value,
    "removed_vertices": [],
    "same_component_pairs": [],
    "solve_status": str(model.solve_details.status)
  }


# if __name__ == "__main__":
#   G = nx.Graph()
#   G.add_edges_from([
#     (1, 2),
#     (2, 3),
#     (3, 4),
#     (4, 5),
#     (1, 5),
#     (2, 4)
#   ])

#   terminals = [1, 3, 5]
#   budget = 1

#   result = solve_ecndp(
#     G=G,
#     terminals=terminals,
#     budget=budget,
#     case=2,
#     time_limit=60
#   )

#   print("status:", result["solve_status"])
#   print("objective value:", result["objective_value"])
  # print("removed vertices:", result["removed_vertices"])
  # print("same-component pairs:", result["same_component_pairs"])
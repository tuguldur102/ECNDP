import cplex
import networkx as nx
from itertools import combinations


def solve_ecndp_cplex(
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

  cpx = cplex.Cplex()
  cpx.objective.set_sense(cpx.objective.sense.minimize)

  if not log_output:
    cpx.set_log_stream(None)
    cpx.set_error_stream(None)
    cpx.set_warning_stream(None)
    cpx.set_results_stream(None)

  if time_limit is not None:
    cpx.parameters.timelimit.set(time_limit)

  remove_vertex_names = [f"remove_vertex_{vertex}" for vertex in vertex_list]
  cpx.variables.add(
    names=remove_vertex_names,
    lb=[0.0] * len(vertex_list),
    ub=[1.0] * len(vertex_list),
    types=["B"] * len(vertex_list),
    obj=[0.0] * len(vertex_list)
  )
  remove_vertex_idx = {
    vertex: cpx.variables.get_indices(f"remove_vertex_{vertex}")
    for vertex in vertex_list
  }

  same_component_names = [
    f"same_component_{first_vertex}_{second_vertex}"
    for first_vertex, second_vertex in unordered_vertex_pairs
  ]
  same_component_obj = [
    1.0 if pairwise_connectivity in terminal_pairs else 0.0
    for pairwise_connectivity in unordered_vertex_pairs
  ]
  cpx.variables.add(
    names=same_component_names,
    lb=[0.0] * len(unordered_vertex_pairs),
    ub=[1.0] * len(unordered_vertex_pairs),
    types=["B"] * len(unordered_vertex_pairs),
    obj=same_component_obj
  )
  same_component_idx = {
    pairwise_connectivity: cpx.variables.get_indices(
      f"same_component_{pairwise_connectivity[0]}_{pairwise_connectivity[1]}"
    )
    for pairwise_connectivity in unordered_vertex_pairs
  }

  # Budget constraint
  if case == 1:
    cpx.linear_constraints.add(
      lin_expr=[cplex.SparsePair(
        ind=[remove_vertex_idx[vertex] for vertex in vertex_list],
        val=[1.0] * len(vertex_list)
      )],
      senses=["L"],
      rhs=[float(budget)],
      names=["budget"]
    )
  else:
    non_terminal_vertices = [
      vertex for vertex in vertex_list if vertex not in terminal_set
    ]
    cpx.linear_constraints.add(
      lin_expr=[cplex.SparsePair(
        ind=[remove_vertex_idx[vertex] for vertex in non_terminal_vertices],
        val=[1.0] * len(non_terminal_vertices)
      )],
      senses=["L"],
      rhs=[float(budget)],
      names=["budget_non_terminals"]
    )

    for terminal_vertex in terminal_set:
      cpx.linear_constraints.add(
        lin_expr=[cplex.SparsePair(
          ind=[remove_vertex_idx[terminal_vertex]],
          val=[1.0]
        )],
        senses=["E"],
        rhs=[0.0],
        names=[f"terminal_not_removed_{terminal_vertex}"]
      )

  # Edge constraints:
  # same_component[u,v] >= 1 - remove_vertex[u] - remove_vertex[v]
  # equivalent: same_component[u,v] + remove_vertex[u] + remove_vertex[v] >= 1
  for first_vertex, second_vertex in normalized_edges:
    cpx.linear_constraints.add(
      lin_expr=[cplex.SparsePair(
        ind=[
          same_component_idx[(first_vertex, second_vertex)],
          remove_vertex_idx[first_vertex],
          remove_vertex_idx[second_vertex]
        ],
        val=[1.0, 1.0, 1.0]
      )],
      senses=["G"],
      rhs=[1.0]
    )

  # Deleted-vertex constraints:
  # same_component[u,v] <= 1 - remove_vertex[u]
  # equivalent: same_component[u,v] + remove_vertex[u] <= 1
  for first_vertex, second_vertex in unordered_vertex_pairs:
    cpx.linear_constraints.add(
      lin_expr=[cplex.SparsePair(
        ind=[
          same_component_idx[(first_vertex, second_vertex)],
          remove_vertex_idx[first_vertex]
        ],
        val=[1.0, 1.0]
      )],
      senses=["L"],
      rhs=[1.0]
    )

    cpx.linear_constraints.add(
      lin_expr=[cplex.SparsePair(
        ind=[
          same_component_idx[(first_vertex, second_vertex)],
          remove_vertex_idx[second_vertex]
        ],
        val=[1.0, 1.0]
      )],
      senses=["L"],
      rhs=[1.0]
    )

  # Transitivity constraints
  # x_ac >= x_ab + x_bc - 1
  # equivalent: x_ab + x_bc - x_ac <= 1
  for first_vertex, second_vertex, third_vertex in combinations(vertex_list, 3):
    pair_ab = unordered_pair(first_vertex, second_vertex)
    pair_ac = unordered_pair(first_vertex, third_vertex)
    pair_bc = unordered_pair(second_vertex, third_vertex)

    cpx.linear_constraints.add(
      lin_expr=[cplex.SparsePair(
        ind=[
          same_component_idx[pair_ab],
          same_component_idx[pair_bc],
          same_component_idx[pair_ac]
        ],
        val=[1.0, 1.0, -1.0]
      )],
      senses=["L"],
      rhs=[1.0]
    )

    cpx.linear_constraints.add(
      lin_expr=[cplex.SparsePair(
        ind=[
          same_component_idx[pair_ac],
          same_component_idx[pair_bc],
          same_component_idx[pair_ab]
        ],
        val=[1.0, 1.0, -1.0]
      )],
      senses=["L"],
      rhs=[1.0]
    )

    cpx.linear_constraints.add(
      lin_expr=[cplex.SparsePair(
        ind=[
          same_component_idx[pair_ab],
          same_component_idx[pair_ac],
          same_component_idx[pair_bc]
        ],
        val=[1.0, 1.0, -1.0]
      )],
      senses=["L"],
      rhs=[1.0]
    )

  cpx.solve()

  return cpx
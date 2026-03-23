from algorithms.compute_PC import obj

def improvement_condition(G, K, best_pc, terminals):
  return obj(G, K, terminals) < best_pc
import math
import random
import networkx as nx
import matplotlib.pyplot as plt


def compute_max_batch_size(terminal_count):
  return min(5, max(1, int(math.floor(0.15 * terminal_count))))


def is_partition_possible(terminal_count, min_batch_size, max_batch_size):
  if min_batch_size > max_batch_size:
    return False

  allowed_sizes = list(range(min_batch_size, max_batch_size + 1))

  reachable = [False] * (terminal_count + 1)
  reachable[0] = True

  for current_sum in range(terminal_count + 1):
    if not reachable[current_sum]:
      continue

    for batch_size in allowed_sizes:
      next_sum = current_sum + batch_size
      if next_sum <= terminal_count:
        reachable[next_sum] = True

  return reachable[terminal_count]


def generate_batch_sizes(
  terminal_count,
  min_batch_size,
  max_batch_size,
  distribution="uniform",
  bias_large=False,
  random_generator=None
):
  if random_generator is None:
    random_generator = random.Random()

  allowed_sizes = list(range(min_batch_size, max_batch_size + 1))

  if distribution == "uniform":
    weights = [1.0 for _ in allowed_sizes]

  elif distribution == "normal-like":
    center = (min_batch_size + max_batch_size) / 2.0
    sigma = max(0.75, (max_batch_size - min_batch_size + 1) / 2.0)
    weights = [
      math.exp(-((batch_size - center) ** 2) / (2.0 * sigma * sigma))
      for batch_size in allowed_sizes
    ]

  else:
    raise ValueError("distribution must be 'uniform' or 'normal-like'")

  if bias_large:
    weights = [
      base_weight * (1.0 + batch_size)
      for batch_size, base_weight in zip(allowed_sizes, weights)
    ]

  batch_sizes = []
  remaining = terminal_count

  while remaining > 0:
    feasible_sizes = [
      batch_size
      for batch_size in allowed_sizes
      if batch_size <= remaining
      and is_partition_possible(
        remaining - batch_size,
        min_batch_size,
        max_batch_size
      )
    ]

    if not feasible_sizes:
      raise RuntimeError("No feasible next batch size found")

    feasible_weights = [
      weights[allowed_sizes.index(batch_size)]
      for batch_size in feasible_sizes
    ]

    chosen_batch_size = random_generator.choices(
      feasible_sizes,
      weights=feasible_weights,
      k=1
    )[0]

    batch_sizes.append(chosen_batch_size)
    remaining -= chosen_batch_size

  return batch_sizes


def assign_terminal_batches(
  G,
  batch_sizes,
  random_generator=None,
  max_seed_trials=300
):
  if random_generator is None:
    random_generator = random.Random()

  assigned_terminals = set()
  terminal_batches = []
  all_nodes = list(G.nodes())

  for batch_size in batch_sizes:
    success = False

    for _ in range(max_seed_trials):
      seed_node = random_generator.choice(all_nodes)

      if seed_node in assigned_terminals:
        continue

      candidate_batch = []
      queue = [seed_node]
      visited = set()

      while queue and len(candidate_batch) < batch_size:
        current_node = queue.pop(0)

        if current_node in visited:
          continue
        visited.add(current_node)

        if current_node in assigned_terminals:
          continue

        has_cross_batch_terminal_edge = any(
          neighbor in assigned_terminals
          for neighbor in G.neighbors(current_node)
        )

        if has_cross_batch_terminal_edge:
          continue

        candidate_batch.append(current_node)

        for neighbor in G.neighbors(current_node):
          if neighbor not in visited:
            queue.append(neighbor)

      if len(candidate_batch) == batch_size:
        terminal_batches.append(candidate_batch)
        assigned_terminals.update(candidate_batch)
        success = True
        break

    if not success:
      return None

  return terminal_batches


def generate_terminals_with_fallback(
  G,
  terminal_count,
  distribution="uniform",
  random_seed=None,
  max_assignment_retries=20
):
  random_generator = random.Random(random_seed)

  base_max_batch_size = compute_max_batch_size(terminal_count)

  # Internal settings only, not user-facing parameters.
  max_seed_trials = 300
  hard_max_batch_size = int(terminal_count / 2)

  for current_max_batch_size in range(
    base_max_batch_size,
    hard_max_batch_size + 1
  ):
    for current_min_batch_size in range(1, current_max_batch_size + 1):
      if not is_partition_possible(
        terminal_count,
        current_min_batch_size,
        current_max_batch_size
      ):
        continue

      for _ in range(max_assignment_retries):
        batch_sizes = generate_batch_sizes(
          terminal_count=terminal_count,
          min_batch_size=current_min_batch_size,
          max_batch_size=current_max_batch_size,
          distribution=distribution,
          bias_large=True,
          random_generator=random_generator
        )

        terminal_batches = assign_terminal_batches(
          G=G,
          batch_sizes=batch_sizes,
          random_generator=random_generator,
          max_seed_trials=max_seed_trials
        )

        if terminal_batches is not None:
          terminals = [node for batch in terminal_batches for node in batch]
          return {
            "terminals": terminals,
            "terminal_batches": terminal_batches,
            "batch_sizes": batch_sizes,
            "min_batch_size": current_min_batch_size,
            "max_batch_size": current_max_batch_size,
            "base_max_batch_size": base_max_batch_size,
            "used_large_bias": True
          }

      for _ in range(max_assignment_retries):
        batch_sizes = generate_batch_sizes(
          terminal_count=terminal_count,
          min_batch_size=current_min_batch_size,
          max_batch_size=current_max_batch_size,
          distribution=distribution,
          bias_large=True,
          random_generator=random_generator
        )

        terminal_batches = assign_terminal_batches(
          G=G,
          batch_sizes=batch_sizes,
          random_generator=random_generator,
          max_seed_trials=max_seed_trials
        )

        if terminal_batches is not None:
          terminals = [node for batch in terminal_batches for node in batch]
          return {
            "terminals": terminals,
            "terminal_batches": terminal_batches,
            "batch_sizes": batch_sizes,
            "min_batch_size": current_min_batch_size,
            "max_batch_size": current_max_batch_size,
            "base_max_batch_size": base_max_batch_size,
            "used_large_bias": True
          }

  raise RuntimeError("Failed to generate terminal batches with fallback policy")

def visualize_terminal_batches(
  G,
  terminal_batches,
  layout_seed=1,
  node_size=220,
  font_size=8,
  show_labels=True
):
  position = nx.spring_layout(G, seed=layout_seed)

  terminal_to_batch = {}
  for batch_idx, terminal_batch in enumerate(terminal_batches):
    for terminal_node in terminal_batch:
      terminal_to_batch[terminal_node] = batch_idx

  terminal_nodes = set(terminal_to_batch.keys())
  non_terminal_nodes = [node for node in G.nodes() if node not in terminal_nodes]

  plt.figure(figsize=(10, 8))

  nx.draw_networkx_edges(
    G,
    position,
    alpha=0.35
  )

  nx.draw_networkx_nodes(
    G,
    position,
    nodelist=non_terminal_nodes,
    node_color="lightgray",
    node_size=node_size
  )

  color_map = [
    "red",
    "blue",
    "green",
    "orange",
    "purple",
    "brown",
    "pink",
    "olive",
    "cyan",
    "magenta"
  ]

  for batch_idx, terminal_batch in enumerate(terminal_batches):
    batch_color = color_map[batch_idx % len(color_map)]

    nx.draw_networkx_nodes(
      G,
      position,
      nodelist=terminal_batch,
      node_color=batch_color,
      node_size=node_size + 80,
      edgecolors="black",
      linewidths=1.2,
      label=f"batch {batch_idx + 1}"
    )

  if show_labels:
    nx.draw_networkx_labels(
      G,
      position,
      font_size=font_size
    )

  cross_batch_edges = []
  for first_node, second_node in G.edges():
    first_is_terminal = first_node in terminal_to_batch
    second_is_terminal = second_node in terminal_to_batch

    if first_is_terminal and second_is_terminal:
      if terminal_to_batch[first_node] != terminal_to_batch[second_node]:
        cross_batch_edges.append((first_node, second_node))

  if cross_batch_edges:
    nx.draw_networkx_edges(
      G,
      position,
      edgelist=cross_batch_edges,
      width=2.5,
      style="dashed",
      edge_color="black"
    )

  plt.legend()
  plt.title("Terminal batches on graph")
  plt.axis("off")
  plt.tight_layout()
  plt.show()

def find_cross_batch_terminal_edges(G, terminal_batches):
  terminal_to_batch = {}

  for batch_idx, terminal_batch in enumerate(terminal_batches):
    for terminal_node in terminal_batch:
      terminal_to_batch[terminal_node] = batch_idx

  cross_batch_edges = []

  for first_node, second_node in G.edges():
    if first_node in terminal_to_batch and second_node in terminal_to_batch:
      if terminal_to_batch[first_node] != terminal_to_batch[second_node]:
        cross_batch_edges.append((first_node, second_node))

  return cross_batch_edges
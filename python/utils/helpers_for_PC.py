from collections import deque
import networkx as nx
import math

def bfs(G: nx.Graph, source: int, visited: set[int]) -> list[int]:
  queue = deque()
  visited.add(source)
  res = []

  # visited[source] = True
  queue.append(source)

  while len(queue) > 0:

    v = queue.popleft()
    res.append(v)
    # print(f"queue: {v}")

    for u in G.neighbors(v):
      if u not in visited:
        visited.add(u)
        queue.append(u)

  return res

def connected_components(G: nx.Graph) -> list[list[int]]:
  visited : set[int] = set()
  components : list[list[int]] = []

  V = list(G.nodes)

  for v in V:
    if v not in visited:
      comp = bfs(G, v, visited)
      components.append(comp)
      # yield comp

  return components

def comb_of_two(components: list[int]) -> int:
  pairwise_connectivity : int = 0

  for comp in components:
    n = int(len(comp))
    pairwise_connectivity += math.comb(n, 2)

  return pairwise_connectivity
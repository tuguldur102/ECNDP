import networkx as nx
from constants import GraphType, Seed
from dataclasses import dataclass

# Enum + dataclass + factory pattern for random graph generating

@dataclass
class GraphSpec:
  graph_type: GraphType
  seed: Seed | None
  n: int
  p: float | None = None
  k: int | None = None
  m: int | None = None

  def __post_init__(self):

    if self.graph_type is GraphType.ERDOS_RENYI and self.p is None:
      raise ValueError("ERDOS_RENYI requires p")
    
    if self.graph_type is GraphType.WATTS_STROGATZ and (self.k is None and self.p is None):
      raise ValueError("WATTS_STROGATTZ requires k and p")
  
    if self.graph_type is GraphType.BARABASI_ALBERT and self.m is None:
      raise ValueError("BARABASI_ALBERT requires m")
    
def make_graph(spec: GraphSpec):

  match spec.graph_type:
    case GraphType.ERDOS_RENYI:
      return nx.erdos_renyi_graph(spec.n, spec.p, spec.seed)
  
    case GraphType.WATTS_STROGATZ:
      return nx.watts_strogatz_graph(spec.n, spec.k, spec.p, spec.seed)
  
    case GraphType.BARABASI_ALBERT:
      return nx.barabasi_albert_graph(spec.n, spec.m, spec.seed)
    
    case _:
     raise ValueError("Unsupported Graph Type!")
  
def create_custom_graph_with_2_comps(terminal_nodes: list[int]) -> nx.Graph:

  G = nx.Graph()

  C1 = [(0, 1), (1, 2), (2, 3)]
  C2 = [(4, 5), (5, 6)]

  G.add_edges_from(C1)
  G.add_edges_from(C2)

  # Add terminal attribute to nodes
  for v in G.nodes:
    if v in terminal_nodes:
      G.nodes[v]["terminal"] = True
    else:
      G.nodes[v]["terminal"] = False
  
  return G
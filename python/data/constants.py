from enum import Enum
import networkx as nx

class GraphType(Enum):
  ERDOS_RENYI = "erdos_renyi"
  WATTS_STROGATZ = "watts_strogatz"
  BARABASI_ALBERT = "barabasi_albert"

class Seed(Enum):
  ONETWOTHREE = 123
  ALL_MEANING = 42
  ONE = 1
  TWO = 2
  THREE = 3

class Graph(Enum):
  V = "vertices"
  E = "edges"
  V_T = "vertices_excluded_terminals"



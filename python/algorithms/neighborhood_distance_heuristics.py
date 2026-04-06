from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import pandas as pd
import pyomo.environ as pyo


# ============================================================
# Core objective
# ============================================================

def terminal_pairwise_connectivity(
    G: nx.Graph,
    terminals: Set,
    deleted: Set,
) -> int:
    """Return \sum_C binom(tau(C), 2) on the original graph after deleting nodes."""
    if not deleted:
        H = G
    else:
        H = G.subgraph([v for v in G.nodes if v not in deleted])

    total = 0
    for comp in nx.connected_components(H):
        tau = sum(1 for v in comp if v in terminals)
        total += tau * (tau - 1) // 2
    return total


# ============================================================
# Exact model (full connectivity / transitivity formulation)
# ============================================================

@dataclass
class ExactResult:
    deleted: Set
    objective: float
    runtime_sec: float
    termination: str


def _pair(u, v):
    if u == v:
        raise ValueError("Self-pairs are not allowed.")
    return (u, v) if u < v else (v, u)


def solve_tcndp_exact_highs(
    G: nx.Graph,
    terminals: Set,
    k: int,
    case: int = 1,
    deletion_cost: Optional[Dict] = None,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    verbose: bool = False,
) -> ExactResult:
    """
    Exact TCNDP via the connectivity/transitivity MIP, solved by HiGHS.

    Parameters
    ----------
    G : undirected NetworkX graph
    terminals : set of terminal nodes
    k : deletion budget
    case : 1 (terminals can be deleted) or 2 (terminals protected)
    deletion_cost : optional deletion cost per node; default 1 for all nodes
    time_limit : optional time limit in seconds
    mip_gap : optional relative MIP gap
    verbose : print HiGHS log
    """
    if case not in (1, 2):
        raise ValueError("case must be 1 or 2")
    if not terminals.issubset(set(G.nodes)):
        raise ValueError("terminals must be a subset of graph nodes")

    nodes = list(G.nodes)
    nset = set(nodes)
    if deletion_cost is None:
        deletion_cost = {v: 1 for v in nodes}
    else:
        missing = nset - set(deletion_cost)
        if missing:
            raise ValueError(f"Missing deletion costs for nodes: {sorted(missing)!r}")

    term_list = sorted(terminals)
    pairs = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]
    t_pairs = [(term_list[i], term_list[j]) for i in range(len(term_list)) for j in range(i + 1, len(term_list))]

    m = pyo.ConcreteModel()
    m.V = pyo.Set(initialize=nodes, ordered=True)
    m.PAIRS = pyo.Set(initialize=pairs, dimen=2, ordered=True)
    m.TPAIRS = pyo.Set(initialize=t_pairs, dimen=2, ordered=True)

    m.s = pyo.Var(m.V, domain=pyo.Binary)
    m.x = pyo.Var(m.PAIRS, domain=pyo.Binary)

    # Objective: pairwise terminal connectivity in the residual graph.
    m.obj = pyo.Objective(expr=sum(m.x[i, j] for i, j in m.TPAIRS), sense=pyo.minimize)

    # Weighted budget.
    m.budget = pyo.Constraint(expr=sum(deletion_cost[v] * m.s[v] for v in m.V) <= k)

    # Terminal protection for Case 2.
    m.protect = pyo.ConstraintList()
    if case == 2:
        for t in terminals:
            m.protect.add(m.s[t] == 0)

    # Adjacent undeleted vertices must be connected.
    m.edge_link = pyo.ConstraintList()
    for u, v in G.edges:
        a, b = _pair(u, v)
        m.edge_link.add(m.x[a, b] >= 1 - m.s[u] - m.s[v])

    # Deleted vertices cannot be declared connected.
    m.upper = pyo.ConstraintList()
    for i, j in pairs:
        m.upper.add(m.x[i, j] <= 1 - m.s[i])
        m.upper.add(m.x[i, j] <= 1 - m.s[j])

    # Transitivity of connectivity.
    # If i-j and j-k are connected, then i-k must be connected.
    m.trans = pyo.ConstraintList()
    for i, j, k2 in itertools.permutations(nodes, 3):
        if len({i, j, k2}) < 3:
            continue
        a, b = _pair(i, j)
        c, d = _pair(j, k2)
        e, f = _pair(i, k2)
        m.trans.add(m.x[e, f] >= m.x[a, b] + m.x[c, d] - 1)

    solver = pyo.SolverFactory("appsi_highs")
    if solver is None or not solver.available():
        raise RuntimeError(
            "HiGHS is not available through Pyomo. Install `highspy` and `pyomo`, "
            "then make sure `SolverFactory('appsi_highs')` is available."
        )

    if time_limit is not None:
        solver.config.time_limit = float(time_limit)
    if not verbose:
        solver.config.stream_solver = False
    if mip_gap is not None:
        # HiGHS option name for relative MIP gap.
        solver.highs_options["mip_rel_gap"] = float(mip_gap)

    t0 = time.perf_counter()
    res = solver.solve(m)
    runtime = time.perf_counter() - t0

    term = str(res.termination_condition)
    deleted = {v for v in nodes if pyo.value(m.s[v]) >= 0.5}
    obj = float(pyo.value(m.obj))

    return ExactResult(
        deleted=deleted,
        objective=obj,
        runtime_sec=runtime,
        termination=term,
    )


# ============================================================
# Baseline heuristics
# ============================================================

@dataclass
class HeuristicResult:
    name: str
    deleted: Set
    objective: float
    runtime_sec: float
    details: Dict[str, object]


def greedy_constructive(
    G: nx.Graph,
    terminals: Set,
    k: int,
    case: int = 1,
) -> HeuristicResult:
    t0 = time.perf_counter()
    if case == 1:
        candidates = set(G.nodes)
    else:
        candidates = set(G.nodes) - set(terminals)

    deleted: Set = set()
    for _ in range(k):
        best_v = None
        best_obj = math.inf
        for v in sorted(candidates - deleted):
            cand = deleted | {v}
            obj = terminal_pairwise_connectivity(G, terminals, cand)
            if obj < best_obj:
                best_obj = obj
                best_v = v
        if best_v is None:
            break
        deleted.add(best_v)

    runtime = time.perf_counter() - t0
    obj = terminal_pairwise_connectivity(G, terminals, deleted)
    return HeuristicResult(
        name="greedy",
        deleted=deleted,
        objective=float(obj),
        runtime_sec=runtime,
        details={},
    )


def greedy_local_search(
    G: nx.Graph,
    terminals: Set,
    initial_deleted: Set,
    case: int = 1,
    max_no_improve_passes: int = 1,
) -> HeuristicResult:
    t0 = time.perf_counter()
    deleted = set(initial_deleted)
    current = terminal_pairwise_connectivity(G, terminals, deleted)

    if case == 1:
        deletable = set(G.nodes)
    else:
        deletable = set(G.nodes) - set(terminals)

    no_improve_passes = 0
    while no_improve_passes < max_no_improve_passes:
        improved = False
        outside = list(sorted(deletable - deleted))
        inside = list(sorted(deleted))
        for v_out in inside:
            for v_in in outside:
                cand = (deleted - {v_out}) | {v_in}
                obj = terminal_pairwise_connectivity(G, terminals, cand)
                if obj < current:
                    deleted = cand
                    current = obj
                    improved = True
                    break
            if improved:
                break
        if improved:
            no_improve_passes = 0
        else:
            no_improve_passes += 1

    runtime = time.perf_counter() - t0
    return HeuristicResult(
        name="greedy+local_search",
        deleted=deleted,
        objective=float(current),
        runtime_sec=runtime,
        details={},
    )


# ============================================================
# New heuristic: coarsen nonterminals, then solve reduced problem exactly
# ============================================================

@dataclass
class CoarsenNode:
    members: Set
    is_terminal: bool
    cost: int


def neighbor_set_distance(G: nx.Graph, u, v) -> float:
    """
    Practical regularization of the user's distance:
        dist(u, v) = |N(u) Δ N(v)| / max(1, |N(u) ∩ N(v)|)
    where Δ is symmetric difference.
    """
    Nu = set(G.neighbors(u))
    Nv = set(G.neighbors(v))
    edit_dist = len(Nu.symmetric_difference(Nv))
    common = len(Nu.intersection(Nv))
    return edit_dist / max(0.000001, common)


def build_initial_coarsened_graph(G: nx.Graph, terminals: Set) -> nx.Graph:
    H = nx.Graph()
    for v in G.nodes:
        H.add_node(
            v,
            data=CoarsenNode(members={v}, is_terminal=(v in terminals), cost=1),
        )
    H.add_edges_from(G.edges)
    return H


def merge_nonterminal_pair(H: nx.Graph, a, b, new_id) -> None:
    data_a: CoarsenNode = H.nodes[a]["data"]
    data_b: CoarsenNode = H.nodes[b]["data"]
    if data_a.is_terminal or data_b.is_terminal:
        raise ValueError("Only nonterminal nodes may be merged.")

    nbrs = (set(H.neighbors(a)) | set(H.neighbors(b))) - {a, b}
    new_data = CoarsenNode(
        members=set(data_a.members) | set(data_b.members),
        is_terminal=False,
        cost=data_a.cost + data_b.cost,
    )

    H.add_node(new_id, data=new_data)
    for w in nbrs:
        H.add_edge(new_id, w)
    H.remove_node(a)
    H.remove_node(b)


def coarsen_graph_by_neighbor_distance(
    G: nx.Graph,
    terminals: Set,
    target_nonterminal_ratio: float = 0.5,
) -> nx.Graph:
    if not (0 < target_nonterminal_ratio <= 1):
        raise ValueError("target_nonterminal_ratio must lie in (0, 1].")

    H = build_initial_coarsened_graph(G, terminals)
    orig_nonterm = sum(1 for v in H.nodes if not H.nodes[v]["data"].is_terminal)
    target_nonterm = max(1, math.ceil(orig_nonterm * target_nonterminal_ratio))

    merge_id = 0
    while True:
        nonterms = [v for v in H.nodes if not H.nodes[v]["data"].is_terminal]
        if len(nonterms) <= target_nonterm or len(nonterms) < 2:
            break

        best_pair = None
        best_dist = math.inf
        # O(r^2) search; acceptable for moderate graphs.
        for i in range(len(nonterms)):
            for j in range(i + 1, len(nonterms)):
                u, v = nonterms[i], nonterms[j]
                d = neighbor_set_distance(H, u, v)
                if d < best_dist:
                    best_dist = d
                    best_pair = (u, v)

        if best_pair is None:
            break

        a, b = best_pair
        new_id = ("cluster", merge_id)
        merge_id += 1
        merge_nonterminal_pair(H, a, b, new_id)

    return H


def lifted_deletion_from_coarsened_solution(H: nx.Graph, deleted_supernodes: Set) -> Set:
    deleted_original: Set = set()
    for v in deleted_supernodes:
        data: CoarsenNode = H.nodes[v]["data"]
        deleted_original |= set(data.members)
    return deleted_original


def coarsen_then_exact_highs(
    G: nx.Graph,
    terminals: Set,
    k: int,
    case: int = 1,
    target_nonterminal_ratio: float = 0.5,
    time_limit_exact_reduced: Optional[float] = None,
    mip_gap: Optional[float] = None,
    verbose: bool = False,
) -> HeuristicResult:
    t0 = time.perf_counter()
    H = coarsen_graph_by_neighbor_distance(
        G=G,
        terminals=terminals,
        target_nonterminal_ratio=target_nonterminal_ratio,
    )

    reduced_terminals = {v for v in H.nodes if H.nodes[v]["data"].is_terminal}
    deletion_cost = {v: H.nodes[v]["data"].cost for v in H.nodes}

    exact_reduced = solve_tcndp_exact_highs(
        G=H,
        terminals=reduced_terminals,
        k=k,
        case=case,
        deletion_cost=deletion_cost,
        time_limit=time_limit_exact_reduced,
        mip_gap=mip_gap,
        verbose=verbose,
    )

    deleted_original = lifted_deletion_from_coarsened_solution(H, exact_reduced.deleted)
    obj_original = terminal_pairwise_connectivity(G, terminals, deleted_original)
    runtime = time.perf_counter() - t0

    return HeuristicResult(
        name="coarsen+exact_highs",
        deleted=deleted_original,
        objective=float(obj_original),
        runtime_sec=runtime,
        details={
            "reduced_n": H.number_of_nodes(),
            "reduced_m": H.number_of_edges(),
            "reduced_nonterminals": sum(1 for v in H.nodes if not H.nodes[v]["data"].is_terminal),
            "reduced_exact_objective": exact_reduced.objective,
            "reduced_exact_runtime_sec": exact_reduced.runtime_sec,
            "reduced_exact_termination": exact_reduced.termination,
            "deleted_supernodes_cost": sum(H.nodes[v]["data"].cost for v in exact_reduced.deleted),
        },
    )


# ============================================================
# Experiment wrapper
# ============================================================


def run_comparison(
    G: nx.Graph,
    terminals: Iterable,
    k: int,
    case: int = 1,
    exact_time_limit: Optional[float] = None,
    reduced_exact_time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    target_nonterminal_ratio: float = 0.5,
    run_local_search: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    terminals = set(terminals)
    rows: List[Dict[str, object]] = []

    exact = solve_tcndp_exact_highs(
        G=G,
        terminals=terminals,
        k=k,
        case=case,
        time_limit=exact_time_limit,
        mip_gap=mip_gap,
        verbose=verbose,
    )
    rows.append(
        {
            "method": "exact_highs",
            "objective": exact.objective,
            "runtime_sec": exact.runtime_sec,
            "gap_to_exact": 0.0,
            "deleted_size": len(exact.deleted),
            "termination": exact.termination,
        }
    )

    g = greedy_constructive(G, terminals, k, case)
    rows.append(
        {
            "method": g.name,
            "objective": g.objective,
            "runtime_sec": g.runtime_sec,
            "gap_to_exact": g.objective - exact.objective,
            "deleted_size": len(g.deleted),
            "termination": "heuristic",
        }
    )

    if run_local_search:
        gls = greedy_local_search(G, terminals, g.deleted, case)
        rows.append(
            {
                "method": gls.name,
                "objective": gls.objective,
                "runtime_sec": g.runtime_sec + gls.runtime_sec,
                "gap_to_exact": gls.objective - exact.objective,
                "deleted_size": len(gls.deleted),
                "termination": "heuristic",
            }
        )

    cex = coarsen_then_exact_highs(
        G=G,
        terminals=terminals,
        k=k,
        case=case,
        target_nonterminal_ratio=target_nonterminal_ratio,
        time_limit_exact_reduced=reduced_exact_time_limit,
        mip_gap=mip_gap,
        verbose=verbose,
    )
    row = {
        "method": cex.name,
        "objective": cex.objective,
        "runtime_sec": cex.runtime_sec,
        "gap_to_exact": cex.objective - exact.objective,
        "deleted_size": len(cex.deleted),
        "termination": "heuristic",
    }
    row.update(cex.details)
    rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values(["objective", "runtime_sec"], ascending=[True, True]).reset_index(drop=True)


# ============================================================
# Example
# ============================================================


def demo() -> None:
    seed = 7
    n = 28
    p = 0.14
    terminal_ratio = 0.25
    k = 4
    case = 2

    G = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    if not nx.is_connected(G):
        # Keep only the largest connected component for a cleaner demo.
        cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(cc).copy()

    rng = __import__("random").Random(seed)
    nodes = list(G.nodes)
    terminals = set(rng.sample(nodes, max(2, round(len(nodes) * terminal_ratio))))

    df = run_comparison(
        G=G,
        terminals=terminals,
        k=k,
        case=case,
        exact_time_limit=300,
        reduced_exact_time_limit=120,
        mip_gap=0.0,
        target_nonterminal_ratio=0.5,
        run_local_search=True,
        verbose=False,
    )

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    print("Terminals:", sorted(terminals))
    print(df.to_string(index=False))


if __name__ == "__main__":
    demo()

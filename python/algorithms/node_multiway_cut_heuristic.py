
from __future__ import annotations

import itertools
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import lil_matrix


@dataclass
class SolveResult:
    method: str
    objective: int
    removed: List
    spent: float
    exact_optimum: Optional[int] = None
    solve_time_sec: Optional[float] = None
    status: Optional[str] = None


def _node_costs(
    G: nx.Graph,
    cost_attr: str = "cost",
    default_cost: float = 1.0,
) -> Dict:
    return {v: float(G.nodes[v].get(cost_attr, default_cost)) for v in G.nodes}


def connected_terminal_pairs_count(
    G: nx.Graph,
    terminals: Sequence,
    removed: Optional[Iterable] = None,
    allow_remove_terminals: bool = False,
) -> int:
    removed = set(removed or [])
    H = G.copy()
    H.remove_nodes_from(removed)

    active_terminals = [t for t in terminals if t in H]
    if len(active_terminals) < 2:
        return 0

    comp_id = {}
    for cid, comp in enumerate(nx.connected_components(H)):
        for v in comp:
            comp_id[v] = cid

    total = 0
    for i, s in enumerate(active_terminals):
        for t in active_terminals[i + 1 :]:
            if comp_id[s] == comp_id[t]:
                total += 1
    return total


def terminal_connectivity_pairs(
    G: nx.Graph,
    terminals: Sequence,
    removed: Optional[Iterable] = None,
) -> Dict[Tuple, int]:
    removed = set(removed or [])
    H = G.copy()
    H.remove_nodes_from(removed)

    active_terminals = [t for t in terminals if t in H]
    comp_id = {}
    for cid, comp in enumerate(nx.connected_components(H)):
        for v in comp:
            comp_id[v] = cid

    ans = {}
    for i, s in enumerate(terminals):
        for t in terminals[i + 1 :]:
            ans[(s, t)] = int(s in H and t in H and comp_id[s] == comp_id[t])
    return ans


def exact_ecndp_highs(
    G: nx.Graph,
    terminals: Sequence,
    budget: float,
    *,
    cost_attr: str = "cost",
    default_cost: float = 1.0,
    allow_remove_terminals: bool = False,
    time_limit: Optional[float] = None,
    mip_rel_gap: Optional[float] = None,
) -> SolveResult:
    """
    Exact ECNDP via SciPy's `milp`, solved by HiGHS.

    Assumed ECNDP objective:
        minimize the number of connected terminal pairs
        after deleting nodes with total deletion cost <= budget.
    """
    if G.is_directed():
        raise ValueError("This implementation assumes an undirected graph.")

    t0 = time.perf_counter()

    nodes = list(G.nodes)
    n = len(nodes)
    node_to_i = {v: i for i, v in enumerate(nodes)}
    terminals = list(terminals)
    costs = _node_costs(G, cost_attr=cost_attr, default_cost=default_cost)

    deletable = set(nodes if allow_remove_terminals else [v for v in nodes if v not in terminals])

    # Variable layout:
    # x_v: deletion binary, size n
    # y_ij: connectivity binary on ordered pairs, size n*n
    num_x = n
    num_y = n * n
    num_vars = num_x + num_y

    def x_idx(v):
        return node_to_i[v]

    def y_idx(u, v):
        return num_x + node_to_i[u] * n + node_to_i[v]

    c = np.zeros(num_vars, dtype=float)
    for i, s in enumerate(terminals):
        for t in terminals[i + 1 :]:
            c[y_idx(s, t)] = 1.0

    integrality = np.ones(num_vars, dtype=int)
    lb = np.zeros(num_vars, dtype=float)
    ub = np.ones(num_vars, dtype=float)

    constraints = []

    # Budget
    A = lil_matrix((1, num_vars), dtype=float)
    for v in deletable:
        A[0, x_idx(v)] = costs[v]
    constraints.append(LinearConstraint(A.tocsr(), -np.inf, np.array([budget], dtype=float)))

    # Fix non-deletable nodes
    fixed_non_deletable = [v for v in nodes if v not in deletable]
    if fixed_non_deletable:
        A = lil_matrix((len(fixed_non_deletable), num_vars), dtype=float)
        for r, v in enumerate(fixed_non_deletable):
            A[r, x_idx(v)] = 1.0
        z = np.zeros(len(fixed_non_deletable), dtype=float)
        constraints.append(LinearConstraint(A.tocsr(), z, z))

    # Diagonal y_vv = 1 - x_v
    A = lil_matrix((n, num_vars), dtype=float)
    for r, v in enumerate(nodes):
        A[r, y_idx(v, v)] = 1.0
        A[r, x_idx(v)] = 1.0
    ones = np.ones(n, dtype=float)
    constraints.append(LinearConstraint(A.tocsr(), ones, ones))

    # Symmetry y_uv = y_vu
    sym_pairs = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :]]
    if sym_pairs:
        A = lil_matrix((len(sym_pairs), num_vars), dtype=float)
        for r, (u, v) in enumerate(sym_pairs):
            A[r, y_idx(u, v)] = 1.0
            A[r, y_idx(v, u)] = -1.0
        z = np.zeros(len(sym_pairs), dtype=float)
        constraints.append(LinearConstraint(A.tocsr(), z, z))

    # Upper bounds y_uv <= 1 - x_u and y_uv <= 1 - x_v for u != v
    ordered_distinct = [(u, v) for u in nodes for v in nodes if u != v]
    A = lil_matrix((2 * len(ordered_distinct), num_vars), dtype=float)
    rhs = np.ones(2 * len(ordered_distinct), dtype=float)
    for r, (u, v) in enumerate(ordered_distinct):
        A[r, y_idx(u, v)] = 1.0
        A[r, x_idx(u)] = 1.0
        A[len(ordered_distinct) + r, y_idx(u, v)] = 1.0
        A[len(ordered_distinct) + r, x_idx(v)] = 1.0
    constraints.append(LinearConstraint(A.tocsr(), -np.inf, rhs))

    # Edge lower bounds:
    # y_uv + x_u + x_v >= 1 and y_vu + x_u + x_v >= 1
    edge_rows = []
    for u, v in G.edges():
        edge_rows.append((u, v))
        edge_rows.append((v, u))
    A = lil_matrix((len(edge_rows), num_vars), dtype=float)
    lb_edge = np.ones(len(edge_rows), dtype=float)
    ub_edge = np.full(len(edge_rows), np.inf, dtype=float)
    for r, (u, v) in enumerate(edge_rows):
        A[r, y_idx(u, v)] = 1.0
        A[r, x_idx(u)] = 1.0
        A[r, x_idx(v)] = 1.0
    constraints.append(LinearConstraint(A.tocsr(), lb_edge, ub_edge))

    # Transitivity:
    # y_uv - y_uk - y_kv >= -1
    triples = [(u, v, k) for u in nodes for v in nodes for k in nodes if len({u, v, k}) == 3 and u != v]
    A = lil_matrix((len(triples), num_vars), dtype=float)
    lb_tr = -np.ones(len(triples), dtype=float)
    ub_tr = np.full(len(triples), np.inf, dtype=float)
    for r, (u, v, k) in enumerate(triples):
        A[r, y_idx(u, v)] = 1.0
        A[r, y_idx(u, k)] = -1.0
        A[r, y_idx(k, v)] = -1.0
    constraints.append(LinearConstraint(A.tocsr(), lb_tr, ub_tr))

    options = {}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    if mip_rel_gap is not None:
        options["mip_rel_gap"] = float(mip_rel_gap)

    res = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        options=options,
    )

    if res.x is None:
        raise RuntimeError(f"HiGHS failed on exact ECNDP model: {res.message}")

    x = res.x[:num_x]
    removed = [v for v in nodes if x[x_idx(v)] >= 0.5]
    spent = sum(costs[v] for v in removed)
    obj = int(round(res.fun))
    elapsed = time.perf_counter() - t0

    return SolveResult(
        method="exact_highs",
        objective=obj,
        removed=removed,
        spent=spent,
        solve_time_sec=elapsed,
        status=res.message,
    )


def removal_greedy_recompute(
    G: nx.Graph,
    terminals: Sequence,
    budget: float,
    *,
    cost_attr: str = "cost",
    default_cost: float = 1.0,
    allow_remove_terminals: bool = False,
) -> SolveResult:
    costs = _node_costs(G, cost_attr=cost_attr, default_cost=default_cost)
    removed: Set = set()
    spent = 0.0
    candidates = list(G.nodes if allow_remove_terminals else [v for v in G.nodes if v not in terminals])

    while True:
        current_obj = connected_terminal_pairs_count(G, terminals, removed, allow_remove_terminals)
        best_v = None
        best_key = None

        for v in candidates:
            if v in removed or spent + costs[v] > budget:
                continue
            trial_obj = connected_terminal_pairs_count(G, terminals, removed | {v}, allow_remove_terminals)
            gain = current_obj - trial_obj
            key = (gain / costs[v], gain, -costs[v], str(v))
            if best_key is None or key > best_key:
                best_key = key
                best_v = v

        if best_v is None:
            break

        removed.add(best_v)
        spent += costs[best_v]
        if spent >= budget - 1e-9:
            break

    return SolveResult(
        method="greedy_recompute",
        objective=connected_terminal_pairs_count(G, terminals, removed, allow_remove_terminals),
        removed=sorted(removed, key=str),
        spent=spent,
    )


def removal_top_degree(
    G: nx.Graph,
    terminals: Sequence,
    budget: float,
    *,
    cost_attr: str = "cost",
    default_cost: float = 1.0,
    allow_remove_terminals: bool = False,
) -> SolveResult:
    costs = _node_costs(G, cost_attr=cost_attr, default_cost=default_cost)
    removed: Set = set()
    spent = 0.0

    while True:
        H = G.copy()
        H.remove_nodes_from(removed)

        candidates = [v for v in H.nodes if allow_remove_terminals or v not in terminals]
        candidates = [v for v in candidates if spent + costs[v] <= budget]
        if not candidates:
            break

        candidates.sort(key=lambda v: (H.degree[v] / costs[v], H.degree[v], -costs[v], str(v)), reverse=True)
        chosen = candidates[0]
        removed.add(chosen)
        spent += costs[chosen]

    return SolveResult(
        method="top_degree",
        objective=connected_terminal_pairs_count(G, terminals, removed, allow_remove_terminals),
        removed=sorted(removed, key=str),
        spent=spent,
    )


def removal_top_betweenness(
    G: nx.Graph,
    terminals: Sequence,
    budget: float,
    *,
    cost_attr: str = "cost",
    default_cost: float = 1.0,
    allow_remove_terminals: bool = False,
) -> SolveResult:
    costs = _node_costs(G, cost_attr=cost_attr, default_cost=default_cost)
    removed: Set = set()
    spent = 0.0

    while True:
        H = G.copy()
        H.remove_nodes_from(removed)

        candidates = [v for v in H.nodes if allow_remove_terminals or v not in terminals]
        candidates = [v for v in candidates if spent + costs[v] <= budget]
        if not candidates:
            break

        bc = nx.betweenness_centrality(H, normalized=True)
        candidates.sort(key=lambda v: (bc[v] / costs[v], bc[v], -costs[v], str(v)), reverse=True)
        chosen = candidates[0]
        removed.add(chosen)
        spent += costs[chosen]

    return SolveResult(
        method="top_betweenness",
        objective=connected_terminal_pairs_count(G, terminals, removed, allow_remove_terminals),
        removed=sorted(removed, key=str),
        spent=spent,
    )


def removal_random(
    G: nx.Graph,
    terminals: Sequence,
    budget: float,
    *,
    seed: int = 0,
    cost_attr: str = "cost",
    default_cost: float = 1.0,
    allow_remove_terminals: bool = False,
) -> SolveResult:
    rng = random.Random(seed)
    costs = _node_costs(G, cost_attr=cost_attr, default_cost=default_cost)
    candidates = list(G.nodes if allow_remove_terminals else [v for v in G.nodes if v not in terminals])
    rng.shuffle(candidates)

    removed = []
    spent = 0.0
    for v in candidates:
        if spent + costs[v] <= budget:
            removed.append(v)
            spent += costs[v]

    return SolveResult(
        method=f"random_seed_{seed}",
        objective=connected_terminal_pairs_count(G, terminals, removed, allow_remove_terminals),
        removed=sorted(removed, key=str),
        spent=spent,
    )


def _solve_multiway_dual_lp_highs(
    H: nx.Graph,
    terminals: Sequence,
    costs: Dict,
) -> Tuple[Dict, Dict[Tuple, float], float]:
    """
    Solve the paper's polynomial-size dual LP P3 with HiGHS via scipy.linprog.

    Variables:
        d_v  for each nonterminal v
        y_(u,s) distance from terminal s to node u
    """
    terminals = list(terminals)
    nodes = list(H.nodes)
    nonterminals = [v for v in nodes if v not in terminals]

    d_index = {v: i for i, v in enumerate(nonterminals)}
    offset = len(nonterminals)
    y_index = {(u, s): offset + r for r, (u, s) in enumerate(itertools.product(nodes, terminals))}
    m = offset + len(y_index)

    c = np.zeros(m, dtype=float)
    for v, i in d_index.items():
        c[i] = costs[v]

    lb = np.zeros(m, dtype=float)
    ub = np.full(m, np.inf, dtype=float)
    for v, i in d_index.items():
        ub[i] = 1.0

    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []

    def row():
        return np.zeros(m, dtype=float)

    # Distance propagation on both directions of each undirected edge
    for s in terminals:
        for u, v in H.edges():
            r1 = row()
            r1[y_index[(v, s)]] = 1.0
            r1[y_index[(u, s)]] = -1.0
            if v in d_index:
                r1[d_index[v]] = -1.0
            A_ub.append(r1)
            b_ub.append(0.0)

            r2 = row()
            r2[y_index[(u, s)]] = 1.0
            r2[y_index[(v, s)]] = -1.0
            if u in d_index:
                r2[d_index[u]] = -1.0
            A_ub.append(r2)
            b_ub.append(0.0)

    # y[s,s] = 0
    for s in terminals:
        r = row()
        r[y_index[(s, s)]] = 1.0
        A_eq.append(r)
        b_eq.append(0.0)

    # y[t,s] >= 1 for t != s  ->  -y[t,s] <= -1
    for s in terminals:
        for t in terminals:
            if t == s:
                continue
            r = row()
            r[y_index[(t, s)]] = -1.0
            A_ub.append(r)
            b_ub.append(-1.0)

    res = linprog(
        c=c,
        A_ub=np.array(A_ub, dtype=float) if A_ub else None,
        b_ub=np.array(b_ub, dtype=float) if b_ub else None,
        A_eq=np.array(A_eq, dtype=float) if A_eq else None,
        b_eq=np.array(b_eq, dtype=float) if b_eq else None,
        bounds=list(zip(lb, ub)),
        method="highs",
    )

    if not res.success:
        raise RuntimeError(f"HiGHS failed on dual LP: {res.message}")

    x = res.x
    d = {v: (x[d_index[v]] if v in d_index else 0.0) for v in nodes}
    y = {(u, s): x[y_index[(u, s)]] for u in nodes for s in terminals}
    return d, y, float(res.fun)


def lp_multiway_budget_heuristic(
    G: nx.Graph,
    terminals: Sequence,
    budget: float,
    *,
    cost_attr: str = "cost",
    default_cost: float = 1.0,
    allow_remove_terminals: bool = False,
    tol: float = 1e-7,
) -> SolveResult:
    r"""
    Budgeted heuristic inspired by Garg-Vazirani-Yannakakis.

    Round-based adaptation:
      1. Solve the dual LP.
      2. Compute zero-distance regions S_i and boundaries Γ(S_i).
      3. Form the paper's rounded cut:
             C = \bigcup_i Γ(S_i) \setminus Γ^{1/2}(S_{j*})
      4. Because ECNDP has a budget, delete affordable nodes from C in the order:
             larger d_v, larger boundary multiplicity, lower cost
      5. Recompute on the residual graph and repeat.

    This is a heuristic adaptation for ECNDP; the paper's approximation guarantee
    does not directly carry over to the budgeted objective.
    """
    costs = _node_costs(G, cost_attr=cost_attr, default_cost=default_cost)
    removed: Set = set()
    spent = 0.0

    while spent < budget - 1e-9:
        H = G.copy()
        H.remove_nodes_from(removed)
        active_terminals = [t for t in terminals if t in H]

        if len(active_terminals) < 2:
            break
        if connected_terminal_pairs_count(G, terminals, removed, allow_remove_terminals) == 0:
            break

        # If two terminals are adjacent, the node-multiway-cut LP form is not the right one.
        # Fallback to a single greedy step.
        if any(H.has_edge(s, t) for i, s in enumerate(active_terminals) for t in active_terminals[i + 1 :]):
            best_v = None
            best_key = None
            current_obj = connected_terminal_pairs_count(G, terminals, removed, allow_remove_terminals)
            for v in H.nodes:
                if (not allow_remove_terminals and v in terminals) or spent + costs[v] > budget:
                    continue
                trial_obj = connected_terminal_pairs_count(G, terminals, removed | {v}, allow_remove_terminals)
                gain = current_obj - trial_obj
                key = (gain / costs[v], gain, -costs[v], str(v))
                if best_key is None or key > best_key:
                    best_key = key
                    best_v = v
            if best_v is None:
                break
            removed.add(best_v)
            spent += costs[best_v]
            continue

        d, y, _ = _solve_multiway_dual_lp_highs(H, active_terminals, costs)

        regions = {s: {u for u in H.nodes if y[(u, s)] <= tol} for s in active_terminals}

        boundaries = {}
        for s, S in regions.items():
            B = set()
            for u in S:
                for v in H.neighbors(u):
                    if v not in S:
                        B.add(v)
            boundaries[s] = B

        M = set().union(*boundaries.values()) if boundaries else set()
        if not M:
            break

        membership = {v: 0 for v in M}
        for s in active_terminals:
            for v in boundaries[s]:
                membership[v] += 1

        M1 = {v for v in M if membership[v] >= 2}
        Mhalf = M - M1
        gamma_half = {s: boundaries[s] & Mhalf for s in active_terminals}

        j_star = max(active_terminals, key=lambda s: sum(costs[v] for v in gamma_half[s]))
        candidate_cut = M - gamma_half[j_star]
        candidate_cut = {
            v for v in candidate_cut
            if v not in removed and (allow_remove_terminals or v not in terminals)
        }

        if not candidate_cut:
            break

        ordered = sorted(
            candidate_cut,
            key=lambda v: (d.get(v, 0.0), membership.get(v, 0), -costs[v], str(v)),
            reverse=True,
        )

        changed = False
        for v in ordered:
            if spent + costs[v] <= budget:
                removed.add(v)
                spent += costs[v]
                changed = True

        if not changed:
            break

    return SolveResult(
        method="lp_multiway_budget",
        objective=connected_terminal_pairs_count(G, terminals, removed, allow_remove_terminals),
        removed=sorted(removed, key=str),
        spent=spent,
    )


def compare_ecndp_methods(
    G: nx.Graph,
    terminals: Sequence,
    budget: float,
    *,
    cost_attr: str = "cost",
    default_cost: float = 1.0,
    allow_remove_terminals: bool = False,
    random_seed: int = 0,
    include_exact: bool = True,
) -> List[SolveResult]:
    results: List[SolveResult] = []

    exact_result = None
    if include_exact:
        exact_result = exact_ecndp_highs(
            G,
            terminals,
            budget,
            cost_attr=cost_attr,
            default_cost=default_cost,
            allow_remove_terminals=allow_remove_terminals,
        )
        results.append(exact_result)

    heuristics = [
        removal_greedy_recompute(
            G, terminals, budget,
            cost_attr=cost_attr,
            default_cost=default_cost,
            allow_remove_terminals=allow_remove_terminals,
        ),
        removal_top_degree(
            G, terminals, budget,
            cost_attr=cost_attr,
            default_cost=default_cost,
            allow_remove_terminals=allow_remove_terminals,
        ),
        removal_top_betweenness(
            G, terminals, budget,
            cost_attr=cost_attr,
            default_cost=default_cost,
            allow_remove_terminals=allow_remove_terminals,
        ),
        removal_random(
            G, terminals, budget,
            seed=random_seed,
            cost_attr=cost_attr,
            default_cost=default_cost,
            allow_remove_terminals=allow_remove_terminals,
        ),
        lp_multiway_budget_heuristic(
            G, terminals, budget,
            cost_attr=cost_attr,
            default_cost=default_cost,
            allow_remove_terminals=allow_remove_terminals,
        ),
    ]
    results.extend(heuristics)

    if exact_result is not None:
        for r in results:
            r.exact_optimum = exact_result.objective

    return results


def print_results_table(results: Sequence[SolveResult]) -> None:
    exact_obj = next((r.objective for r in results if r.method == "exact_highs"), None)
    header = f"{'method':24s} {'obj':>5s} {'spent':>8s} {'gap_vs_exact':>12s} removed"
    print(header)
    print("-" * len(header))
    for r in results:
        gap = "n/a" if exact_obj is None else f"{r.objective - exact_obj:+d}"
        print(f"{r.method:24s} {r.objective:5d} {r.spent:8.2f} {gap:>12s} {r.removed}")


if __name__ == "__main__":
    G = nx.Graph()
    G.add_edges_from([
        ("t1", "a"), ("a", "b"), ("b", "t2"),
        ("a", "c"), ("c", "d"), ("d", "t3"),
        ("b", "e"), ("e", "f"), ("f", "t3"),
        ("c", "g"), ("g", "t2"),
    ])
    for v in G.nodes:
        G.nodes[v]["cost"] = 1.0

    terminals = ["t1", "t2", "t3"]
    budget = 2

    results = compare_ecndp_methods(G, terminals, budget)
    print_results_table(results)

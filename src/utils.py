from itertools import combinations
from pathlib import Path

import numpy as np
from hypergraphx import Hypergraph


def hyperedge_homogeneity_list(H):
    """Return homogeneity score per hyperedge based on pair co-occurrence."""
    edges = [list(e) for e in H.get_edges()]

    pair_degree = {}
    for edge in edges:
        for u, v in combinations(sorted(edge), 2):
            pair_degree[(u, v)] = pair_degree.get((u, v), 0) + 1

    homogeneity = []
    for edge in edges:
        if len(edge) <= 1:
            homogeneity.append(0.0)
            continue

        pair_values = []
        for u, v in combinations(sorted(edge), 2):
            pair_values.append(pair_degree[(u, v)])

        homogeneity.append(sum(pair_values) / len(pair_values))

    return homogeneity


def hypergraph_disjointness(H):
    """
    Fraction of hyperedge pairs that overlap in more than one node.
    Higher values indicate stronger disjointness violations.
    """
    edges = [set(e) for e in H.get_edges()]

    overlapping = 0
    total = 0
    for i, e1 in enumerate(edges):
        for j, e2 in enumerate(edges):
            if i >= j:
                continue
            total += 1
            if len(e1 & e2) > 1:
                overlapping += 1
                
    return overlapping / total


def load_txt(path: str | Path) -> Hypergraph:
    edges = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            edges.append([int(x) for x in line.replace(",", " ").split()])
    return Hypergraph(edges)


def analyze_transition_cases_with_selection(
    H: Hypergraph,
    *,
    walk_length: int = 40,
    num_walks: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    r: float = 1.0,
    gamma: float = -1.0,
    delta: float = 0.0,
    seed: int = 0,
) -> dict:
    """Return selection and availability case counts for one (p, q) run."""
    nodes = sorted(H.get_nodes())
    edges = [frozenset(e) for e in H.get_edges()]
    n = len(nodes)

    idx = {u: i for i, u in enumerate(nodes)}
    inc_edges = [[] for _ in range(n)]
    d = np.zeros(n, dtype=float)

    for e_i, e in enumerate(edges):
        for u in e:
            i = idx[u]
            inc_edges[i].append(e_i)
            d[i] += 1.0

    selected_counts = {"first": 0, "middle": 0, "third": 0}
    available_counts = {"first": 0, "middle": 0, "third": 0}

    rng = np.random.default_rng(seed)

    for _ in range(num_walks):
        for start in rng.permutation(n):
            v_i = int(start)
            prev_u_i = None
            prev_e_i = None

            for _step in range(walk_length - 1):
                cand_e = inc_edges[v_i]
                if not cand_e:
                    continue

                def classify_case(e2_i: int) -> str:
                    if prev_e_i is None or prev_u_i is None:
                        return "middle"
                    if e2_i == prev_e_i:
                        return "first"
                    v_node = nodes[v_i]
                    if len((edges[prev_e_i] - {v_node}) & (edges[e2_i] - {v_node})) > 0:
                        return "middle"
                    return "third"

                for e2_i in cand_e:
                    available_counts[classify_case(e2_i)] += 1

                raw_e = np.zeros(len(cand_e), dtype=float)
                for k, e2_i in enumerate(cand_e):
                    e2 = edges[e2_i]
                    if prev_e_i is None or prev_u_i is None:
                        a_edge = 1.0
                    elif e2_i == prev_e_i:
                        a_edge = 1.0 / p
                    elif len((edges[prev_e_i] - {nodes[v_i]}) & (e2 - {nodes[v_i]})) > 0:
                        a_edge = 1.0
                    else:
                        a_edge = 1.0 / q
                    size_term = max(len(e2) - 1, 1)
                    raw_e[k] = a_edge * (size_term ** gamma)

                s_e = raw_e.sum()
                if s_e <= 0 or not np.isfinite(s_e):
                    next_e_i = int(rng.choice(cand_e))
                else:
                    next_e_i = cand_e[int(rng.choice(len(cand_e), p=raw_e / s_e))]

                selected_counts[classify_case(next_e_i)] += 1

                e2 = edges[next_e_i]
                cand_x = [idx[x] for x in e2 if idx[x] != v_i]
                if not cand_x:
                    continue

                raw_x = np.zeros(len(cand_x), dtype=float)
                prev_e_nodes = edges[prev_e_i] if prev_e_i is not None else None
                for k, x_i in enumerate(cand_x):
                    if prev_u_i is None:
                        a_node = 1.0
                    elif prev_e_nodes is not None and nodes[x_i] in prev_e_nodes:
                        a_node = 1.0
                    else:
                        a_node = 1.0 / r
                    raw_x[k] = a_node * (max(float(d[x_i]), 1.0) ** delta)

                s_x = raw_x.sum()
                if s_x <= 0 or not np.isfinite(s_x):
                    x_i = int(rng.choice(cand_x))
                else:
                    x_i = cand_x[int(rng.choice(len(cand_x), p=raw_x / s_x))]

                prev_u_i = v_i
                prev_e_i = next_e_i
                v_i = x_i

    return {
        "selected": selected_counts,
        "available": available_counts,
    }


def theoretical_case_values(H: Hypergraph) -> np.ndarray:
    degsum = 0.0
    for deg, cnt in H.degree_distribution().items():
        degsum += deg * cnt
    avgdeg = degsum / H.num_nodes()

    avghedgesize = float(np.mean(H.get_sizes()))
    hom = float(np.mean(hyperedge_homogeneity_list(H)))

    same = 1.0 / avgdeg
    middle = (avghedgesize - 1.0) * (hom - 1.0) / avgdeg
    other = 1.0 - same - middle
    return np.array([same, middle, other], dtype=float)


def empirical_availability_values(
    H: Hypergraph,
    seed: int = 42,
    *,
    walk_length: int = 20,
    num_walks: int = 50,
    p: float = 1.0,
    q: float = 1.0,
    r: float = 1.0,
    gamma: float = 0.0,
    delta: float = 0.0,
) -> np.ndarray:
    out = analyze_transition_cases_with_selection(
        H,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        r=r,
        gamma=gamma,
        delta=delta,
        seed=seed,
    )
    labels = ["first", "middle", "third"]
    av = np.array([out["available"][k] for k in labels], dtype=float)
    return av / max(av.sum(), 1.0)

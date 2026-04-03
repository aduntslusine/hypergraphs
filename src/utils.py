"""Hyper2Vec, clique+node2vec, and HMRW embeddings via random walks + Skip-gram."""

from __future__ import annotations

import warnings

import numpy as np
from gensim.models import Word2Vec
from hypergraphx import Hypergraph
from hypergraphx.representations.projections import clique_projection


def hyper2vec_embeddings(
    hg: Hypergraph,
    *,
    p: float = 1.0,
    q: float = 1.0,
    r: float = 0.0,
    num_walks: int = 10,
    walk_length: int = 80,
    window: int = 10,
    dim: int = 128,
    seed: int | None = None,
    workers: int = 1,
) -> np.ndarray:
    """
    Hyper2Vec node embeddings: transition from Eqs. (1)–(4) in the paper, then Skip-gram.
    Rows follow ``sorted(hg.get_nodes())``. Unweighted edges use w(e)=1; weighted graphs
    need ``hg.weighted`` and ``hg.get_weights()`` aligned with ``hg.get_edges()``.
    ``num_walks`` is the number of walks **per node**.
    """
    # --- Nodes in a fixed order (row i of the output = nodes[i]) -----------------
    nodes = sorted(hg.get_nodes())
    n = len(nodes)
    if n == 0:
        return np.zeros((0, dim), dtype=float)

    idx = {u: i for i, u in enumerate(nodes)}  # hypergraph node id -> row index 0..n-1

    # --- Hyperedges and weights w(e); unweighted => all weights 1 ----------------
    edges = [frozenset(e) for e in hg.get_edges()]
    w_e = np.ones(len(edges), dtype=float)
    if getattr(hg, "weighted", False) and hasattr(hg, "get_weights"):
        w_e = np.asarray(hg.get_weights(), dtype=float)

    # --- Weighted vertex degree d(v) = sum of w(e) over hyperedges incident to v (paper §2)
    d = np.zeros(n, dtype=float)
    inc_edges = [[] for _ in range(n)]  # incident edge ids for each node index
    for e_i, (e, w) in enumerate(zip(edges, w_e)):
        for u in e:
            i = idx[u]
            d[i] += float(w)
            inc_edges[i].append(e_i)

    # --- For 2nd-order factor α: "same hyperedge as u" = co-membership with previous node
    #     (Eq. 2: cheap to test via co[u] rather than scanning edges each time)
    co = [set() for _ in range(n)]
    for e in edges:
        ids = [idx[u] for u in e]
        for i in ids:
            co[i].update(ids)

    # --- π1 support and values by current node v:
    #     only nodes sharing at least one hyperedge with v can have non-zero π1.
    #     cache each row once to avoid scanning all nodes every step.
    pi1_cache = [dict() for _ in range(n)]  # v_i -> {x_i: pi1(x|v)}
    for v_i in range(n):
        vd = d[v_i]
        if vd <= 0:
            continue
        accum: dict[int, float] = {}
        for e_i in inc_edges[v_i]:
            e = edges[e_i]
            w = float(w_e[e_i])
            contrib = w / len(e)
            for x in e:
                x_i = idx[x]
                accum[x_i] = accum.get(x_i, 0.0) + contrib
        pi1_cache[v_i] = {x_i: val / vd for x_i, val in accum.items() if val > 0.0}

    rng = np.random.default_rng(seed)
    walks: list[list[int]] = []

    # --- Random walks: each step samples next vertex x from unnormalized p′₂(x|v,u) (Eq. 4)
    #     num_walks means walks per node.
    for _ in range(num_walks):
        for start in rng.permutation(n):
            walk = [int(start)]
            for _ in range(walk_length - 1):
                # u = previous vertex, v = current; first step has no u → only 1st-order + degree bias
                u_i = walk[-2] if len(walk) > 1 else None
                v_i = walk[-1]
                vd = d[v_i]
                raw = np.zeros(n, dtype=float)

                # Restrict candidates to co-membership neighbors of current node.
                for x_i in co[v_i]:
                    # π₁(x|v): pick incident hyperedge (weight w), then uniform endpoint on that edge (Eq. 1).
                    # Here we sum over hyperedges e containing both v and x: (w(e)/d(v))·(1/|e|).
                    pi1 = pi1_cache[v_i].get(x_i, 0.0)
                    if pi1 <= 0:
                        continue

                    # α(x|v,u): node2vec-style return / explore (Eq. 2); skipped when there is no u
                    if u_i is None:
                        a = 1.0
                    elif x_i == u_i:
                        a = 1.0 / p  # go back to previous
                    elif x_i in co[u_i]:
                        a = 1.0  # stay "local" to u (share a hyperedge with u)
                    else:
                        a = 1.0 / q  # push outward

                    # β(x): degree bias — BDBS if r>0, SDBS if r<0, off if r=0 (Eq. 3)
                    if r > 0:
                        b = float(d[x_i]) + r
                    elif r < 0:
                        b = 1.0 / (float(d[x_i]) - r)
                    else:
                        b = 1.0

                    # Eq. 4: unnormalized weight ∝ α · β · (1st-order term); Z is the sum over x
                    raw[x_i] = a * b * pi1

                s = raw.sum()
                if s <= 0 or not np.isfinite(s):
                    walk.append(v_i)  # nowhere to go → stay put
                else:
                    walk.append(int(rng.choice(n, p=raw / s)))  # normalize raw to get p′₂
            walks.append(walk)

    # --- Skip-gram on walks: words = row indices as strings so row i matches nodes[i] ---
    w2v = Word2Vec(
        sentences=[[str(i) for i in w] for w in walks],
        vector_size=dim,
        window=window,
        min_count=1,
        sg=1,  # skip-gram (not CBOW)
        workers=workers,
        seed=seed if seed is not None else 0,
        epochs=5,
    )

    # --- Pack vectors in node order (i-th row ↔ nodes[i])
    out = np.zeros((n, dim), dtype=float)
    for i in range(n):
        k = str(i)
        if k in w2v.wv:
            out[i] = w2v.wv[k]
    return out


def clique_node2vec_embeddings(
    hg: Hypergraph,
    *,
    p: float = 1.0,
    q: float = 1.0,
    num_walks: int = 10,
    walk_length: int = 80,
    window: int = 10,
    dim: int = 128,
    seed: int | None = None,
    workers: int = 1,
) -> np.ndarray:
    """
    Clique-expand the hypergraph (each hyperedge becomes a clique), then biased 2nd-order
    walks on that *graph* (Grover–Leskovec node2vec-style α with *p*, *q* on graph
    neighbors of the previous vertex; optional same β(*r*) as Hyper2Vec but using **graph**
    degrees). Skip-gram training matches ``hyper2vec_embeddings``. Row order:
    ``sorted(hg.get_nodes())``. ``num_walks`` is the number of walks **per node**.
    """
    nodes = sorted(hg.get_nodes())
    n = len(nodes)
    if n == 0:
        return np.zeros((0, dim), dtype=float)

    idx = {u: i for i, u in enumerate(nodes)}
    G = clique_projection(hg)

    # Neighbors in the projected simple graph (indices 0..n-1)
    nbr = [set() for _ in range(n)]
    for u, v in G.edges():
        if u in idx and v in idx:
            iu, iv = idx[u], idx[v]
            nbr[iu].add(iv)
            nbr[iv].add(iu)

    d = np.array([len(nbr[i]) for i in range(n)], dtype=float)

    rng = np.random.default_rng(seed)
    walks: list[list[int]] = []

    for _ in range(num_walks):
        for start in rng.permutation(n):
            walk = [int(start)]
            for _ in range(walk_length - 1):
                u_i = walk[-2] if len(walk) > 1 else None
                v_i = walk[-1]
                dv = d[v_i]
                raw = np.zeros(n, dtype=float)

                for x_i in range(n):
                    # Simple random walk on the graph: uniform over neighbors of v (no edge => 0)
                    if dv <= 0 or x_i not in nbr[v_i]:
                        pi1 = 0.0
                    else:
                        pi1 = 1.0 / dv

                    if pi1 <= 0:
                        continue

                    # node2vec α: return to u / stay near u in the *clique graph* / outward
                    if u_i is None:
                        a = 1.0
                    elif x_i == u_i:
                        a = 1.0 / p
                    elif x_i in nbr[u_i]:
                        a = 1.0
                    else:
                        a = 1.0 / q

                    raw[x_i] = a * pi1

                s = raw.sum()
                if s <= 0 or not np.isfinite(s):
                    walk.append(v_i)
                else:
                    walk.append(int(rng.choice(n, p=raw / s)))
            walks.append(walk)

    w2v = Word2Vec(
        sentences=[[str(i) for i in w] for w in walks],
        vector_size=dim,
        window=window,
        min_count=1,
        sg=1,
        workers=workers,
        seed=seed if seed is not None else 0,
        epochs=5,
    )

    out = np.zeros((n, dim), dtype=float)
    for i in range(n):
        k = str(i)
        if k in w2v.wv:
            out[i] = w2v.wv[k]
    return out


def hmrw_embeddings(
    hg: Hypergraph,
    *,
    p: float = 1.0,
    q: float = 1.0,
    r: float = 1.0,
    num_walks: int = 10,
    walk_length: int = 80,
    window: int = 10,
    dim: int = 128,
    seed: int | None = None,
    workers: int = 1,
) -> np.ndarray:
    """
    Hyperedge-Memory Random Walk (HMRW), then Skip-gram.

    Based on your method in `hyperedge_aware_rw.pdf`:
    - Keep state (u, v, e): previous node u, current node v, previous hyperedge e.
    - Stage 1: sample next hyperedge e' incident to v with
      pi(e'|e,u,v) ∝ a_edge(e',e,u) * h(v,e') / |e'|.
    - Stage 2: sample next node x in e' with
      pi(x|e',v,e,u) ∝ a_node(x,e,u) * h(x,e') / d(x).
    ``num_walks`` is the number of walks **per node**.
    """
    nodes = sorted(hg.get_nodes())
    n = len(nodes)
    if n == 0:
        return np.zeros((0, dim), dtype=float)

    edges = [frozenset(e) for e in hg.get_edges()]
    m = len(edges)
    if m == 0:
        return np.zeros((n, dim), dtype=float)

    idx = {u: i for i, u in enumerate(nodes)}

    # Node degree d(x) = number of incident hyperedges.
    d = np.zeros(n, dtype=float)
    inc_edges = [[] for _ in range(n)]  # incident edge ids for each node index
    for e_i, e in enumerate(edges):
        for u in e:
            i = idx[u]
            d[i] += 1.0
            inc_edges[i].append(e_i)

    rng = np.random.default_rng(seed)
    walks: list[list[int]] = []
    isolated_step_count = 0

    for _ in range(num_walks):
        for start in rng.permutation(n):
            v_i = int(start)
            walk = [v_i]
            prev_u_i: int | None = None
            prev_e_i: int | None = None

            for _ in range(walk_length - 1):
                # --- Stage 1: choose next hyperedge e' incident to current node v.
                cand_e = inc_edges[v_i]
                if not cand_e:
                    isolated_step_count += 1
                    walk.append(v_i)
                    continue

                raw_e = np.zeros(len(cand_e), dtype=float)
                for k, e2_i in enumerate(cand_e):
                    e2 = edges[e2_i]
                    if len(e2) == 0:
                        continue
                    if prev_e_i is None or prev_u_i is None:
                        a_edge = 1.0
                    elif e2_i == prev_e_i:
                        a_edge = 1.0 / p
                    elif nodes[prev_u_i] in e2:
                        a_edge = 1.0
                    else:
                        a_edge = 1.0 / q
                    raw_e[k] = a_edge / len(e2)  # h(v,e')=1 since e' incident to v

                s_e = raw_e.sum()
                if s_e <= 0 or not np.isfinite(s_e):
                    next_e_i = int(rng.choice(cand_e))
                else:
                    next_e_i = cand_e[int(rng.choice(len(cand_e), p=raw_e / s_e))]

                # --- Stage 2: choose next node x in e'.
                e2 = edges[next_e_i]
                cand_x = [idx[x] for x in e2]
                raw_x = np.zeros(len(cand_x), dtype=float)
                prev_e_nodes = edges[prev_e_i] if prev_e_i is not None else None

                for k, x_i in enumerate(cand_x):
                    if d[x_i] <= 0:
                        continue
                    if prev_u_i is None:
                        a_node = 1.0
                    elif prev_e_nodes is not None and nodes[x_i] in prev_e_nodes:
                        a_node = 1.0
                    else:
                        a_node = 1.0 / r
                    raw_x[k] = a_node * (1.0 / d[x_i])  # h(x,e')=1 for x in e'

                s_x = raw_x.sum()
                if s_x <= 0 or not np.isfinite(s_x):
                    x_i = v_i
                else:
                    x_i = cand_x[int(rng.choice(len(cand_x), p=raw_x / s_x))]

                # Advance walk memory: (u,v,e) <- (v,x,e').
                prev_u_i = v_i
                prev_e_i = next_e_i
                v_i = x_i
                walk.append(v_i)

            walks.append(walk)

    if isolated_step_count > 0:
        warnings.warn(
            f"HMRW encountered {isolated_step_count} step(s) at nodes with no incident hyperedges; "
            "the walk stayed on the same node for those steps.",
            RuntimeWarning,
            stacklevel=2,
        )

    w2v = Word2Vec(
        sentences=[[str(i) for i in w] for w in walks],
        vector_size=dim,
        window=window,
        min_count=1,
        sg=1,
        workers=workers,
        seed=seed if seed is not None else 0,
        epochs=5,
    )

    out = np.zeros((n, dim), dtype=float)
    for i in range(n):
        k = str(i)
        if k in w2v.wv:
            out[i] = w2v.wv[k]
    return out


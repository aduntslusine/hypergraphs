import numpy as np
from gensim.models import Word2Vec


def harw_embeddings(
    hg,
    p=1.0,
    q=1.0,
    r=1.0,
    gamma=-1.0,
    delta=0.0,
    num_walks=10,
    walk_length=80,
    window=10,
    dim=128,
    seed=None,
    workers=1,
):
    nodes = sorted(hg.get_nodes())
    n = len(nodes)
    if n == 0:
        return np.zeros((0, dim), dtype=float)

    edges = [frozenset(e) for e in hg.get_edges()]
    m = len(edges)
    if m == 0:
        return np.zeros((n, dim), dtype=float)

    idx = {u: i for i, u in enumerate(nodes)}

    d = np.zeros(n, dtype=float)
    inc_edges = [[] for _ in range(n)]
    for e_i, e in enumerate(edges):
        for u in e:
            i = idx[u]
            d[i] += 1.0
            inc_edges[i].append(e_i)

    rng = np.random.default_rng(seed)
    walks = []
    isolated_step_count = 0

    for _ in range(num_walks):
        for start in rng.permutation(n):
            v_i = int(start)
            walk = [v_i]
            prev_u_i = None
            prev_e_i = None

            for _ in range(walk_length - 1):
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

                e2 = edges[next_e_i]
                cand_x = [idx[x] for x in e2 if idx[x] != v_i]
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
                    deg_base = max(float(d[x_i]), 1.0)
                    raw_x[k] = a_node * (deg_base**delta)

                s_x = raw_x.sum()
                x_i = cand_x[int(rng.choice(len(cand_x), p=raw_x / s_x))]

                prev_u_i = v_i
                prev_e_i = next_e_i
                v_i = x_i
                walk.append(v_i)

            walks.append(walk)


    w2v = Word2Vec(
        sentences=[[str(i) for i in w] for w in walks],
        vector_size=dim,
        window=window,
        min_count=1,
        sg=1,
        workers=workers,
        seed=seed if seed is not None else 0,
        epochs=20,
    )

    out = np.zeros((n, dim), dtype=float)
    for i in range(n):
        k = str(i)
        if k in w2v.wv:
            out[i] = w2v.wv[k]
    return out



def hyper2vec_embeddings(
    hg,
    p=1.0,
    q=1.0,
    r=0.0,
    num_walks=10,
    walk_length=80,
    window=10,
    dim=128,
    seed=None,
    workers=1,
):
    nodes = sorted(hg.get_nodes())
    n = len(nodes)
    if n == 0:
        return np.zeros((0, dim), dtype=float)

    idx = {u: i for i, u in enumerate(nodes)}

    edges = [frozenset(e) for e in hg.get_edges()]
    w_e = np.ones(len(edges), dtype=float)
    if getattr(hg, "weighted", False) and hasattr(hg, "get_weights"):
        w_e = np.asarray(hg.get_weights(), dtype=float)

    d = np.zeros(n, dtype=float)
    inc_edges = [[] for _ in range(n)]
    for e_i, (e, w) in enumerate(zip(edges, w_e)):
        for u in e:
            i = idx[u]
            d[i] += float(w)
            inc_edges[i].append(e_i)

    co = [set() for _ in range(n)]
    for e in edges:
        ids = [idx[u] for u in e]
        for i in ids:
            co[i].update(ids)

    pi1_cache = [dict() for _ in range(n)]
    for v_i in range(n):
        vd = d[v_i]
        if vd <= 0:
            continue
        accum = {}
        for e_i in inc_edges[v_i]:
            e = edges[e_i]
            w = float(w_e[e_i])
            contrib = w / len(e)
            for x in e:
                x_i = idx[x]
                accum[x_i] = accum.get(x_i, 0.0) + contrib
        pi1_cache[v_i] = {x_i: val / vd for x_i, val in accum.items() if val > 0.0}

    rng = np.random.default_rng(seed)
    walks = []

    for _ in range(num_walks):
        for start in rng.permutation(n):
            walk = [int(start)]
            for _ in range(walk_length - 1):
                u_i = walk[-2] if len(walk) > 1 else None
                v_i = walk[-1]
                raw = np.zeros(n, dtype=float)

                for x_i in co[v_i]:
                    pi1 = pi1_cache[v_i].get(x_i, 0.0)
                    if pi1 <= 0:
                        continue

                    if u_i is None:
                        a = 1.0
                    elif x_i == u_i:
                        a = 1.0 / p
                    elif x_i in co[u_i]:
                        a = 1.0
                    else:
                        a = 1.0 / q

                    if r > 0:
                        b = float(d[x_i]) + r
                    elif r < 0:
                        b = 1.0 / (float(d[x_i]) - r)
                    else:
                        b = 1.0

                    raw[x_i] = a * b * pi1

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
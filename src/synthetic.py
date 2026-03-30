import random
import numpy as np
import xgi
from hypergraphx import Hypergraph


def generate_dataset(
    n=100,
    avg_degree=15,
    num_communities=3,
    intra_strength=0.85,
    avg_edge_size=3,
    max_hyperedge_size=5,
    seed=None,
):
    """Generate one synthetic DCSBM hypergraph dataset."""
    print(
        "generate_dataset params -> "
        f"n={n}, avg_degree={avg_degree}, num_communities={num_communities}, "
        f"intra_strength={intra_strength}, avg_edge_size={avg_edge_size}, "
        f"max_hyperedge_size={max_hyperedge_size}, seed={seed}"
    )

    rng = random.Random(seed)
    total_stubs = n * avg_degree
    
    # 1) Omega
    inter_strength = (1 - intra_strength) / (num_communities - 1)
    omega = np.full((num_communities, num_communities), inter_strength, dtype=float)
    np.fill_diagonal(omega, intra_strength)
    omega = (omega / omega.sum() * total_stubs).astype(int)
    omega[0, 0] += total_stubs - int(omega.sum())
    
    # 2) Node degrees k1
    k1 = {}
    remaining = total_stubs
    for i in range(n):
        if i == n - 1:
            k1[i] = remaining
        else:
            max_possible = min(remaining - (n - i - 1), 2 * avg_degree)
            k1[i] = rng.randint(1, max_possible)
            remaining -= k1[i]
    
    # 3) Edge sizes k2
    target_edges = max(1, total_stubs // avg_edge_size)
    edge_sizes = []
    remaining_stubs = total_stubs
    
    while remaining_stubs > 1:
        max_size = min(max_hyperedge_size, remaining_stubs)
        min_size = 2 if len(edge_sizes) < target_edges - 1 else max(2, remaining_stubs)
        size = rng.randint(min_size, max_size)
        edge_sizes.append(size)
        remaining_stubs -= size
    
    if remaining_stubs > 0 and edge_sizes:
        edge_sizes[-1] += remaining_stubs
    
    k2 = {i: edge_sizes[i] for i in range(len(edge_sizes))}
    
    # 4) Communities
    g1 = {i: rng.randint(0, num_communities - 1) for i in range(n)}
    g2 = {}
    node_ids = list(k1.keys())
    for edge_id, size in k2.items():
        chosen = rng.sample(node_ids, min(size, len(node_ids)))
        counts = np.zeros(num_communities)
        for node in chosen:
            counts[g1[node]] += 1
        g2[edge_id] = int(np.argmax(counts))
    
    # 5) Build with xgi, then convert to hypergraphx.Hypergraph
    H_xgi = xgi.dcsbm_hypergraph(k1, k2, g1, g2, omega)
    H_xgi.cleanup(connected=False)
    edge_list = [tuple(e) for e in H_xgi.edges.members()]
    H = Hypergraph(edge_list)
    
    return {
        "H": H,
        "Y": np.array([g1[i] for i in sorted(g1)]),
        "k1": k1,
        "k2": k2,
        "g1": g1,
        "g2": g2,
        "omega": omega,
    }


def generate_from_vector(v):
    """
    Vector format:
    [n, avg_degree, num_communities, intra_strength, avg_edge_size, max_hyperedge_size, seed(optional)]
    """
    if len(v) not in (6, 7):
        raise ValueError("Vector must have length 6 or 7.")
    seed = v[6] if len(v) == 7 else None
    return generate_dataset(
        n=int(v[0]),
        avg_degree=int(v[1]),
        num_communities=int(v[2]),
        intra_strength=float(v[3]),
        avg_edge_size=int(v[4]),
        max_hyperedge_size=int(v[5]),
        seed=seed,
    )


def generate_many(vectors):
    """Generate multiple datasets from a list of vectors."""
    return [generate_from_vector(v) for v in vectors]


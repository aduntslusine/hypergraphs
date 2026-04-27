import random

import numpy as np
from hypergraphx import Hypergraph
from sklearn.metrics import average_precision_score, roc_auc_score


def split_hypergraph(H, test_ratio, seed):
    rng = random.Random(seed)

    edges = list(H.get_edges())
    num_edges = len(edges)
    if num_edges == 0:
        print("Achieved test ratio: 0.0 (no hyperedges to split)")
        return Hypergraph([]), []

    target_test = int(test_ratio * num_edges)
    if target_test == 0:
        print(f"Achieved test ratio: 0.0 (target was 0 of {num_edges} edges)")
        return Hypergraph(edges[:]), []

    def collect_test_edges(require_connected):
        nonlocal working_edges, test_edges
        order = working_edges[:]
        rng.shuffle(order)
        for e in order:
            if len(test_edges) >= target_test:
                break
            if e not in working_edges:
                continue
            trial_edges = list(working_edges)
            trial_edges.remove(e)
            H_trial = Hypergraph(trial_edges)
            if H.num_nodes() != H_trial.num_nodes():
                continue
            if require_connected and not H_trial.is_connected():
                continue
            working_edges = trial_edges
            test_edges.append(e)

    working_edges = edges[:]
    test_edges = []
    collect_test_edges(require_connected=True)
    if len(test_edges) < target_test:
        print(
            "Connectivity constraint too strict for this hypergraph; "
            "continuing split without requiring a connected training hypergraph."
        )
        collect_test_edges(require_connected=False)

    H_train = Hypergraph(working_edges)

    achieved = len(test_edges) / num_edges
    print(
        f"Achieved test ratio: {achieved:.4f} "
        f"({len(test_edges)} / {num_edges} hyperedges held out)"
    )

    return H_train, test_edges


def generate_negative_hyperedges(
    H_full,
    test_edges,
    seed,
):
    real_edges = {frozenset(e) for e in H_full.get_edges()}
    all_nodes = list(H_full.get_nodes())
    rng = random.Random(seed)

    negatives = []
    negatives_set = set()

    for pos_edge in test_edges:
        k = len(pos_edge)
        found = False

        for _ in range(1000):
            cand = frozenset(rng.sample(all_nodes, k))
            if cand in real_edges:
                continue
            if cand in negatives_set:
                continue

            negatives.append(cand)
            negatives_set.add(cand)
            found = True
            break

        if not found:
            raise RuntimeError(f"Could not sample a valid negative hyperedge of size {k}.")

    return negatives


def score_edge(edge, emb):
    nodes = list(edge)
    sims = []

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            v1 = emb[nodes[i]]
            v2 = emb[nodes[j]]
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            sims.append(sim)

    return float(np.mean(sims))


def evaluate(test_edges, negatives, emb):
    pos_scores = [score_edge(e, emb) for e in test_edges]
    neg_scores = [score_edge(e, emb) for e in negatives]

    y_true = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
    y_score = np.array(pos_scores + neg_scores)

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return float(auc), float(ap)

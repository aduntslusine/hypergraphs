"""
Hyperedge prediction: build splits, run HMRW + Hyper2Vec, save CSV (append + resume).

Example:
  from run_hedge_prediction_experiments import run_hedge_prediction

  run_hedge_prediction(
      "contact_primary",
      "../www21-hyperlap-master/dataset/contact-primary_graph.txt",
      log_path="results/contact_primary/my_run.log",  # optional: mirror prints to file
  )
  run_hedge_prediction("threads_math", "path/to/graph.txt", test_ratio=0.1, n_splits=20)

  # Or from repo root:
  #   python run_hedge_prediction_experiments.py
  # (runs contact-primary; see bottom of this file.)
"""

import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from hypergraphx import Hypergraph

from src.hypergraph_train_test_split import (
    evaluate,
    generate_negative_hyperedges,
    split_hypergraph_train_test,
)
from src.utils import hmrw_embeddings, hyper2vec_embeddings

NUM_WALKS = 10
WALK_LENGTH = 100
DIM = 128

CONFIGS_HMRW = [
    {"p": 1.0, "q": 1.0, "r": 1.0, "gamma": -1.0, "delta": -1.0},
    {"p": 0.5, "q": 2.0, "r": 1.0, "gamma": -1.0, "delta": -1.0},
    {"p": 2.0, "q": 0.5, "r": 1.0, "gamma": -1.0, "delta": -1.0},
    {"p": 4.0, "q": 1.0, "r": 1.0, "gamma": -1.0, "delta": -1.0},
    {"p": 1.0, "q": 4.0, "r": 1.0, "gamma": -1.0, "delta": -1.0},
    {"p": 2.0, "q": 2.0, "r": 1.0, "gamma": -1.0, "delta": -1.0},
    {"p": 1.0, "q": 1.0, "r": 2.0, "gamma": -1.0, "delta": -1.0},
    {"p": 1.0, "q": 1.0, "r": 1.0, "gamma": 1.0, "delta": -1.0},
    {"p": 4.0, "q": 1.0, "r": 1.0, "gamma": 1.0, "delta": -1.0},
    
    {"p": 1.0, "q": 1.0, "r": 1.0, "gamma": -1.0, "delta": 0.0},
    {"p": 0.5, "q": 2.0, "r": 1.0, "gamma": -1.0, "delta": 0.0},
    {"p": 2.0, "q": 0.5, "r": 1.0, "gamma": -1.0, "delta": 0.0},
    {"p": 4.0, "q": 1.0, "r": 1.0, "gamma": -1.0, "delta": 0.0},
    {"p": 1.0, "q": 4.0, "r": 1.0, "gamma": -1.0, "delta": 0.0},
    {"p": 2.0, "q": 2.0, "r": 1.0, "gamma": -1.0, "delta": 0.0},
    {"p": 1.0, "q": 1.0, "r": 2.0, "gamma": -1.0, "delta": 0.0},
    {"p": 1.0, "q": 1.0, "r": 1.0, "gamma": 1.0, "delta": 0.0},
    {"p": 4.0, "q": 1.0, "r": 1.0, "gamma": 1.0, "delta": 0.0},
]

CONFIGS_H2V = [
    {"p": 1.0, "q": 1.0},
    {"p": 1.0, "q": 4.0},
    {"p": 4.0, "q": 1.0},
    {"p": 2.0, "q": 2.0},
    {"p": 0.5, "q": 2.0},
    {"p": 2.0, "q": 0.5},
]

R_H2V = 0.0


class _TeeStdout:
    """Send stdout to the terminal and to a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()



def load_txt(path: Path) -> Hypergraph:
    edges = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            edges.append([int(x) for x in line.replace(",", " ").split()])
    return Hypergraph(edges)


def resume_key(parts: list) -> str:
    return "|".join(str(x) for x in parts)


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if path.exists():
        df_new = pd.concat([pd.read_csv(path), df_new], ignore_index=True)
    df_new.to_csv(path, index=False)


def run_hedge_prediction(
    data_name: str,
    graph_path: str | Path,
    *,
    test_ratio: float = 0.1,
    n_splits: int = 20,
    force_splits: bool = False,
    log_path: str | Path | None = None,
) -> Path:
    """
    Run the full pipeline for one dataset. Results go under results/<data_name>/.
    Returns the path to the results CSV.

    If log_path is set, prints are copied to that file (append mode).
    """
    log_fp = None
    old_out = sys.stdout
    if log_path is not None:
        lp = Path(log_path)
        lp.parent.mkdir(parents=True, exist_ok=True)
        log_fp = open(lp, "a", encoding="utf-8")
        log_fp.write(f"\n===== start {datetime.now().isoformat()} =====\n")
        log_fp.write(
            f"data_name={data_name!r} graph_path={graph_path!r} "
            f"test_ratio={test_ratio} n_splits={n_splits}\n"
        )
        log_fp.flush()
        sys.stdout = _TeeStdout(old_out, log_fp)
    graph = Path(graph_path)
    seeds = list(range(n_splits))
    pct = int(round(test_ratio * 100))
    out_dir = Path("results") / data_name
    pkl_path = out_dir / f"splits_{pct}perc_{n_splits}splits.pkl"
    csv_path = out_dir / f"hedge_prediction_{pct}perc_{n_splits}splits.csv"
    graph_arg = str(graph)

    print("Loading graph:", graph.resolve())
    H = load_txt(graph)
    print("|V|=", len(H.get_nodes()), " |E|=", len(H.get_edges()))

    # --- splits: load or build ---
    splits = None
    if pkl_path.exists() and not force_splits:
        bundle = pickle.loads(pkl_path.read_bytes())
        if (
            bundle.get("dataset_path") == graph_arg
            and bundle.get("test_ratio") == test_ratio
            and bundle.get("split_seeds") == seeds
        ):
            splits = bundle["splits"]
            print("Loaded splits from", pkl_path)

    if splits is None:
        splits = []
        for fold, seed in enumerate(seeds):
            H_tr, test_e = split_hypergraph_train_test(H, test_ratio, seed)
            neg_e = generate_negative_hyperedges(H, test_e, seed)
            nodes_tr = sorted(H_tr.get_nodes())
            splits.append(
                {
                    "fold": fold,
                    "split_seed": seed,
                    "train_edge_list": list(H_tr.get_edges()),
                    "test_edges": test_e,
                    "negative_edges": neg_e,
                    "nodes_train": nodes_tr,
                }
            )
            print(
                f"  fold {fold} seed={seed} |E|_train={len(H_tr.get_edges())} "
                f"pos={len(test_e)} neg={len(neg_e)}"
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        pkl_path.write_bytes(
            pickle.dumps(
                {
                    "dataset_path": graph_arg,
                    "test_ratio": test_ratio,
                    "split_seeds": seeds,
                    "splits": splits,
                },
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        )
        print("Saved splits to", pkl_path)

    # --- which runs are already in the csv? ---
    done = set()
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        if "delta" not in old.columns:
            old["delta"] = 0.0
        for _, r in old.iterrows():
            done.add(
                resume_key(
                    [
                        r["method"],
                        r["config_id"],
                        int(r["fold"]),
                        int(r["split_seed"]),
                        float(r["p"]),
                        float(r["q"]),
                        float(r["r"]),
                        r["gamma"],
                        r["delta"],
                        int(r["num_walks"]),
                        int(r["walk_length"]),
                        int(r["dim"]),
                    ]
                )
            )
    print("Already have", len(done), "rows in", csv_path)

    # --- HMRW ---
    for i, cfg in enumerate(CONFIGS_HMRW):
        dlt = cfg.get("delta", 0.0)
        for sp in splits:
            fold, seed = sp["fold"], sp["split_seed"]
            k = resume_key(
                [
                    "hmrw",
                    f"hmrw_{i}",
                    fold,
                    seed,
                    cfg["p"],
                    cfg["q"],
                    cfg["r"],
                    cfg["gamma"],
                    dlt,
                    NUM_WALKS,
                    WALK_LENGTH,
                    DIM,
                ]
            )
            if k in done:
                continue

            H_tr = Hypergraph(sp["train_edge_list"])
            X = hmrw_embeddings(
                H_tr,
                p=cfg["p"],
                q=cfg["q"],
                r=cfg["r"],
                gamma=cfg["gamma"],
                delta=dlt,
                num_walks=NUM_WALKS,
                walk_length=WALK_LENGTH,
                dim=DIM,
                seed=seed,
            )
            emb = {u: X[j] for j, u in enumerate(sp["nodes_train"])}
            auc, ap = evaluate(sp["test_edges"], sp["negative_edges"], emb)
            row = {
                "method": "hmrw",
                "config_id": f"hmrw_{i}",
                "fold": fold,
                "split_seed": seed,
                "p": cfg["p"],
                "q": cfg["q"],
                "r": cfg["r"],
                "gamma": cfg["gamma"],
                "delta": dlt,
                "num_walks": NUM_WALKS,
                "walk_length": WALK_LENGTH,
                "dim": DIM,
                "auc": auc,
                "ap": ap,
            }
            append_csv(csv_path, row)
            done.add(k)
            print(f"HMRW {i} fold={fold} AUC={auc:.4f} AP={ap:.4f}")

    # --- Hyper2Vec ---
    nan = float("nan")
    for i, cfg in enumerate(CONFIGS_H2V):
        for sp in splits:
            fold, seed = sp["fold"], sp["split_seed"]
            k = resume_key(
                [
                    "hyper2vec",
                    f"h2v_{i}",
                    fold,
                    seed,
                    cfg["p"],
                    cfg["q"],
                    R_H2V,
                    nan,
                    nan,
                    NUM_WALKS,
                    WALK_LENGTH,
                    DIM,
                ]
            )
            if k in done:
                continue

            H_tr = Hypergraph(sp["train_edge_list"])
            X = hyper2vec_embeddings(
                H_tr,
                p=cfg["p"],
                q=cfg["q"],
                r=R_H2V,
                num_walks=NUM_WALKS,
                walk_length=WALK_LENGTH,
                dim=DIM,
                seed=seed,
            )
            emb = {u: X[j] for j, u in enumerate(sp["nodes_train"])}
            auc, ap = evaluate(sp["test_edges"], sp["negative_edges"], emb)
            row = {
                "method": "hyper2vec",
                "config_id": f"h2v_{i}",
                "fold": fold,
                "split_seed": seed,
                "p": cfg["p"],
                "q": cfg["q"],
                "r": R_H2V,
                "gamma": nan,
                "delta": nan,
                "num_walks": NUM_WALKS,
                "walk_length": WALK_LENGTH,
                "dim": DIM,
                "auc": auc,
                "ap": ap,
            }
            append_csv(csv_path, row)
            done.add(k)
            print(f"Hyper2Vec {i} fold={fold} AUC={auc:.4f} AP={ap:.4f}")

    # --- mean columns ---
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        m = df.groupby(
            ["method", "config_id", "p", "q", "r", "gamma", "delta"],
            dropna=False,
        )[["auc", "ap"]].transform("mean")
        df["mean_auc"] = m["auc"]
        df["mean_ap"] = m["ap"]
        df.to_csv(csv_path, index=False)

    print("Done. Wrote", csv_path)
    if log_fp is not None:
        sys.stdout = old_out
        log_fp.write(f"\n===== end {datetime.now().isoformat()} =====\n")
        log_fp.flush()
        log_fp.close()
    return csv_path


if __name__ == "__main__":
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_hedge_prediction(
        "contact_primary",
        "../www21-hyperlap-master/dataset/contact-primary_graph.txt",
        log_path=Path("results") / "contact_primary" / f"run_{_ts}.log",
    )

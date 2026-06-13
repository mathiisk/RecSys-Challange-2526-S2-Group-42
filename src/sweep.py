"""Hyperparameter sweep for BPR and GRU4Rec, and ensemble weight sweep.

CLI:
    python sweep.py --bpr-only
    python sweep.py --gru-only
    python sweep.py --weights-only --load checkpoints/final
    python sweep.py --output results/sweep_v1.csv
"""
import argparse
import itertools
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bpr import train_bpr, BPRRecommender, BPRMF
from gru4rec import train_gru4rec, GRU4RecRecommender, GRU4Rec
from lightgcn import LightGCNRecommender, LightGCN
from popularity import PopularityRecommender
from evaluate import evaluate_model
from ensemble import chunked_ensemble_sweep

DATA_DIR = Path("data/processed")

BPR_GRID = {
    "dim":          [64, 128, 256],
    "lr":           [5e-4, 1e-3, 2e-3],
    "weight_decay": [1e-6, 1e-5, 1e-4],
    "n_epochs":     [50, 100],
}

GRU_GRID = {
    "hidden_dim":  [64, 128, 256],
    "lr":          [5e-4, 1e-3],
    "dropout":     [0.1, 0.2, 0.3],
    "n_epochs":    [100, 150],
    "max_seq_len": [50, 100],
}


def dict_combinations(grid: dict) -> list[dict]:
    """Return all combinations of a hyperparameter grid.

    Args:
        grid (dict): Mapping of parameter name to list of values to try.

    Returns:
        (list[dict]): List of parameter dicts, one per combination.
    """
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*grid.values())]


def sweep_bpr(train_df: pd.DataFrame, val_df: pd.DataFrame, n_users: int, n_items: int, device: str) -> list[dict]:
    """Sweep BPR hyperparameters and return results sorted by recall.

    Args:
        train_df (pd.DataFrame): Training interactions.
        val_df (pd.DataFrame): Validation interactions.
        n_users (int): Maximum user ID.
        n_items (int): Maximum item ID.
        device (str): Torch device string.

    Returns:
        (list[dict]): Results sorted by recall@10 descending.
    """
    combos = dict_combinations(BPR_GRID)
    results = []
    print(f"\n=== BPR sweep: {len(combos)} combinations ===")

    for i, params in enumerate(combos):
        t0 = time.time()
        model = train_bpr(train_df, n_users, n_items, dim=params["dim"], n_epochs=params["n_epochs"],
            lr=params["lr"], weight_decay=params["weight_decay"], batch_size=1024, device=device, verbose=False)
        rec = BPRRecommender(model, device=device)
        recall = evaluate_model(lambda user_id, seen_items, k: rec.recommend(user_id, seen_items, k), val_df, train_df, k=10)
        elapsed = time.time() - t0
        results.append({"model": "bpr", "recall@10": recall, "time_s": round(elapsed, 1), **params})
        print(f"  [{i+1:3d}/{len(combos)}] recall={recall:.4f} | {params} | {elapsed:.1f}s")

    return sorted(results, key=lambda x: x["recall@10"], reverse=True)


def sweep_gru(train_df: pd.DataFrame, val_df: pd.DataFrame, n_items: int, device: str) -> list[dict]:
    """Sweep GRU4Rec hyperparameters and return results sorted by recall.

    Args:
        train_df (pd.DataFrame): Training interactions.
        val_df (pd.DataFrame): Validation interactions.
        n_items (int): Maximum item ID.
        device (str): Torch device string.

    Returns:
        (list[dict]): Results sorted by recall@10 descending.
    """
    combos = dict_combinations(GRU_GRID)
    results = []
    print(f"\n=== GRU sweep: {len(combos)} combinations ===")

    for i, params in enumerate(combos):
        t0 = time.time()
        model, sequences = train_gru4rec(train_df, n_items, embed_dim=params["hidden_dim"], hidden_dim=params["hidden_dim"],
            n_epochs=params["n_epochs"], lr=params["lr"], dropout=params["dropout"], max_seq_len=params["max_seq_len"],
            num_layers=1, batch_size=256, weight_decay=1e-5, device=device, verbose=False)
        rec = GRU4RecRecommender(model, sequences, device=device)
        recall = evaluate_model(lambda user_id, seen_items, k: rec.recommend(user_id, seen_items, k), val_df, train_df, k=10)
        elapsed = time.time() - t0
        results.append({"model": "gru", "recall@10": recall, "time_s": round(elapsed, 1), **params})
        print(f"  [{i+1:3d}/{len(combos)}] recall={recall:.4f} | {params} | {elapsed:.1f}s")

    return sorted(results, key=lambda x: x["recall@10"], reverse=True)


def load_checkpoint_recs(checkpoint_dir: Path, device: str) -> tuple:
    """Load recommender wrappers from checkpoint saved by train.py.

    Args:
        checkpoint_dir (Path): Directory containing checkpoint files.
        device (str): Torch device string.

    Returns:
        (tuple): bpr_rec, gru_rec, lightgcn_rec, pop_rec, n_items.
    """
    bpr_blob = torch.load(checkpoint_dir / "bpr.pt", map_location=device, weights_only=False)
    bpr_model = BPRMF(bpr_blob["n_users"], bpr_blob["n_items"], dim=bpr_blob["dim"]).to(device)
    bpr_model.load_state_dict(bpr_blob["state_dict"])

    gru_blob = torch.load(checkpoint_dir / "gru.pt", map_location=device, weights_only=False)
    gru_model = GRU4Rec(gru_blob["n_items"], embed_dim=gru_blob["embed_dim"], hidden_dim=gru_blob["hidden_dim"]).to(device)
    gru_model.load_state_dict(gru_blob["state_dict"])

    lightgcn_blob = torch.load(checkpoint_dir / "lightgcn.pt", map_location=device, weights_only=False)
    lightgcn_model = LightGCN(lightgcn_blob["n_users"], lightgcn_blob["n_items"],
        dim=lightgcn_blob["dim"], n_layers=lightgcn_blob["n_layers"]).to(device)
    lightgcn_model.load_state_dict(lightgcn_blob["state_dict"])
    norm_adjacency = torch.load(checkpoint_dir / "norm_adjacency.pt", map_location=device, weights_only=False)

    with open(checkpoint_dir / "gru_sequences.pkl", "rb") as f:
        gru_sequences = pickle.load(f)
    with open(checkpoint_dir / "popularity.pkl", "rb") as f:
        popularity, n_items = pickle.load(f)

    return (BPRRecommender(bpr_model, device=device), GRU4RecRecommender(gru_model, gru_sequences, device=device),
        LightGCNRecommender(lightgcn_model, norm_adjacency, device=device), PopularityRecommender(popularity, n_items, device=device), n_items)


def sweep_weights(load_dir: Path, step: float = 1.0) -> None:
    """Load checkpoint and sweep all ensemble weights.

    Args:
        load_dir (Path): Directory containing checkpoint files.
        step (float): Weight grid step size.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")

    print(f"loading checkpoint from {load_dir}...")
    bpr_rec, gru_rec, lightgcn_rec, pop_rec, n_items = load_checkpoint_recs(load_dir, device)

    weight_values = [float(w) for w in range(0, 11)]
    weight_grid = [
        (bpr, gru, lgcn, pop)
        for bpr, gru, lgcn, pop in itertools.product(weight_values, weight_values, weight_values, weight_values)
        if not (bpr == 0.0 and gru == 0.0 and lgcn == 0.0 and pop == 0.0)
    ]
    print(f"sweeping {len(weight_grid)} weight combinations...")

    components = [("bpr", bpr_rec), ("gru", gru_rec), ("lightgcn", lightgcn_rec), ("pop", pop_rec)]
    t0 = time.time()
    recalls = chunked_ensemble_sweep(components, weight_grid, val_df, train_df, k=10, batch_size=1024, device=device)
    print(f"sweep done in {time.time() - t0:.1f}s\n")

    results = sorted(zip(weight_grid, recalls), key=lambda x: x[1], reverse=True)
    print("bpr  | gru  | lgcn | pop    recall@10")
    for weights, recall in results[:20]:
        print(f"{weights[0]:>4.1f} | {weights[1]:>4.1f} | {weights[2]:>4.1f} | {weights[3]:>4.1f}   {recall:.4f}")

    best_weights, best_recall = results[0]
    print(f"\nbest: --weights {','.join(str(w) for w in best_weights)}  recall={best_recall:.4f}")


def main(sweep_bpr_flag: bool = True, sweep_gru_flag: bool = True, sweep_weights_flag: bool = False,
    load_dir: Path | None = None, output_path: Path = Path("sweep_results.csv"), step: float = 1.0) -> None:
    """Run hyperparameter and/or weight sweeps.

    Args:
        sweep_bpr_flag (bool): Whether to sweep BPR hyperparameters.
        sweep_gru_flag (bool): Whether to sweep GRU4Rec hyperparameters.
        sweep_weights_flag (bool): Whether to sweep ensemble weights from checkpoint.
        load_dir (Path | None): Checkpoint directory, required for weight sweep.
        output_path (Path): Path to write combined hyperparam results CSV.
        step (float): Weight grid step size for ensemble sweep.
    """
    if sweep_weights_flag:
        assert load_dir is not None, "--load is required for --weights-only"
        sweep_weights(load_dir, step=step)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    n_users = int(max(train_df["user_id"].max(), val_df["user_id"].max()))
    n_items = int(max(train_df["item_id"].max(), val_df["item_id"].max()))
    print(f"users: {n_users}, items: {n_items}, device: {device}")

    all_results = []

    if sweep_bpr_flag:
        bpr_results = sweep_bpr(train_df, val_df, n_users, n_items, device)
        all_results.extend(bpr_results)
        print(f"\n--- BPR top 5 ---")
        for r in bpr_results[:5]:
            print(f"  recall={r['recall@10']:.4f} | dim={r['dim']} lr={r['lr']} wd={r['weight_decay']} epochs={r['n_epochs']}")

    if sweep_gru_flag:
        gru_results = sweep_gru(train_df, val_df, n_items, device)
        all_results.extend(gru_results)
        print(f"\n--- GRU top 5 ---")
        for r in gru_results[:5]:
            print(f"  recall={r['recall@10']:.4f} | hidden={r['hidden_dim']} lr={r['lr']} dropout={r['dropout']} epochs={r['n_epochs']} seq={r['max_seq_len']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_csv(output_path, index=False)
    print(f"\nsaved {len(all_results)} results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bpr-only", action="store_true")
    parser.add_argument("--gru-only", action="store_true")
    parser.add_argument("--weights-only", action="store_true")
    parser.add_argument("--load", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("sweep_results.csv"))
    parser.add_argument("--step", type=float, default=1.0)
    args = parser.parse_args()

    main(
        sweep_bpr_flag=not args.gru_only and not args.weights_only,
        sweep_gru_flag=not args.bpr_only and not args.weights_only,
        sweep_weights_flag=args.weights_only,
        load_dir=args.load,
        output_path=args.output,
        step=args.step,
    )
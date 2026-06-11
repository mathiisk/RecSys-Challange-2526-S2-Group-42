"""Hyperparameter sweep for BPR and GRU4Rec.

Trains each combination on processed/train, evaluates on processed/val.
Results are printed and saved to sweep_results.csv.

CLI:
    python sweep.py
    python sweep.py --output results/sweep_v1.csv
"""
import argparse
import itertools
import time
from pathlib import Path

import pandas as pd
import torch

from bpr import train_bpr, BPRRecommender
from gru4rec import train_gru4rec, GRU4RecRecommender
from evaluate import evaluate_model

DATA_DIR = Path("data/processed")

BPR_GRID = {
    "dim":          [64, 128, 256],
    "lr":           [5e-4, 1e-3, 2e-3],
    "weight_decay": [1e-6, 1e-5, 1e-4],
    "n_epochs":     [50, 100],
}

GRU_GRID = {
    "hidden_dim":   [64, 128, 256],
    "lr":           [5e-4, 1e-3],
    "dropout":      [0.1, 0.2, 0.3],
    "n_epochs":     [100, 150],
    "max_seq_len":  [50, 100],
}


def dict_combinations(grid: dict) -> list[dict]:
    """Return all combinations of a hyperparameter grid.

    Args:
        grid (dict): Mapping of parameter name to list of values to try.

    Returns:
        (list[dict]): List of parameter dicts, one per combination.
    """
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def sweep_bpr(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    device: str,
) -> list[dict]:
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
        model = train_bpr(
            train_df, n_users, n_items,
            dim=params["dim"],
            n_epochs=params["n_epochs"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            batch_size=1024,
            device=device,
            verbose=False,
        )
        rec = BPRRecommender(model, device=device)
        recall = evaluate_model(
            lambda user_id, seen_items, k: rec.recommend(user_id, seen_items, k),
            val_df, train_df, k=10,
        )
        elapsed = time.time() - t0
        result = {"model": "bpr", "recall@10": recall, "time_s": round(elapsed, 1), **params}
        results.append(result)
        print(f"  [{i+1:3d}/{len(combos)}] recall={recall:.4f} | {params} | {elapsed:.1f}s")

    return sorted(results, key=lambda x: x["recall@10"], reverse=True)


def sweep_gru(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_items: int,
    device: str,
) -> list[dict]:
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
        model, sequences = train_gru4rec(
            train_df, n_items,
            embed_dim=params["hidden_dim"],
            hidden_dim=params["hidden_dim"],
            n_epochs=params["n_epochs"],
            lr=params["lr"],
            dropout=params["dropout"],
            max_seq_len=params["max_seq_len"],
            num_layers=1,
            batch_size=256,
            weight_decay=1e-5,
            device=device,
            verbose=False,
        )
        rec = GRU4RecRecommender(model, sequences, device=device)
        recall = evaluate_model(
            lambda user_id, seen_items, k: rec.recommend(user_id, seen_items, k),
            val_df, train_df, k=10,
        )
        elapsed = time.time() - t0
        result = {"model": "gru", "recall@10": recall, "time_s": round(elapsed, 1), **params}
        results.append(result)
        print(f"  [{i+1:3d}/{len(combos)}] recall={recall:.4f} | {params} | {elapsed:.1f}s")

    return sorted(results, key=lambda x: x["recall@10"], reverse=True)


def main(
    sweep_bpr_flag: bool = True,
    sweep_gru_flag: bool = True,
    output_path: Path = Path("sweep_results.csv"),
) -> None:
    """Run hyperparameter sweeps and save results.

    Args:
        sweep_bpr_flag (bool): Whether to sweep BPR hyperparameters.
        sweep_gru_flag (bool): Whether to sweep GRU4Rec hyperparameters.
        output_path (Path): Path to write combined results CSV.
    """
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
            print(f"  recall={r['recall@10']:.4f} | dim={r['dim']} lr={r['lr']} "
                  f"wd={r['weight_decay']} epochs={r['n_epochs']}")

    if sweep_gru_flag:
        gru_results = sweep_gru(train_df, val_df, n_items, device)
        all_results.extend(gru_results)
        print(f"\n--- GRU top 5 ---")
        for r in gru_results[:5]:
            print(f"  recall={r['recall@10']:.4f} | hidden={r['hidden_dim']} lr={r['lr']} "
                  f"dropout={r['dropout']} epochs={r['n_epochs']} seq={r['max_seq_len']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_csv(output_path, index=False)
    print(f"\nsaved {len(all_results)} results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bpr-only", action="store_true")
    parser.add_argument("--gru-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("sweep_results.csv"))
    args = parser.parse_args()

    sweep_bpr_flag = not args.gru_only
    sweep_gru_flag = not args.bpr_only

    main(
        sweep_bpr_flag=sweep_bpr_flag,
        sweep_gru_flag=sweep_gru_flag,
        output_path=args.output,
    )
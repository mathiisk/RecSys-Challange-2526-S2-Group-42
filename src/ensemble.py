import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATA_DIR = Path("data/processed")


def minmax_normalize_rows(scores: torch.Tensor) -> torch.Tensor:
    """Per-row min-max normalize to [0, 1]. Works for 1D and 2D tensors."""
    s_min = scores.min(dim=-1, keepdim=True).values
    s_max = scores.max(dim=-1, keepdim=True).values
    denom = (s_max - s_min).clamp(min=1e-12)
    return (scores - s_min) / denom


def build_seen_mask(
    batch_users: list[int],
    seen_per_user: dict[int, set],
    ref_tensor: torch.Tensor,
) -> torch.Tensor:
    """Build a (B, V) mask of -inf at seen items and at index 0, 0 elsewhere.

    `ref_tensor` is any (B, V) tensor on the right device/dtype to mirror.
    """
    mask_rows: list[int] = []
    mask_cols: list[int] = []
    for i, uid in enumerate(batch_users):
        seen = seen_per_user.get(uid, set())
        if seen:
            mask_rows.extend([i] * len(seen))
            mask_cols.extend(seen)
    mask = torch.zeros_like(ref_tensor)
    mask[:, 0] = float("-inf")
    if mask_rows:
        r = torch.tensor(mask_rows, dtype=torch.long, device=ref_tensor.device)
        c = torch.tensor(mask_cols, dtype=torch.long, device=ref_tensor.device)
        mask[r, c] = float("-inf")
    return mask


class EnsembleRecommender:
    """Linearly combines several score sources at the score level.

    Each component must expose:
      - `score_user(user_id) -> 1D tensor` of length n_items + 1
      - `batch_score_users(user_ids) -> 2D tensor` of shape (B, n_items + 1)
    """

    def __init__(
        self,
        components: list[tuple[str, object, float]],
        device: str = "cpu",
    ):
        self.components = components
        self.device = device

    @torch.no_grad()
    def score_user(self, user_id: int) -> torch.Tensor:
        combined: torch.Tensor | None = None
        for _, rec, weight in self.components:
            s = rec.score_user(user_id).float().to(self.device)
            s = minmax_normalize_rows(s) * weight
            combined = s if combined is None else combined + s
        assert combined is not None, "ensemble has no components"
        return combined

    @torch.no_grad()
    def batch_score_users(self, user_ids: list[int]) -> torch.Tensor:
        combined: torch.Tensor | None = None
        for _, rec, weight in self.components:
            s = rec.batch_score_users(user_ids).float().to(self.device)
            s = minmax_normalize_rows(s) * weight
            combined = s if combined is None else combined + s
        assert combined is not None, "ensemble has no components"
        return combined

    def recommend(self, user_id: int, seen_items: set, k: int = 10) -> list[int]:
        scores = self.score_user(user_id)
        scores[0] = float("-inf")
        if seen_items:
            seen = torch.tensor(list(seen_items), dtype=torch.long, device=self.device)
            scores[seen] = float("-inf")
        return torch.topk(scores, k).indices.cpu().tolist()


def chunked_ensemble_sweep(
    components: list[tuple[str, object]],
    weight_grid: list[tuple[float, ...]],
    val_df: pd.DataFrame,
    train_df: pd.DataFrame,
    k: int = 10,
    batch_size: int = 1024,
    device: str = "cpu",
) -> list[float]:
    """Sweep ensemble weights over a fixed component set, sharing per-chunk score matrices.

    Returns recall@k for each weight combo, in the same order as weight_grid.
    """
    seen_per_user = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    user_ids = val_df["user_id"].values.astype(np.int64)
    target_items = val_df["item_id"].values.astype(np.int64)
    n = len(user_ids)
    n_combos = len(weight_grid)

    hits = np.zeros(n_combos, dtype=np.int64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_users = user_ids[start:end].tolist()
        batch_targets = torch.from_numpy(target_items[start:end]).to(device)

        per_comp = []
        for _, rec in components:
            s = rec.batch_score_users(batch_users).float().to(device)
            per_comp.append(minmax_normalize_rows(s))

        mask = build_seen_mask(batch_users, seen_per_user, per_comp[0])

        for w_idx, weights in enumerate(weight_grid):
            combined = torch.zeros_like(per_comp[0])
            for w, s in zip(weights, per_comp):
                if w != 0:
                    combined = combined + w * s
            combined = combined + mask
            top_k = torch.topk(combined, k, dim=-1).indices  # (B, k)
            hits[w_idx] += (top_k == batch_targets.unsqueeze(-1)).any(dim=-1).sum().item()

    return [float(h) / n for h in hits]


DEFAULT_WEIGHT_GRID: list[tuple[float, float, float, float]] = [
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0, 0.0),
    (1.0, 0.0, 1.0, 0.0),
    (0.0, 1.0, 1.0, 0.0),
    (1.0, 1.0, 1.0, 0.0),
    (1.0, 1.0, 1.0, 1.0),
    (2.0, 1.0, 1.0, 0.0),
    (1.0, 2.0, 1.0, 0.0),
    (1.0, 1.0, 2.0, 0.0),
    (2.0, 1.0, 2.0, 0.0),
    (1.0, 2.0, 2.0, 0.0),
    (2.0, 2.0, 1.0, 0.0),
    (3.0, 2.0, 1.0, 0.0),
    (1.0, 2.0, 3.0, 0.0),
]


def run_sweep(
    bpr_epochs: int = 20,
    gru_epochs: int = 20,
    bpr_dim: int = 64,
    gru_hidden_dim: int = 64,
    gru_max_seq_len: int = 50,
    weight_grid: list[tuple[float, ...]] | None = None,
    device: str | None = None,
    seed: int = 42,
) -> tuple[list[tuple[float, ...]], list[float]]:
    """Train BPR + GRU on processed/train, sweep ensemble weights on processed/val."""
    from popularity import compute_item_popularity, PopularityRecommender
    from bpr import train_bpr, BPRRecommender
    from gru4rec import train_gru4rec, GRU4RecRecommender

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")

    n_users = int(max(train_df["user_id"].max(), val_df["user_id"].max()))
    n_items = int(max(train_df["item_id"].max(), val_df["item_id"].max()))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"users: {n_users}, items: {n_items}, device: {device}")

    t0 = time.time()
    print(f"\n[1/3] training BPR-MF ({bpr_epochs} epochs)...")
    bpr_model = train_bpr(
        train_df, n_users, n_items,
        dim=bpr_dim, n_epochs=bpr_epochs, batch_size=1024,
        lr=1e-3, weight_decay=1e-5, device=device, seed=seed, verbose=False,
    )
    bpr_rec = BPRRecommender(bpr_model, device=device)
    print(f"  done in {time.time() - t0:.1f}s")

    t0 = time.time()
    print(f"[2/3] training GRU4Rec ({gru_epochs} epochs)...")
    gru_model, sequences = train_gru4rec(
        train_df, n_items,
        embed_dim=gru_hidden_dim, hidden_dim=gru_hidden_dim,
        num_layers=1, dropout=0.2,
        max_seq_len=gru_max_seq_len, n_epochs=gru_epochs, batch_size=256,
        lr=1e-3, weight_decay=1e-5, device=device, seed=seed, verbose=False,
    )
    gru_rec = GRU4RecRecommender(gru_model, sequences, device=device)
    print(f"  done in {time.time() - t0:.1f}s")

    print("[3/3] computing popularity...")
    popularity = compute_item_popularity(train_df, n_items=n_items)
    pop_rec = PopularityRecommender(popularity, n_items, device=device)

    components = [("bpr", bpr_rec), ("gru", gru_rec), ("pop", pop_rec)]
    weight_grid = weight_grid or DEFAULT_WEIGHT_GRID

    print("\nrunning chunked sweep...")
    t0 = time.time()
    recalls = chunked_ensemble_sweep(
        components, weight_grid, val_df, train_df,
        k=10, batch_size=1024, device=device,
    )
    print(f"sweep done in {time.time() - t0:.1f}s\n")

    print("bpr | gru | pop   recall@10")
    for weights, r in zip(weight_grid, recalls):
        ws = " | ".join(f"{w:>3.1f}" for w in weights)
        print(f"{ws}   {r:.4f}")

    return weight_grid, recalls


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    run_sweep()

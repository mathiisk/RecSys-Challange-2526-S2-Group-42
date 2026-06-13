import numpy as np
import pandas as pd
import torch


def minmax_normalize_rows(scores: torch.Tensor) -> torch.Tensor:
    """Per-row min-max normalize to [0, 1]. Works for 1D and 2D tensors.

    Args:
        scores (torch.Tensor): Score tensor, 1D or 2D.

    Returns:
        (torch.Tensor): Normalized scores, same shape as input.
    """
    s_min = scores.min(dim=-1, keepdim=True).values
    s_max = scores.max(dim=-1, keepdim=True).values
    denom = (s_max - s_min).clamp(min=1e-12)
    return (scores - s_min) / denom


def build_seen_mask(batch_users: list[int], seen_per_user: dict[int, set], ref_tensor: torch.Tensor) -> torch.Tensor:
    """Build a (B, V) mask of -inf at seen items and at index 0, 0 elsewhere.

    Args:
        batch_users (list[int]): User IDs in this batch.
        seen_per_user (dict[int, set]): Mapping of user_id -> set of seen item IDs.
        ref_tensor (torch.Tensor): Any (B, V) tensor on the right device/dtype to mirror.

    Returns:
        (torch.Tensor): Mask of the same shape as ref_tensor.
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

    Args:
        components (list[tuple[str, object]]): Named recommender components.
        weight_grid (list[tuple[float, ...]]): Weight combinations to evaluate.
        val_df (pd.DataFrame): Validation interactions with columns [user_id, item_id].
        train_df (pd.DataFrame): Training interactions used to build the seen-items mask.
        k (int): Cutoff rank for recall.
        batch_size (int): Number of users to score per batch.
        device (str): Torch device string.

    Returns:
        (list[float]): Recall@k for each weight combo, in the same order as weight_grid.
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
            top_k = torch.topk(combined, k, dim=-1).indices
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
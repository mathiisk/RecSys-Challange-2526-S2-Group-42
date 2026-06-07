import numpy as np
import pandas as pd
import torch
from pathlib import Path

DATA_DIR = Path("data/processed")

def compute_item_popularity(train_df: pd.DataFrame, n_items: int | None = None) -> np.ndarray:
    """Compute interaction count for each item, indexed by item_id - 1.

    Args:
        train_df: Training interactions with columns [user_id, item_id, timestamp].
        n_items: If given, size the output to (n_items,). Otherwise use the largest
            item_id observed in train_df. Provide this explicitly when downstream code
            uses a different `n_items` (e.g. covering val/test/submission too).

    Returns:
        np.ndarray of shape (n_items,) where index `i` holds the count for item_id == i+1.
    """
    if n_items is None:
        n_items = int(train_df["item_id"].max())
    popularity = np.zeros(n_items, dtype=np.int32)
    counts = train_df["item_id"].value_counts()
    in_range = counts.index <= n_items
    popularity[counts.index[in_range] - 1] = counts.values[in_range]
    return popularity


def recommend_popular(user_id: int, seen_items: set[int], k: int, popularity: np.ndarray) -> list[int]:
    """Recommend top-k most popular items, excluding items the user has already seen.

    Args:
        user_id (int): User ID (unused, popularity is user-agnostic)
        seen_items (set[int]): Items to exclude from recommendations
        k (int): Number of items to recommend
        popularity (np.ndarray): Popularity scores indexed by item_id - 1

    Returns:
        list[int]: Top-k item IDs (1-indexed)
    """
    scores = popularity.copy().astype(float)
    if seen_items:
        scores[np.array(list(seen_items), dtype=np.int64) - 1] = -1
    top_k_indices = np.argpartition(scores, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]
    return (top_k_indices + 1).tolist()


class PopularityRecommender:
    """Wraps an item-popularity array as a `(user_id, seen_items, k) -> list[int]` callable
    that also exposes `score_user(user_id) -> tensor[n_items+1]` for ensembling.
    """

    def __init__(self, popularity: np.ndarray, n_items: int, device: str = "cpu"):
        scores = np.zeros(n_items + 1, dtype=np.float32)
        usable = min(len(popularity), n_items)
        scores[1 : usable + 1] = popularity[:usable].astype(np.float32)
        self.scores = torch.from_numpy(scores).to(device)
        self.device = device

    def score_user(self, user_id: int) -> torch.Tensor:
        return self.scores.clone()

    def batch_score_users(self, user_ids: list[int]) -> torch.Tensor:
        return self.scores.unsqueeze(0).expand(len(user_ids), -1).clone()

    def recommend(self, user_id: int, seen_items: set, k: int = 10) -> list[int]:
        scores = self.score_user(user_id)
        scores[0] = float("-inf")
        if seen_items:
            seen = torch.tensor(list(seen_items), dtype=torch.long, device=self.device)
            scores[seen] = float("-inf")
        return torch.topk(scores, k).indices.cpu().tolist()


if __name__ == "__main__":
    from evaluate import evaluate_model
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    popularity = compute_item_popularity(train_df)
    recommend_fn = lambda user_id, seen_items, k: recommend_popular(user_id, seen_items, k, popularity)
    evaluate_model(recommend_fn, val_df, train_df)

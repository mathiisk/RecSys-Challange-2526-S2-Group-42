import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

def compute_item_popularity(train_df: pd.DataFrame) -> np.ndarray:
    """Compute interaction count for each item, indexed by item_id

    Args:
        train_df (pd.DataFrame): Training interactions with colums [user_id, item_id, timestamp]

    Returns:
        np.ndarray: Array of shape (n_items, ) where index is item_id - 1 and value is interaction count
    """
    n_items = train_df["item_id"].max()
    popularity = np.zeros(n_items, dtype=np.int32)
    counts = train_df["item_id"].value_counts()
    popularity[counts.index - 1] = counts.values
    return popularity


def recommend_popular(used_id: int, seen_items: set[int], k: int, popularity: np.ndarray) -> list[int]:
    """Recommend top-k most popular items, excluding items the user has already seen.

    Args:
        user_id (int): User ID (unused, popularity is user-agnostic)
        seen_items (set[int]): Items to exclude from recommendations
        k (int): Number of items to recommend
        popularity (np.ndarray): Popularity scores indexed by item_id - 1

    Returns:
        list[int]: Top-k item IDs (1-indexed)
    """
    item_ids = np.arange(1, len(popularity) + 1)
    scores = popularity.copy().astype(float)
    scores[np.array(list(seen_items), dtype=np.int32) - 1] = -1
    top_k_indices = np.argpartition(scores, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]
    return (top_k_indices + 1).tolist()


if __name__ == "__main__":
    from evaluate import evaluate_model
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    popularity = compute_item_popularity(train_df)
    recommend_fn = lambda user_id, seen_items, k: recommend_popular(user_id, seen_items, k, popularity)
    evaluate_model(recommend_fn, val_df, train_df)

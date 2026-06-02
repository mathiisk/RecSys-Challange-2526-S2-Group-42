import numpy as np
import pandas as pd

def recall_at_k(recommended_items: list[int], ground_truth_item: int, k: int =10) -> float:
    """Compute Recall@K for a single user

    Args:
        recommended_items (list[int]): Ranked list of recommended item IDs
        ground_truth_item (int): The single held-out item for this user
        k (int, optional): Cutoff rank. Defaults to 10.

    Returns:
        float: 1.0 if ground_truth_item is in top-k recommendations, esle 0.0
    """
    return float(ground_truth_item in recommended_items[:k])


def evaluate_model(recommend_fn, val_df: pd.DataFrame, train_df: pd.DataFrame, k: int = 10) -> float:
    """Evaluate a recommendation function on the validation set using Recall@K

    Args:
        recommend_fn (callable): Function with signature (user_id, seen_items, k) -> list[int]
        val_df (pd.DataFrame): Validation set with colums [user_id, item_id]
        train_df (pd.DataFrame): Training set used to get each user's seen items
        k (int, optional): Cutoff rank. Defaults to 10.

    Returns:
        float: Mean Recall@K across all users in val_df
    """
    seen_item_per_user = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    scores = []
    for row in val_df.itertuples():
        seen_items = seen_item_per_user.get(row.user_id, set())
        recommendations = recommend_fn(row.user_id, seen_items, k)
        scores.append(recall_at_k(recommendations, row.item_id, k))
        
    mean_recall = float(np.mean(scores))
    print(f"recall@{k}: {mean_recall:.4f}")
    return mean_recall
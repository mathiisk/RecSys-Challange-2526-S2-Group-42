import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

DATA_DIR = Path("data/processed")


class BPRMF(nn.Module):
    """Bayesian Personalized Ranking with Matrix Factorization.

    IDs are 1-indexed in the data, so embedding tables are sized to max_id + 1
    and index 0 is reserved (masked out in the recommender).

    Score(user, item) = user_emb(user) x item_emb(item) + item_bias(item)
    """

    def __init__(self, n_users: int, n_items: int, dim: int = 64):
        """Initialize embedding tables and biases.

        Args:
            n_users (int): Maximum user ID in the data.
            n_items (int): Maximum item ID in the data.
            dim (int): Embedding dimension.
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.user_emb = nn.Embedding(n_users + 1, dim)
        self.item_emb = nn.Embedding(n_items + 1, dim)
        self.item_bias = nn.Embedding(n_items + 1, 1)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """Compute scores for (user, item) pairs.

        Args:
            user (torch.Tensor): 1D tensor of user IDs.
            item (torch.Tensor): 1D tensor of item IDs.

        Returns:
            (torch.Tensor): 1D tensor of scores, one per pair.
        """
        user_embed = self.user_emb(user)
        item_embed = self.item_emb(item)
        item_bias = self.item_bias(item).squeeze(-1)
        return (user_embed * item_embed).sum(-1) + item_bias

    @torch.no_grad()
    def score_all_items(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Score every item for each user in the batch.

        Args:
            user_ids (torch.Tensor): 1D tensor of user IDs.

        Returns:
            (torch.Tensor): Shape (B, n_items + 1), scores for all items.
        """
        user_embed = self.user_emb(user_ids)
        scores = user_embed @ self.item_emb.weight.T
        scores = scores + self.item_bias.weight.squeeze(-1)
        return scores


def sample_negatives(users: np.ndarray, user_pos: dict, n_items: int) -> np.ndarray:
    """Sample one negative item per user via rejection sampling.

    Args:
        users (np.ndarray): Array of user IDs, one per training sample.
        user_pos (dict): Mapping of user_id -> set of positive item IDs.
        n_items (int): Total number of items (items are 1-indexed).

    Returns:
        (np.ndarray): Array of negative item IDs, same length as users.
    """
    neg = np.random.randint(1, n_items + 1, size=len(users))
    for i in range(len(users)):
        while neg[i] in user_pos[users[i]]:
            neg[i] = np.random.randint(1, n_items + 1)
    return neg


def train_bpr(train_df: pd.DataFrame, n_users: int, n_items: int, dim: int = 256, n_epochs: int = 100, batch_size: int = 1024,
    lr: float = 5e-4, weight_decay: float = 1e-6, device: str = "cpu", verbose: bool = True) -> BPRMF:
    """Train a BPR-MF model on (user, item) interactions.

    Args:
        train_df (pd.DataFrame): Interactions with columns [user_id, item_id].
        n_users (int): Size of the user embedding table (max user_id in data).
        n_items (int): Size of the item embedding table (max item_id in data).
        dim (int): Embedding dimension.
        n_epochs (int): Number of training passes over the interactions.
        batch_size (int): SGD minibatch size.
        lr (float): Adam learning rate.
        weight_decay (float): L2 regularization coefficient.
        device (str): Torch device string.
        seed (int): Random seed for reproducibility.
        verbose (bool): Print per-epoch loss if True.

    Returns:
        (BPRMF): Trained model on the requested device.
    """

    model = BPRMF(n_users, n_items, dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    user_pos = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    users_arr = train_df["user_id"].values.astype(np.int64)
    pos_arr = train_df["item_id"].values.astype(np.int64)
    n_samples = len(users_arr)

    for epoch in range(n_epochs):
        perm = np.random.permutation(n_samples)
        shuffled_users = users_arr[perm]
        shuffled_pos = pos_arr[perm]
        shuffled_neg = sample_negatives(shuffled_users, user_pos, n_items)

        model.train()
        total_loss = 0.0
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            user_tensor = torch.from_numpy(shuffled_users[start:end]).to(device)
            pos_tensor = torch.from_numpy(shuffled_pos[start:end]).to(device)
            neg_tensor = torch.from_numpy(shuffled_neg[start:end]).to(device)

            pos_scores = model(user_tensor, pos_tensor)
            neg_scores = model(user_tensor, neg_tensor)
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * (end - start)

        if verbose:
            print(f"epoch {epoch + 1:2d}/{n_epochs}: loss = {total_loss / n_samples:.4f}")

    return model


class BPRRecommender:
    """Wraps a trained BPRMF as a `(user_id, seen_items, k) -> list[int]` callable
    compatible with `evaluate.evaluate_model`.
    """

    def __init__(self, model: BPRMF, device: str = "cpu"):
        self.model = model.eval()
        self.device = device


    @torch.no_grad()
    def score_user(self, user_id: int) -> torch.Tensor:
        """Return raw scores over all item indices for a single user.

        Args:
            user_id (int): User to score.

        Returns:
            (torch.Tensor): 1D tensor of length n_items + 1.
        """
        safe_user_id = user_id if user_id < self.model.user_emb.num_embeddings else 0
        user_tensor = torch.tensor([safe_user_id], dtype=torch.long, device=self.device)
        return self.model.score_all_items(user_tensor).squeeze(0).clone()


    @torch.no_grad()
    def batch_score_users(self, user_ids: list[int]) -> torch.Tensor:
        """Return scores for a batch of users.

        Args:
            user_ids (list[int]): User IDs to score. Out-of-range IDs map to index 0.

        Returns:
            (torch.Tensor): Shape (B, n_items + 1).
        """
        n_user_emb = self.model.user_emb.num_embeddings
        safe_user_ids = [uid if uid < n_user_emb else 0 for uid in user_ids]
        user_tensor = torch.tensor(safe_user_ids, dtype=torch.long, device=self.device)
        return self.model.score_all_items(user_tensor)


    def recommend(self, user_id: int, seen_items: set, k: int = 10) -> list[int]:
        """Recommend top-k items for a user, excluding already seen items.

        Args:
            user_id (int): Target user.
            seen_items (set): Item IDs to exclude.
            k (int): Number of items to return.

        Returns:
            (list[int]): Top-k item IDs.
        """
        scores = self.score_user(user_id)
        scores[0] = float("-inf")
        if seen_items:
            seen = torch.tensor(list(seen_items), dtype=torch.long, device=self.device)
            scores[seen] = float("-inf")
        top_k = torch.topk(scores, k).indices.cpu().tolist()
        return top_k


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate import evaluate_model

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")

    n_users = int(max(train_df["user_id"].max(), val_df["user_id"].max()))
    n_items = int(max(train_df["item_id"].max(), val_df["item_id"].max()))
    print(f"users: {n_users}, items: {n_items}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = train_bpr(train_df, n_users, n_items, dim=256, n_epochs=100, batch_size=1024, lr=5e-4, weight_decay=1e-6, device=device)

    rec = BPRRecommender(model, device=device)
    recommend_fn = lambda user_id, seen_items, k: rec.recommend(user_id, seen_items, k)
    evaluate_model(recommend_fn, val_df, train_df, k=10)

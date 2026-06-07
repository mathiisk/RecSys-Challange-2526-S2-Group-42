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

    Score(u, i) = user_emb(u) . item_emb(i) + item_bias(i)
    """

    def __init__(self, n_users: int, n_items: int, dim: int = 64):
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
        u_e = self.user_emb(user)
        i_e = self.item_emb(item)
        i_b = self.item_bias(item).squeeze(-1)
        return (u_e * i_e).sum(-1) + i_b

    @torch.no_grad()
    def score_all_items(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Score every item for each user. Returns (B, n_items + 1)."""
        u_e = self.user_emb(user_ids)
        scores = u_e @ self.item_emb.weight.T
        scores = scores + self.item_bias.weight.squeeze(-1)
        return scores


def sample_negatives(users: np.ndarray, user_pos: dict, n_items: int) -> np.ndarray:
    """Uniform-random negative items, rejection-sampled against each user's positive set."""
    neg = np.random.randint(1, n_items + 1, size=len(users))
    for i in range(len(users)):
        while neg[i] in user_pos[users[i]]:
            neg[i] = np.random.randint(1, n_items + 1)
    return neg


def train_bpr(
    train_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    dim: int = 64,
    n_epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
) -> BPRMF:
    """Train a BPR-MF model on (user, item) interactions.

    Args:
        train_df: Interactions with columns [user_id, item_id, ...].
        n_users: Size for the user embedding table (max user_id seen anywhere).
        n_items: Size for the item embedding table (max item_id seen anywhere).
        dim: Embedding dimension.
        n_epochs: Number of training passes over the interactions.
        batch_size: SGD minibatch size.
        lr: Adam learning rate.
        weight_decay: L2 regularization via Adam weight_decay.
        device: torch device string.
        seed: Random seed for reproducibility.
        verbose: Print per-epoch loss.

    Returns:
        Trained BPRMF model (on the requested device).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = BPRMF(n_users, n_items, dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    user_pos = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    users_arr = train_df["user_id"].values.astype(np.int64)
    pos_arr = train_df["item_id"].values.astype(np.int64)
    n_samples = len(users_arr)

    for epoch in range(n_epochs):
        perm = np.random.permutation(n_samples)
        u_shuf = users_arr[perm]
        p_shuf = pos_arr[perm]
        neg_shuf = sample_negatives(u_shuf, user_pos, n_items)

        model.train()
        total_loss = 0.0
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            u = torch.from_numpy(u_shuf[start:end]).to(device)
            p = torch.from_numpy(p_shuf[start:end]).to(device)
            n = torch.from_numpy(neg_shuf[start:end]).to(device)

            s_pos = model(u, p)
            s_neg = model(u, n)
            loss = -F.logsigmoid(s_pos - s_neg).mean()

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
        """Return raw (un-masked) score for every item index for one user."""
        if user_id >= self.model.user_emb.num_embeddings:
            # Cold user beyond embedding range: fall back to item bias (~ popularity).
            return self.model.item_bias.weight.detach().squeeze(-1).clone()
        u = torch.tensor([user_id], dtype=torch.long, device=self.device)
        return self.model.score_all_items(u).squeeze(0).clone()

    @torch.no_grad()
    def batch_score_users(self, user_ids: list[int]) -> torch.Tensor:
        """Return scores of shape (B, n_items + 1) for the given user_ids."""
        n_user_emb = self.model.user_emb.num_embeddings
        # OOB user_ids map to index 0 (unused init), the post-softmax bias term then
        # dominates and acts as a popularity-like fallback for cold users.
        safe = [uid if uid < n_user_emb else 0 for uid in user_ids]
        u = torch.tensor(safe, dtype=torch.long, device=self.device)
        return self.model.score_all_items(u)

    def recommend(self, user_id: int, seen_items: set, k: int = 10) -> list[int]:
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

    model = train_bpr(
        train_df, n_users, n_items,
        dim=64, n_epochs=20, batch_size=1024,
        lr=1e-3, weight_decay=1e-5, device=device,
    )

    rec = BPRRecommender(model, device=device)
    recommend_fn = lambda u, s, k: rec.recommend(u, s, k)
    evaluate_model(recommend_fn, val_df, train_df, k=10)

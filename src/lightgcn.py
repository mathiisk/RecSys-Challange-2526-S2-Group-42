import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

DATA_DIR = Path("data/processed")


class LightGCN(nn.Module):
    """Light Graph Convolution Network for collaborative filtering.

    Learns user and item embeddings by propagating signals over the
    user-item interaction graph. Final embeddings are the mean of
    representations across all layers (including layer 0).

    IDs are 1-indexed. Index 0 is reserved (padding/cold).
    Score(u, i) = user_emb(u) . item_emb(i)
    """

    def __init__(self, n_users: int, n_items: int, dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers

        self.user_emb = nn.Embedding(n_users + 1, dim)
        self.item_emb = nn.Embedding(n_items + 1, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        with torch.no_grad():
            self.user_emb.weight[0].zero_()
            self.item_emb.weight[0].zero_()

    def forward(self, norm_adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate embeddings over the graph and return layer-mean embeddings.

        Args:
            norm_adjacency (torch.Tensor): Sparse normalised adjacency matrix of
                shape (n_users + n_items + 2, n_users + n_items + 2).

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): user_embeddings (n_users + 1, dim),
                item_embeddings (n_items + 1, dim).
        """
        all_embeddings = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        layer_embeddings = [all_embeddings]

        current = all_embeddings
        for _ in range(self.n_layers):
            current = torch.sparse.mm(norm_adjacency, current)
            layer_embeddings.append(current)

        final_embeddings = torch.stack(layer_embeddings, dim=1).mean(dim=1)
        user_embeddings = final_embeddings[:self.n_users + 1]
        item_embeddings = final_embeddings[self.n_users + 1:]
        return user_embeddings, item_embeddings

    def score_all_items(self, user_ids: torch.Tensor, user_embeddings: torch.Tensor, item_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute dot-product scores between users and all items.

        Args:
            user_ids (torch.Tensor): 1D tensor of user IDs.
            user_embeddings (torch.Tensor): Full user embedding matrix (n_users + 1, dim).
            item_embeddings (torch.Tensor): Full item embedding matrix (n_items + 1, dim).

        Returns:
            (torch.Tensor): Shape (B, n_items + 1).
        """
        user_vecs = user_embeddings[user_ids]
        return user_vecs @ item_embeddings.T


def build_norm_adjacency(train_df: pd.DataFrame, n_users: int, n_items: int, device: str = "cpu") -> torch.Tensor:
    """Build the symmetric normalised adjacency matrix for the user-item graph.

    Constructs a (n_users + n_items + 2) square sparse matrix where
    users occupy rows/cols [0, n_users] and items occupy [n_users+1, n_users+n_items+1].
    Normalisation: D^(-1/2) A D^(-1/2).

    Args:
        train_df (pd.DataFrame): Interactions with columns [user_id, item_id].
        n_users (int): Maximum user ID.
        n_items (int): Maximum item ID.
        device (str): Torch device string.

    Returns:
        (torch.Tensor): Sparse normalised adjacency matrix on the given device.
    """
    user_ids = train_df["user_id"].values.astype(np.int64)
    item_ids = train_df["item_id"].values.astype(np.int64)

    # items are offset by n_users + 1 in the joint adjacency matrix
    item_ids_offset = item_ids + n_users + 1
    n_nodes = n_users + n_items + 2

    # build symmetric edges: (user -> item) and (item -> user)
    row_indices = np.concatenate([user_ids, item_ids_offset])
    col_indices = np.concatenate([item_ids_offset, user_ids])

    indices = torch.tensor(np.stack([row_indices, col_indices]), dtype=torch.long)
    values = torch.ones(len(row_indices), dtype=torch.float32)
    adjacency = torch.sparse_coo_tensor(indices, values, size=(n_nodes, n_nodes))
    adjacency = adjacency.coalesce()

    # degree vector for D^(-1/2) A D^(-1/2) normalisation
    degree = torch.sparse.sum(adjacency, dim=1).to_dense().clamp(min=1)
    degree_inv_sqrt = degree.pow(-0.5)

    norm_values = (degree_inv_sqrt[adjacency.indices()[0]] * adjacency.values() * degree_inv_sqrt[adjacency.indices()[1]])
    norm_adjacency = torch.sparse_coo_tensor(adjacency.indices(), norm_values, size=(n_nodes, n_nodes)).to(device)

    return norm_adjacency


def train_lightgcn(train_df: pd.DataFrame, n_users: int, n_items: int, dim: int = 64, n_layers: int = 2, n_epochs: int = 50, batch_size: int = 1024,
    lr: float = 1e-3, weight_decay: float = 1e-5, device: str = "cpu", verbose: bool = True) -> tuple[LightGCN, torch.Tensor]:
    """Train LightGCN with BPR loss on user-item interactions.

    Args:
        train_df (pd.DataFrame): Interactions with columns [user_id, item_id].
        n_users (int): Maximum user ID in the data.
        n_items (int): Maximum item ID in the data.
        dim (int): Embedding dimension.
        n_layers (int): Number of graph convolution layers.
        n_epochs (int): Number of training epochs.
        batch_size (int): Number of interactions per minibatch.
        lr (float): Adam learning rate.
        weight_decay (float): L2 regularisation coefficient.
        device (str): Torch device string.
        seed (int): Random seed for reproducibility.
        verbose (bool): Print per-epoch loss if True.

    Returns:
        (tuple[LightGCN, torch.Tensor]): Trained model and normalised adjacency matrix.
    """

    model = LightGCN(n_users, n_items, dim, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    norm_adjacency = build_norm_adjacency(train_df, n_users, n_items, device=device)

    user_pos = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    users_arr = train_df["user_id"].values.astype(np.int64)
    pos_arr = train_df["item_id"].values.astype(np.int64)
    n_samples = len(users_arr)

    for epoch in range(n_epochs):
        model.train()
        perm = np.random.permutation(n_samples)
        shuffled_users = users_arr[perm]
        shuffled_pos = pos_arr[perm]
        shuffled_neg = _sample_negatives(shuffled_users, user_pos, n_items)

        total_loss = 0.0
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            user_tensor = torch.from_numpy(shuffled_users[start:end]).to(device)
            pos_tensor = torch.from_numpy(shuffled_pos[start:end]).to(device)
            neg_tensor = torch.from_numpy(shuffled_neg[start:end]).to(device)

            user_embeddings, item_embeddings = model(norm_adjacency)

            pos_scores = (user_embeddings[user_tensor] * item_embeddings[pos_tensor]).sum(-1)
            neg_scores = (user_embeddings[user_tensor] * item_embeddings[neg_tensor]).sum(-1)
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * (end - start)

        if verbose:
            print(f"epoch {epoch + 1:2d}/{n_epochs}: loss = {total_loss / n_samples:.4f}")

    return model, norm_adjacency


def _sample_negatives(users: np.ndarray, user_pos: dict, n_items: int) -> np.ndarray:
    """Sample one negative item per user via rejection sampling.

    Args:
        users (np.ndarray): Array of user IDs, one per training sample.
        user_pos (dict): Mapping of user_id -> set of positive item IDs.
        n_items (int): Total number of items (items are 1-indexed).

    Returns:
        (np.ndarray): Array of negative item IDs, same length as users.
    """
    negatives = np.random.randint(1, n_items + 1, size=len(users))
    for i in range(len(users)):
        while negatives[i] in user_pos[users[i]]:
            negatives[i] = np.random.randint(1, n_items + 1)
    return negatives


class LightGCNRecommender:
    """Wraps a trained LightGCN as a (user_id, seen_items, k) -> list[int] callable
    compatible with evaluate.evaluate_model and the ensemble.
    """

    def __init__(self, model: LightGCN, norm_adjacency: torch.Tensor, device: str = "cpu"):
        self.model = model.eval()
        self.device = device
        with torch.no_grad():
            self.user_embeddings, self.item_embeddings = model(norm_adjacency)

    def score_user(self, user_id: int) -> torch.Tensor:
        """Return raw scores over all item indices for a single user.

        Args:
            user_id (int): User to score. Falls back to index 0 if out of range.

        Returns:
            (torch.Tensor): 1D tensor of length n_items + 1.
        """
        safe_user_id = user_id if user_id < self.model.user_emb.num_embeddings else 0
        user_tensor = torch.tensor([safe_user_id], dtype=torch.long, device=self.device)
        return self.model.score_all_items(user_tensor, self.user_embeddings, self.item_embeddings).squeeze(0).clone()

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
        return self.model.score_all_items(user_tensor, self.user_embeddings, self.item_embeddings)

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
        return torch.topk(scores, k).indices.cpu().tolist()


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

    model, norm_adjacency = train_lightgcn(
        train_df, n_users, n_items,
        dim=64, n_layers=2, n_epochs=50, batch_size=1024,
        lr=1e-3, weight_decay=1e-5, device=device,
    )

    rec = LightGCNRecommender(model, norm_adjacency, device=device)
    recommend_fn = lambda user_id, seen_items, k: rec.recommend(user_id, seen_items, k)
    evaluate_model(recommend_fn, val_df, train_df, k=10)
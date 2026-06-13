import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence

DATA_DIR = Path("data/processed")


class GRU4Rec(nn.Module):
    """GRU-based sequential recommender.

    Predicts the next item from a sequence of interactions ordered by timestamp.
    Output logits = projected hidden state @ item_emb.T (tied embeddings) + item_bias.
    Index 0 in the item table is reserved for padding.
    """

    def __init__(self, n_items: int, embed_dim: int = 64, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.3):
        """Initialize embedding tables, GRU, and output projection.

        Args:
            n_items (int): Maximum item ID in the data.
            embed_dim (int): Item embedding dimension.
            hidden_dim (int): GRU hidden state dimension.
            num_layers (int): Number of GRU layers.
            dropout (float): Dropout probability applied to embeddings and between GRU layers.
        """
        super().__init__()
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.item_bias = nn.Embedding(n_items + 1, 1, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, embed_dim)

        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def _logits_from_hidden(self, h_proj: torch.Tensor) -> torch.Tensor:
        """Apply the tied output projection and item bias to get item logits.

        Args:
            h_proj (torch.Tensor): Projected hidden states, shape (..., embed_dim).

        Returns:
            (torch.Tensor): Logits over all items, shape (..., n_items + 1).
        """
        return h_proj @ self.item_emb.weight.T + self.item_bias.weight.squeeze(-1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Compute next-item logits at every position of the input sequences.

        Args:
            seq (torch.Tensor): Item index sequences, shape (B, L).

        Returns:
            (torch.Tensor): Logits at every position, shape (B, L, n_items + 1).
        """
        x = self.item_emb(seq)
        x = self.dropout(x)
        h, _ = self.gru(x)
        h = self.proj(h)
        return self._logits_from_hidden(h)

    @torch.no_grad()
    def score_last(self, seq: torch.Tensor) -> torch.Tensor:
        """Score every item using the hidden state from the end of a single sequence.

        Args:
            seq (torch.Tensor): Item index sequence with no padding, shape (1, L).

        Returns:
            (torch.Tensor): 1D tensor of length n_items + 1.
        """
        x = self.item_emb(seq)
        h, _ = self.gru(x)
        h_proj = self.proj(h[:, -1, :])
        return self._logits_from_hidden(h_proj).squeeze(0)


def build_user_sequences(train_df: pd.DataFrame, max_seq_len: int = 100) -> dict[int, list[int]]:
    """Build per-user item sequences ordered by timestamp, capped to the most recent items.

    Args:
        train_df (pd.DataFrame): Interactions with columns [user_id, item_id, timestamp].
        max_seq_len (int): Maximum sequence length; longer sequences keep only the most recent items.

    Returns:
        (dict[int, list[int]]): Mapping of user_id -> ordered list of item IDs.
    """
    sequences: dict[int, list[int]] = {}
    sorted_df = train_df.sort_values("timestamp")
    for user_id, group in sorted_df.groupby("user_id"):
        items = group["item_id"].tolist()
        if len(items) > max_seq_len:
            items = items[-max_seq_len:]
        sequences[int(user_id)] = items
    return sequences


def train_gru4rec(train_df: pd.DataFrame, n_items: int, embed_dim: int = 64, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.3, max_seq_len: int = 100,
    n_epochs: int = 100, batch_size: int = 1024, lr: float = 5e-4, weight_decay: float = 1e-6, device: str = "cpu", verbose: bool = True) -> tuple[GRU4Rec, dict[int, list[int]]]:
    """Train a GRU4Rec model with next-item cross-entropy loss.

    Args:
        train_df (pd.DataFrame): Interactions with columns [user_id, item_id, timestamp].
        n_items (int): Size of the item embedding table (max item_id in data).
        embed_dim (int): Item embedding dimension.
        hidden_dim (int): GRU hidden state dimension.
        num_layers (int): Number of GRU layers.
        dropout (float): Dropout probability.
        max_seq_len (int): Maximum per-user sequence length.
        n_epochs (int): Number of training epochs.
        batch_size (int): Number of user sequences per minibatch.
        lr (float): Adam learning rate.
        weight_decay (float): L2 regularization coefficient.
        device (str): Torch device string.
        verbose (bool): Print per-epoch loss if True.

    Returns:
        (tuple[GRU4Rec, dict[int, list[int]]]): Trained model and full per-user sequences,
            reused at inference.
    """
    sequences = build_user_sequences(train_df, max_seq_len=max_seq_len)
    train_seqs = {u: s for u, s in sequences.items() if len(s) >= 2}
    train_users = list(train_seqs.keys())

    model = GRU4Rec(n_items, embed_dim, hidden_dim, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_samples = len(train_users)

    for epoch in range(n_epochs):

        model.train()
        total_loss = 0.0
        total_tokens = 0
        for start in range(0, n_samples, batch_size):
            batch_users = train_users[start : start + batch_size]
            batch_seqs = [train_seqs[u] for u in batch_users]

            inp = [torch.tensor(s[:-1], dtype=torch.long) for s in batch_seqs]
            tgt = [torch.tensor(s[1:], dtype=torch.long) for s in batch_seqs]
            inp = pad_sequence(inp, batch_first=True, padding_value=0).to(device)
            tgt = pad_sequence(tgt, batch_first=True, padding_value=0).to(device)

            logits = model(inp)
            loss = F.cross_entropy(logits.view(-1, n_items + 1), tgt.view(-1), ignore_index=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_tok = (tgt != 0).sum().item()
            total_loss += loss.item() * n_tok
            total_tokens += n_tok

        if verbose:
            print(
                f"epoch {epoch + 1:2d}/{n_epochs}: loss = {total_loss / total_tokens:.4f} "
                f"({n_samples} users, {total_tokens} tokens)"
            )

    return model, sequences


class GRU4RecRecommender:
    """Wraps a trained GRU4Rec as a `(user_id, seen_items, k) -> list[int]` callable
    compatible with `evaluate.evaluate_model` and the ensemble.
    """

    def __init__(self, model: GRU4Rec, sequences: dict[int, list[int]], device: str = "cpu"):
        self.model = model.eval()
        self.sequences = sequences
        self.device = device

    @torch.no_grad()
    def score_user(self, user_id: int) -> torch.Tensor:
        """Return raw scores over all item indices for a single user.

        Cold users with no train history are scored by feeding a single padding
        token through the GRU, matching the formula used in `batch_score_users`.

        Args:
            user_id (int): User to score.

        Returns:
            (torch.Tensor): 1D tensor of length n_items + 1.
        """
        seq = self.sequences.get(int(user_id), []) or [0]
        seq_t = torch.tensor([seq], dtype=torch.long, device=self.device)
        return self.model.score_last(seq_t).clone()

    @torch.no_grad()
    def batch_score_users(self, user_ids: list[int]) -> torch.Tensor:
        """Return scores for a batch of users.

        Empty-sequence users are scored with a single padding token so the whole
        batch shares one GRU forward, giving the same "cold prior" as `score_user`.

        Args:
            user_ids (list[int]): User IDs to score.

        Returns:
            (torch.Tensor): Shape (B, n_items + 1).
        """
        seqs: list[list[int]] = []
        lengths: list[int] = []
        for uid in user_ids:
            s = self.sequences.get(int(uid), []) or [0]
            seqs.append(s)
            lengths.append(len(s))
        max_len = max(lengths)
        padded = torch.zeros(len(seqs), max_len, dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            padded[i, : lengths[i]] = torch.tensor(s, dtype=torch.long)
        lengths_t = torch.tensor(lengths, dtype=torch.long, device=self.device)

        x = self.model.item_emb(padded)            # (B, L, E)
        h, _ = self.model.gru(x)                    # (B, L, H)
        idx = (lengths_t - 1).view(-1, 1, 1).expand(-1, 1, h.size(-1))
        h_last = h.gather(1, idx).squeeze(1)        # (B, H)
        h_proj = self.model.proj(h_last)            # (B, E)
        return self.model._logits_from_hidden(h_proj)  # (B, V)

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

    n_items = int(max(train_df["item_id"].max(), val_df["item_id"].max()))
    print(f"items: {n_items}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model, sequences = train_gru4rec(
        train_df, n_items,
        embed_dim=64, hidden_dim=64, num_layers=1, dropout=0.3,
        max_seq_len=100, n_epochs=100, batch_size=1024,
        lr=5e-4, weight_decay=1e-6, device=device,
    )

    rec = GRU4RecRecommender(model, sequences, device=device)
    recommend_fn = lambda u, s, k: rec.recommend(u, s, k)
    evaluate_model(recommend_fn, val_df, train_df, k=10)
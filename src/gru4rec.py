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

    def __init__(
        self,
        n_items: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.item_bias = nn.Embedding(n_items + 1, 1, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, embed_dim)

        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def _logits_from_hidden(self, h_proj: torch.Tensor) -> torch.Tensor:
        """Apply tied output + item bias. h_proj: (..., embed_dim) -> (..., n_items+1)."""
        return h_proj @ self.item_emb.weight.T + self.item_bias.weight.squeeze(-1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) item indices. Returns: (B, L, n_items+1) logits at every position."""
        x = self.item_emb(seq)
        x = self.dropout(x)
        h, _ = self.gru(x)
        h = self.proj(h)
        return self._logits_from_hidden(h)

    @torch.no_grad()
    def score_last(self, seq: torch.Tensor) -> torch.Tensor:
        """Score every item from a single sequence's last hidden state.

        seq: (1, L) item indices, no padding. Returns: (n_items+1,) scores.
        """
        x = self.item_emb(seq)
        h, _ = self.gru(x)
        h_proj = self.proj(h[:, -1, :])
        return self._logits_from_hidden(h_proj).squeeze(0)


def build_user_sequences(
    train_df: pd.DataFrame, max_seq_len: int = 50
) -> dict[int, list[int]]:
    """Per-user item sequence sorted by timestamp, capped at max_seq_len (keep most recent)."""
    sequences: dict[int, list[int]] = {}
    sorted_df = train_df.sort_values("timestamp")
    for user_id, group in sorted_df.groupby("user_id"):
        items = group["item_id"].tolist()
        if len(items) > max_seq_len:
            items = items[-max_seq_len:]
        sequences[int(user_id)] = items
    return sequences


def train_gru4rec(
    train_df: pd.DataFrame,
    n_items: int,
    embed_dim: int = 64,
    hidden_dim: int = 64,
    num_layers: int = 1,
    dropout: float = 0.2,
    max_seq_len: int = 50,
    n_epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[GRU4Rec, dict[int, list[int]]]:
    """Train GRU4Rec with next-item cross-entropy.

    Returns: (model, full per-user sequences) — sequences are reused at inference.
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
            loss = F.cross_entropy(
                logits.view(-1, n_items + 1),
                tgt.view(-1),
                ignore_index=0,
            )

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
    """Wraps a trained GRU4Rec as a `(user_id, seen_items, k) -> list[int]` callable."""

    def __init__(
        self,
        model: GRU4Rec,
        sequences: dict[int, list[int]],
        device: str = "cpu",
    ):
        self.model = model.eval()
        self.sequences = sequences
        self.device = device

    @torch.no_grad()
    def score_user(self, user_id: int) -> torch.Tensor:
        """Return raw (un-masked) score for every item index for one user.

        Cold users (no train history) get the same formula as the batched path:
        feed a single padding token through the GRU and use its projected output.
        """
        seq = self.sequences.get(int(user_id), []) or [0]
        seq_t = torch.tensor([seq], dtype=torch.long, device=self.device)
        return self.model.score_last(seq_t).clone()

    @torch.no_grad()
    def batch_score_users(self, user_ids: list[int]) -> torch.Tensor:
        """Return scores of shape (B, n_items + 1) for the given user_ids.

        Empty-sequence users are scored with a single padding token so the whole
        batch shares one GRU forward — this gives a "cold prior" identical to
        what `score_user` returns for the same user.
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
        embed_dim=64, hidden_dim=64, num_layers=1, dropout=0.2,
        max_seq_len=50, n_epochs=20, batch_size=256,
        lr=1e-3, weight_decay=1e-5, device=device,
    )

    rec = GRU4RecRecommender(model, sequences, device=device)
    recommend_fn = lambda u, s, k: rec.recommend(u, s, k)
    evaluate_model(recommend_fn, val_df, train_df, k=10)

import argparse
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from popularity import compute_item_popularity, PopularityRecommender
from bpr import train_bpr, BPRRecommender, BPRMF
from gru4rec import train_gru4rec, GRU4RecRecommender, GRU4Rec
from lightgcn import train_lightgcn, LightGCNRecommender, LightGCN
from ensemble import chunked_ensemble_sweep, DEFAULT_WEIGHT_GRID


DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"


def save_checkpoint(checkpoint_dir: Path, bpr_model: BPRMF, gru_model: GRU4Rec, gru_sequences: dict[int, list[int]], lightgcn_model: LightGCN,
    norm_adjacency: torch.Tensor, popularity: object, n_users: int, n_items: int) -> None:
    """Save trained models and popularity scores to disk.

    Args:
        checkpoint_dir (Path): Directory to write checkpoint files.
        bpr_model (BPRMF): Trained BPR-MF model.
        gru_model (GRU4Rec): Trained GRU4Rec model.
        gru_sequences (dict[int, list[int]]): Per-user item sequences used by GRU4Rec.
        popularity (object): Popularity array from compute_item_popularity.
        n_users (int): Number of users the models were trained with.
        n_items (int): Number of items the models were trained with.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": bpr_model.state_dict(), "n_users": n_users, "n_items": n_items, "dim": bpr_model.user_emb.embedding_dim}, checkpoint_dir / "bpr.pt")
    torch.save({"state_dict": gru_model.state_dict(), "n_items": n_items, "embed_dim": gru_model.embed_dim, "hidden_dim": gru_model.hidden_dim}, checkpoint_dir / "gru.pt")
    torch.save({"state_dict": lightgcn_model.state_dict(), "n_users": n_users, "n_items": n_items, "dim": lightgcn_model.user_emb.embedding_dim, "n_layers": lightgcn_model.n_layers,}, checkpoint_dir / "lightgcn.pt")
    torch.save(norm_adjacency, checkpoint_dir / "norm_adjacency.pt")
    with open(checkpoint_dir / "gru_sequences.pkl", "wb") as f:
        pickle.dump(gru_sequences, f)
    with open(checkpoint_dir / "popularity.pkl", "wb") as f:
        pickle.dump((popularity, n_items), f)
    print(f"saved checkpoint to {checkpoint_dir}")


def main(bpr_epochs: int = 150, gru_epochs: int = 150, lightgcn_epochs: int = 20, bpr_dim: int = 64, gru_embed_dim: int = 64, gru_hidden_dim: int = 64,
    gru_max_seq_len: int = 50, lightgcn_dim: int = 64, save_dir: Path | None = None, load_dir: Path | None = None) -> None:
    """Train models on processed/train, sweep weights on processed/val, optionally save.

    Args:
        bpr_epochs (int): Number of BPR-MF training epochs.
        gru_epochs (int): Number of GRU4Rec training epochs.
        bpr_dim (int): BPR-MF embedding dimension.
        gru_embed_dim (int): GRU4Rec item embedding dimension.
        gru_hidden_dim (int): GRU4Rec hidden state dimension.
        gru_max_seq_len (int): Maximum sequence length for GRU4Rec.
        save_dir (Path | None): If given, save checkpoint here after training.
        seed (int): Random seed for reproducibility.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")

    n_users = int(max(train_df["user_id"].max(), val_df["user_id"].max()))
    n_items = int(max(train_df["item_id"].max(), val_df["item_id"].max()))
    print(f"users: {n_users}, items: {n_items}, device: {device}")

    # ====================================================================BPR 
    t0 = time.time()
    print(f"\n[1/4] training BPR-MF ({bpr_epochs} epochs)...")
    bpr_model = train_bpr(train_df, n_users, n_items,
        dim=bpr_dim, n_epochs=bpr_epochs, batch_size=1024,
        lr=1e-3, weight_decay=1e-5, device=device, verbose=True,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    # ===================================================================GRU4REC
    t0 = time.time()
    print(f"[2/4] training GRU4Rec ({gru_epochs} epochs)...")
    gru_model, gru_sequences = train_gru4rec(
        train_df, n_items,
        embed_dim=gru_embed_dim, hidden_dim=gru_hidden_dim,
        num_layers=1, dropout=0.2,
        max_seq_len=gru_max_seq_len, n_epochs=gru_epochs, batch_size=256,
        lr=1e-3, weight_decay=1e-5, device=device, verbose=True,
    )
    print(f"  done in {time.time() - t0:.1f}s")
    
    # ==================================================================LIGHTGCN
    t0 = time.time()
    print(f"[3/4] training LightGCN ({lightgcn_epochs} epochs)...")
    lightgcn_model, norm_adjacency = train_lightgcn(
        train_df, n_users, n_items,
        dim=lightgcn_dim, n_layers=3, n_epochs=lightgcn_epochs, batch_size=1024,
        lr=1e-3, weight_decay=1e-5, device=device, verbose=True,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    # ==================================================================POPULARITY
    print("[4/4] computing popularity...")
    popularity = compute_item_popularity(train_df, n_items=n_items)

    if save_dir is not None: 
        save_checkpoint(save_dir, bpr_model, gru_model, gru_sequences, lightgcn_model, norm_adjacency, popularity, n_users, n_items)

    bpr_rec = BPRRecommender(bpr_model, device=device)
    gru_rec = GRU4RecRecommender(gru_model, gru_sequences, device=device)
    lightgcn_rec = LightGCNRecommender(lightgcn_model, norm_adjacency, device=device)
    pop_rec = PopularityRecommender(popularity, n_items, device=device)
    components = [("bpr", bpr_rec), ("gru", gru_rec), ("lightgcn", lightgcn_rec), ("pop", pop_rec)]
    
    print("\nrunning ensemble weight sweep on validation set...")
    t0 = time.time()
    recalls = chunked_ensemble_sweep(
        components, DEFAULT_WEIGHT_GRID, val_df, train_df,
        k=10, batch_size=1024, device=device,
    )
    print(f"sweep done in {time.time() - t0:.1f}s\n")

    best_index = int(max(range(len(recalls)), key=lambda i: recalls[i]))
    print("bpr | gru | pop   recall@10")
    for weights, recall in zip(DEFAULT_WEIGHT_GRID, recalls):
        marker = " <-- best" if weights == DEFAULT_WEIGHT_GRID[best_index] else ""
        weight_str = " | ".join(f"{w:>3.1f}" for w in weights)
        print(f"{weight_str}   {recall:.4f}{marker}")

    best_weights = DEFAULT_WEIGHT_GRID[best_index]
    print(f"\nbest weights: --weights {','.join(str(w) for w in best_weights)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bpr-epochs", type=int, default=100)
    parser.add_argument("--gru-epochs", type=int, default=100)
    parser.add_argument("--lightgcn-epochs", type=int, default=20)
    parser.add_argument("--bpr-dim", type=int, default=256)
    parser.add_argument("--gru-embed-dim", type=int, default=64)
    parser.add_argument("--gru-hidden-dim", type=int, default=64)
    parser.add_argument("--gru-max-seq-len", type=int, default=100)
    parser.add_argument("--lightgcn-dim", type=int, default=128)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--load", type=Path, default=None)
    args = parser.parse_args()

    main(
        bpr_epochs=args.bpr_epochs,
        gru_epochs=args.gru_epochs,
        # lightgcn_epochs=args.lightgcn_epochs,
        bpr_dim=args.bpr_dim,
        gru_embed_dim=args.gru_embed_dim,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_max_seq_len=args.gru_max_seq_len,
        # lightgcn_dim=args.lightgcn_dim,      
        save_dir=args.save,
    )
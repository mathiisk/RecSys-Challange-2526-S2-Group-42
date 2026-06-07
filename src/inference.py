"""Train on train+val and write the Kaggle submission CSV.

Targets the 2,255 users in `data/sample_submission.csv` using the
BPR + GRU + popularity ensemble.

CLI:
    python -m src.inference                           # defaults
    python -m src.inference --bpr-epochs 40 --gru-epochs 25
    python -m src.inference --weights 1.0,2.0,0.5     # bpr,gru,pop
    python -m src.inference --output data/sub_v2.csv
    python -m src.inference --load checkpoints/v1     # skip training, reuse saved models
    python -m src.inference --save checkpoints/v1     # save after training
"""
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
from ensemble import minmax_normalize_rows, build_seen_mask

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
DEFAULT_OUTPUT = DATA_DIR / "submission.csv"


def load_full_training_data() -> pd.DataFrame:
    """Concatenate processed train + val into the deduped full-history train set."""
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")
    return pd.concat([train_df, val_df], ignore_index=True)


def compute_dimensions(*dfs: pd.DataFrame) -> tuple[int, int]:
    """Return (n_users, n_items) sized to cover the union of provided frames.

    Frames whose `item_id` column is a string of comma-separated ids (e.g. the
    submission template) contribute their user range but not their item range.
    """
    n_users = 0
    n_items = 0
    for df in dfs:
        if "user_id" in df.columns:
            n_users = max(n_users, int(df["user_id"].max()))
        if "item_id" in df.columns and df["item_id"].dtype != object:
            n_items = max(n_items, int(df["item_id"].max()))
    return n_users, n_items


def batch_predict_topk(
    components: list[tuple[str, object]],
    weights: tuple[float, ...],
    user_ids: list[int],
    seen_per_user: dict[int, set],
    k: int = 10,
    batch_size: int = 1024,
    device: str = "cpu",
) -> dict[int, list[int]]:
    """Run the ensemble in chunks and return user_id -> top-k item_id list (1-indexed)."""
    out: dict[int, list[int]] = {}
    n = len(user_ids)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_users = user_ids[start:end]

        per_comp = []
        for _, rec in components:
            s = rec.batch_score_users(batch_users).float().to(device)
            per_comp.append(minmax_normalize_rows(s))

        mask = build_seen_mask(batch_users, seen_per_user, per_comp[0])

        combined = torch.zeros_like(per_comp[0])
        for w, s in zip(weights, per_comp):
            if w != 0:
                combined = combined + w * s
        combined = combined + mask

        top_k = torch.topk(combined, k, dim=-1).indices.cpu().numpy()
        for i, uid in enumerate(batch_users):
            out[uid] = top_k[i].tolist()

    return out


def write_submission(
    submission_template: pd.DataFrame,
    predictions: dict[int, list[int]],
    output_path: Path,
) -> None:
    """Match sample_submission row order and write `ID,user_id,"i1,...,iK"`."""
    rows = []
    for _, row in submission_template.iterrows():
        uid = int(row["user_id"])
        items = predictions.get(uid)
        assert items is not None, f"missing prediction for user_id={uid}"
        rows.append({
            "ID": int(row["ID"]),
            "user_id": uid,
            "item_id": ",".join(str(i) for i in items),
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_checkpoint(
    checkpoint_dir: Path,
    bpr_model: BPRMF,
    gru_model: GRU4Rec,
    gru_sequences: dict[int, list[int]],
    popularity,
    n_users: int,
    n_items: int,
) -> None:
    """Persist the trained models + popularity so inference can re-run without training."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": bpr_model.state_dict(),
        "n_users": n_users,
        "n_items": n_items,
        "dim": bpr_model.user_emb.embedding_dim,
    }, checkpoint_dir / "bpr.pt")
    torch.save({
        "state_dict": gru_model.state_dict(),
        "n_items": n_items,
        "embed_dim": gru_model.embed_dim,
        "hidden_dim": gru_model.hidden_dim,
    }, checkpoint_dir / "gru.pt")
    with open(checkpoint_dir / "gru_sequences.pkl", "wb") as f:
        pickle.dump(gru_sequences, f)
    with open(checkpoint_dir / "popularity.pkl", "wb") as f:
        pickle.dump((popularity, n_items), f)


def load_checkpoint(checkpoint_dir: Path, device: str) -> tuple:
    """Load models saved by `save_checkpoint`. Returns (bpr, gru, sequences, popularity, n_users, n_items)."""
    bpr_blob = torch.load(checkpoint_dir / "bpr.pt", map_location=device, weights_only=False)
    bpr = BPRMF(bpr_blob["n_users"], bpr_blob["n_items"], dim=bpr_blob["dim"]).to(device)
    bpr.load_state_dict(bpr_blob["state_dict"])

    gru_blob = torch.load(checkpoint_dir / "gru.pt", map_location=device, weights_only=False)
    gru = GRU4Rec(
        gru_blob["n_items"],
        embed_dim=gru_blob["embed_dim"],
        hidden_dim=gru_blob["hidden_dim"],
    ).to(device)
    gru.load_state_dict(gru_blob["state_dict"])

    with open(checkpoint_dir / "gru_sequences.pkl", "rb") as f:
        sequences = pickle.load(f)
    with open(checkpoint_dir / "popularity.pkl", "rb") as f:
        popularity, n_items = pickle.load(f)

    return bpr, gru, sequences, popularity, bpr_blob["n_users"], n_items


def main(
    bpr_epochs: int = 30,
    gru_epochs: int = 20,
    bpr_dim: int = 64,
    gru_embed_dim: int = 64,
    gru_hidden_dim: int = 64,
    gru_max_seq_len: int = 50,
    weights: tuple[float, float, float] = (1.0, 1.0, 0.0),
    output_path: Path = DEFAULT_OUTPUT,
    k: int = 10,
    seed: int = 42,
    load_dir: Path | None = None,
    save_dir: Path | None = None,
) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("loading data...")
    full_train = load_full_training_data()
    raw_test = pd.read_csv(DATA_DIR / "test.csv")
    submission_template = pd.read_csv(DATA_DIR / "sample_submission.csv")

    n_users, n_items = compute_dimensions(full_train, raw_test, submission_template)
    print(f"  full train: {len(full_train):,} interactions, "
          f"{full_train['user_id'].nunique():,} users")
    print(f"  submission users: {submission_template['user_id'].nunique():,}")
    print(f"  embedding sizing: n_users={n_users}, n_items={n_items}, device={device}")

    if load_dir is not None:
        print(f"\nloading models from {load_dir}...")
        bpr_model, gru_model, sequences, popularity, ck_n_users, ck_n_items = (
            load_checkpoint(load_dir, device=device)
        )
        n_users = max(n_users, ck_n_users)
        n_items = max(n_items, ck_n_items)
    else:
        t0 = time.time()
        print(f"\n[1/3] training BPR-MF for {bpr_epochs} epochs on full data...")
        bpr_model = train_bpr(
            full_train, n_users, n_items,
            dim=bpr_dim, n_epochs=bpr_epochs, batch_size=1024,
            lr=1e-3, weight_decay=1e-5, device=device, seed=seed, verbose=False,
        )
        print(f"  done in {time.time() - t0:.1f}s")

        t0 = time.time()
        print(f"[2/3] training GRU4Rec for {gru_epochs} epochs on full data...")
        gru_model, sequences = train_gru4rec(
            full_train, n_items,
            embed_dim=gru_embed_dim, hidden_dim=gru_hidden_dim,
            num_layers=1, dropout=0.2,
            max_seq_len=gru_max_seq_len, n_epochs=gru_epochs, batch_size=256,
            lr=1e-3, weight_decay=1e-5, device=device, seed=seed, verbose=False,
        )
        print(f"  done in {time.time() - t0:.1f}s")

        print("[3/3] computing popularity...")
        popularity = compute_item_popularity(full_train, n_items=n_items)

        if save_dir is not None:
            print(f"saving checkpoints to {save_dir}...")
            save_checkpoint(save_dir, bpr_model, gru_model, sequences, popularity, n_users, n_items)

    bpr_rec = BPRRecommender(bpr_model, device=device)
    gru_rec = GRU4RecRecommender(gru_model, sequences, device=device)
    pop_rec = PopularityRecommender(popularity, n_items, device=device)

    print(f"\npredicting top-{k} for submission users with weights "
          f"BPR={weights[0]}, GRU={weights[1]}, POP={weights[2]}...")
    t0 = time.time()
    components = [("bpr", bpr_rec), ("gru", gru_rec), ("pop", pop_rec)]
    submission_users = submission_template["user_id"].astype(int).tolist()
    seen_per_user = full_train.groupby("user_id")["item_id"].apply(set).to_dict()

    predictions = batch_predict_topk(
        components, weights, submission_users, seen_per_user,
        k=k, batch_size=1024, device=device,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    print(f"\nwriting submission to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission(submission_template, predictions, output_path)
    print(f"wrote {len(submission_template)} rows to {output_path}")
    return output_path


def parse_weights(s: str) -> tuple[float, float, float]:
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected 3 comma-separated weights, got: {s!r}")
    return tuple(float(p) for p in parts)  # type: ignore[return-value]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bpr-epochs", type=int, default=30)
    p.add_argument("--gru-epochs", type=int, default=20)
    p.add_argument("--bpr-dim", type=int, default=64)
    p.add_argument("--gru-embed-dim", type=int, default=64)
    p.add_argument("--gru-hidden-dim", type=int, default=64)
    p.add_argument("--gru-max-seq-len", type=int, default=50)
    p.add_argument("--weights", type=parse_weights, default=(1.0, 1.0, 0.0),
                   help="comma-separated bpr,gru,pop weights")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--save", type=Path, default=None, help="save trained models to dir")
    p.add_argument("--load", type=Path, default=None, help="load trained models from dir (skips training)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    main(
        bpr_epochs=args.bpr_epochs,
        gru_epochs=args.gru_epochs,
        bpr_dim=args.bpr_dim,
        gru_embed_dim=args.gru_embed_dim,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_max_seq_len=args.gru_max_seq_len,
        weights=args.weights,
        output_path=args.output,
        seed=args.seed,
        load_dir=args.load,
        save_dir=args.save,
    )

"""Load trained models from checkpoint and generate Kaggle submission CSV.

    python inference.py --load checkpoints/v1 --weights 1.0,2.0,0.5
    python inference.py --load checkpoints/v1 --weights 1.0,1.0,0.0 --output data/sub_v2.csv
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from popularity import PopularityRecommender
from bpr import BPRRecommender, BPRMF
from gru4rec import GRU4RecRecommender, GRU4Rec
from lightgcn import LightGCNRecommender, LightGCN
from ensemble import minmax_normalize_rows, build_seen_mask

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
PROCESSED_DIR = DATA_DIR / "processed"
DEFAULT_OUTPUT = OUTPUT_DIR / "submission.csv"


def load_checkpoint(checkpoint_dir: Path, device: str) -> tuple[BPRMF, GRU4Rec, dict[int, list[int]], LightGCN, torch.Tensor, object, int, int]:
    """Load models and popularity saved by train.py.

    Args:
        checkpoint_dir (Path): Directory containing checkpoint files.
        device (str): Torch device string to load models onto.

    Returns:
        (tuple): bpr_model, gru_model, gru_sequences, popularity, n_users, n_items.
    """
    bpr_blob = torch.load(checkpoint_dir / "bpr.pt", map_location=device, weights_only=False)
    bpr_model = BPRMF(bpr_blob["n_users"], bpr_blob["n_items"], dim=bpr_blob["dim"]).to(device)
    bpr_model.load_state_dict(bpr_blob["state_dict"])

    gru_blob = torch.load(checkpoint_dir / "gru.pt", map_location=device, weights_only=False)
    gru_model = GRU4Rec(gru_blob["n_items"], embed_dim=gru_blob["embed_dim"], hidden_dim=gru_blob["hidden_dim"]).to(device)
    gru_model.load_state_dict(gru_blob["state_dict"])
    
    lightgcn_blob = torch.load(checkpoint_dir / "lightgcn.pt", map_location=device, weights_only=False)
    lightgcn_model = LightGCN(lightgcn_blob["n_users"], lightgcn_blob["n_items"], dim=lightgcn_blob["dim"], n_layers=lightgcn_blob["n_layers"]).to(device)
    lightgcn_model.load_state_dict(lightgcn_blob["state_dict"])
    norm_adjacency = torch.load(checkpoint_dir / "norm_adjacency.pt", map_location=device, weights_only=False)

    with open(checkpoint_dir / "gru_sequences.pkl", "rb") as f:
        gru_sequences = pickle.load(f)
    with open(checkpoint_dir / "popularity.pkl", "rb") as f:
        popularity, n_items = pickle.load(f)

    return bpr_model, gru_model, gru_sequences, lightgcn_model, norm_adjacency, popularity, bpr_blob["n_users"], n_items


def batch_predict_topk(components: list[tuple[str, object]], weights: tuple[float, ...], user_ids: list[int],
    seen_per_user: dict[int, set], k: int = 10, batch_size: int = 1024, device: str = "cpu") -> dict[int, list[int]]:
    """Run ensemble inference in batches and return top-k items per user.

    Args:
        components (list[tuple[str, object]]): Named recommender components.
        weights (tuple[float, ...]): Per-component ensemble weights.
        user_ids (list[int]): Users to generate recommendations for.
        seen_per_user (dict[int, set]): Training items to exclude per user.
        k (int): Number of items to recommend per user.
        batch_size (int): Number of users to score per batch.
        device (str): Torch device string.

    Returns:
        (dict[int, list[int]]): Mapping of user_id -> top-k item IDs.
    """
    predictions: dict[int, list[int]] = {}

    for start in range(0, len(user_ids), batch_size):
        batch_users = user_ids[start:start + batch_size]

        component_scores = []
        for _, recommender in components:
            scores = recommender.batch_score_users(batch_users).float().to(device)
            component_scores.append(minmax_normalize_rows(scores))

        seen_mask = build_seen_mask(batch_users, seen_per_user, component_scores[0])

        combined_scores = torch.zeros_like(component_scores[0])
        for weight, scores in zip(weights, component_scores):
            if weight != 0:
                combined_scores = combined_scores + weight * scores
        combined_scores = combined_scores + seen_mask

        top_k_indices = torch.topk(combined_scores, k, dim=-1).indices.cpu().numpy()
        for i, user_id in enumerate(batch_users):
            predictions[user_id] = top_k_indices[i].tolist()

    return predictions


def write_submission(submission_template: pd.DataFrame, predictions: dict[int, list[int]], output_path: Path) -> None:
    """Write predictions to CSV in Kaggle submission format.

    Args:
        submission_template (pd.DataFrame): Sample submission with ID and user_id columns.
        predictions (dict[int, list[int]]): Mapping of user_id -> top-k item IDs.
        output_path (Path): Path to write the submission CSV.
    """
    rows = []
    for _, row in submission_template.iterrows():
        user_id = int(row["user_id"])
        items = predictions.get(user_id)
        assert items is not None, f"missing prediction for user_id={user_id}"
        rows.append({
            "ID": int(row["ID"]),
            "user_id": user_id,
            "item_id": ",".join(str(item) for item in items),
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main(load_dir: Path,weights: tuple[float, float, float],output_path: Path = DEFAULT_OUTPUT,k: int = 10) -> None:
    """Load checkpoint, run ensemble inference, write submission CSV.

    Args:
        load_dir (Path): Directory containing saved checkpoint files.
        weights (tuple[float, float, float]): BPR, GRU, popularity ensemble weights.
        output_path (Path): Path to write submission CSV.
        k (int): Number of items to recommend per user.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"loading checkpoint from {load_dir}...")
    bpr_model, gru_model, gru_sequences, lightgcn_model, norm_adjacency, popularity, n_users, n_items = load_checkpoint(load_dir, device)
    print(f"  n_users={n_users}, n_items={n_items}, device={device}")

    full_train = pd.concat([pd.read_csv(PROCESSED_DIR / "train.csv"), pd.read_csv(PROCESSED_DIR / "val.csv")], ignore_index=True)
    submission_template = pd.read_csv(DATA_DIR / "sample_submission.csv")

    bpr_rec = BPRRecommender(bpr_model, device=device)
    gru_rec = GRU4RecRecommender(gru_model, gru_sequences, device=device)
    lightgcn_rec = LightGCNRecommender(lightgcn_model, norm_adjacency, device=device)
    pop_rec = PopularityRecommender(popularity, n_items, device=device)
    components = [("bpr", bpr_rec), ("gru", gru_rec), ("lightgcn", lightgcn_rec), ("pop", pop_rec)]

    submission_user_ids = submission_template["user_id"].astype(int).tolist()
    seen_per_user = full_train.groupby("user_id")["item_id"].apply(set).to_dict()

    print(f"predicting top-{k} for {len(submission_user_ids)} users "
          f"with weights BPR={weights[0]}, GRU={weights[1]}, POP={weights[2]}...")
    t0 = time.time()
    predictions = batch_predict_topk(
        components, weights, submission_user_ids, seen_per_user,
        k=k, batch_size=1024, device=device,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission(submission_template, predictions, output_path)
    print(f"wrote {len(submission_template)} rows to {output_path}")


def parse_weights(weights_str: str) -> tuple[float, float, float, float]:
    """Parse a comma-separated string of 4 floats into a weight tuple.

    Args:
        weights_str (str): Comma-separated weights e.g. '1.0,1.0,2.0,0.0'.

    Returns:
        (tuple[float, float, float, float]): Parsed weights for BPR, GRU, LightGCN, popularity.
    """
    parts = weights_str.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"expected 4 comma-separated weights, got: {weights_str!r}")
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--load", type=Path, required=True)
    parser.add_argument("--weights", type=parse_weights, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    main(
        load_dir=args.load,
        weights=args.weights,
        output_path=args.output,
        k=args.k,
    )
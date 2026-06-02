import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_raw_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw CSV files from disk

    Args:
        data_dir (Path): Directory containing train.csv, test.csv, item_meta.csv

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] as raw DataFrames
    """
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    item_meta_df = pd.read_csv(data_dir / "item_meta.csv")
    return train_df, test_df, item_meta_df


def deduplicate_interactions(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate (user_id, item_id) pairs, keeping the most recent interaction.

    Args:
        interactions_df (pd.DataFrame): Raw interactions with columns [user_id, item_id, timestamp]

    Returns:
        pd.DataFrame: Deduplicated interactions
    """
    before = len(interactions_df)
    interactions_df = (
        interactions_df
        .sort_values("timestamp")
        .drop_duplicates(subset=["user_id", "item_id"], keep="last")
        .reset_index(drop=True)
    )
    print(f"{before} to {len(interactions_df)} rows")
    return interactions_df


def split_leave_one_out(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split intercations into train and validation sets using leave-one-out strategy.
       The last interaction (by timestamp) per user goes to validation.

    Args:
        train_df (pd.DataFrame): Full interactions with columns [user_id, item_id, timestamp]

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: All interactions except the last per user / one most recent interaction per user
    """
    train_df = train_df.sort_values(["user_id", "timestamp"])
    last_interaction_idx = train_df.groupby("user_id")["timestamp"].idxmax()
    
    val_df = train_df.loc[last_interaction_idx].reset_index(drop=True)
    train_split_df = train_df.drop(index=last_interaction_idx).reset_index(drop=True)
    print(f"{len(train_split_df)}, {len(val_df)}")
    return train_split_df, val_df


def save_processed(train_split_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    """Save processed DataFrames and ID maps to disk.

    Args:
        train_split_df (pd.DataFrame): Training interactions after split
        val_df (pd.DataFrame): Validation interactions
        test_df (pd.DataFrame): Test users with remapped IDs
        user_id_map (dict): original to int user ID mapping
        item_id_map (dict): original to int item ID mappiong
        output_dir (Path): Directory to write outputs
    """
    
    train_split_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    print(f"saved to {output_dir}")
    
    
if __name__ == "__main__":
    train_df, test_df, item_meta_df = load_raw_data(DATA_DIR)
    train_df = deduplicate_interactions(train_df)
    train_split_df, val_df = split_leave_one_out(train_df)
    save_processed(train_split_df, val_df, test_df, OUTPUT_DIR)


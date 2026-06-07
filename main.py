"""Pipeline entry point.

Usage:
    python main.py preprocess   # raw csv -> data/processed/{train,val,test}.csv
    python main.py eval         # train BPR + GRU on train.csv, sweep ensemble weights on val.csv
    python main.py infer        # train on train+val, write data/submission.csv
    python main.py all          # preprocess -> eval -> infer
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def cmd_preprocess() -> None:
    from preprocess import (
        DATA_DIR, OUTPUT_DIR,
        load_raw_data, deduplicate_interactions,
        split_leave_one_out, save_processed,
    )
    train_df, test_df, _ = load_raw_data(DATA_DIR)
    train_df = deduplicate_interactions(train_df)
    train_split, val_df = split_leave_one_out(train_df)
    save_processed(train_split, val_df, test_df, OUTPUT_DIR)


def cmd_eval() -> None:
    from ensemble import run_sweep
    run_sweep()


def cmd_infer() -> None:
    from inference import main as run_inference
    run_inference()


COMMANDS = {
    "preprocess": cmd_preprocess,
    "eval": cmd_eval,
    "infer": cmd_infer,
    "all": lambda: (cmd_preprocess(), cmd_eval(), cmd_infer()),
}


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]]()

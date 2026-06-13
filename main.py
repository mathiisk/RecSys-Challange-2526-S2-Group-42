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


def cmd_train(args: list[str]) -> None:
    from train import main as run_train
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bpr-epochs", type=int, default=100)
    parser.add_argument("--gru-epochs", type=int, default=100)
    parser.add_argument("--lightgcn-epochs", type=int, default=20)
    parser.add_argument("--bpr-dim", type=int, default=256)
    parser.add_argument("--gru-embed-dim", type=int, default=64)
    parser.add_argument("--gru-hidden-dim", type=int, default=64)
    parser.add_argument("--gru-max-seq-len", type=int, default=100)
    parser.add_argument("--lightgcn-dim", type=int, default=128)
    parser.add_argument("--save", type=Path, default=None)
    parsed = parser.parse_args(args)

    run_train(
        bpr_epochs=parsed.bpr_epochs,
        gru_epochs=parsed.gru_epochs,
        lightgcn_epochs=parsed.lightgcn_epochs,
        bpr_dim=parsed.bpr_dim,
        gru_embed_dim=parsed.gru_embed_dim,
        gru_hidden_dim=parsed.gru_hidden_dim,
        gru_max_seq_len=parsed.gru_max_seq_len,
        lightgcn_dim=parsed.lightgcn_dim,
        save_dir=parsed.save,
    )


def cmd_sweep(args: list[str]) -> None:
    from sweep import main as run_sweep
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bpr-only", action="store_true")
    parser.add_argument("--gru-only", action="store_true")
    parser.add_argument("--lightgcn-only", action="store_true")
    parser.add_argument("--weights-only", action="store_true")
    parser.add_argument("--load", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("sweep_results.csv"))
    parser.add_argument("--step", type=float, default=1.0)
    parsed = parser.parse_args(args)

    only_flags = [parsed.bpr_only, parsed.gru_only, parsed.lightgcn_only, parsed.weights_only]
    run_sweep(
        sweep_bpr_flag=not any(only_flags) or parsed.bpr_only,
        sweep_gru_flag=not any(only_flags) or parsed.gru_only,
        sweep_lightgcn_flag=not any(only_flags) or parsed.lightgcn_only,
        sweep_weights_flag=parsed.weights_only,
        load_dir=parsed.load,
        output_path=parsed.output,
        step=parsed.step,
    )


def cmd_infer(args: list[str]) -> None:
    from inference import main as run_inference, parse_weights, DEFAULT_OUTPUT
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=Path, required=True)
    parser.add_argument("--weights", type=parse_weights, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k", type=int, default=10)
    parsed = parser.parse_args(args)

    run_inference(
        load_dir=parsed.load,
        weights=parsed.weights,
        output_path=parsed.output,
        k=parsed.k,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command, rest = sys.argv[1], sys.argv[2:]

    if command == "preprocess":
        cmd_preprocess()
    elif command == "train":
        cmd_train(rest)
    elif command == "sweep":
        cmd_sweep(rest)
    elif command == "infer":
        cmd_infer(rest)
    elif command == "all":
        cmd_preprocess()
        cmd_train(["--save", "checkpoints/final"])
        cmd_infer(["--load", "checkpoints/final", "--weights", "3,9,2,2"])
    else:
        print(__doc__)
        sys.exit(1)
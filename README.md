# Recommender System Pipeline

Hybrid recommender combining BPR-MF, GRU4Rec, LightGCN, and a popularity
baseline via a weighted score ensemble.

## Setup

```bash
pip install torch pandas numpy
```
or 
```bash
pip install -r requirements.txt
```

Place raw data in `data/`: `train.csv`, `test.csv`, `item_meta.csv`,
`sample_submission.csv`.

## Pipeline

```bash
python main.py preprocess
```
Deduplicates interactions, splits off the last interaction per user as
validation (leave-one-out), writes `data/processed/{train,val,test}.csv`.

```bash
python main.py train --save checkpoints/v1
```
Trains BPR-MF, GRU4Rec, LightGCN, and computes item popularity on
`train.csv`, evaluates on `val.csv`, runs a small default ensemble sweep,
and saves model checkpoints to the given directory. Key flags:
`--bpr-epochs`, `--gru-epochs`, `--lightgcn-epochs`, `--bpr-dim`,
`--gru-embed-dim`, `--gru-hidden-dim`, `--gru-max-seq-len`, `--lightgcn-dim`.

```bash
python main.py sweep --load checkpoints/v1 --weights-only --step 1.0
```
Sweeps ensemble weights over a loaded checkpoint on `val.csv`. Without
`--weights-only`, sweeps model hyperparameters instead (`--bpr-only`,
`--gru-only`, `--lightgcn-only` restrict to one model); results are written
to `--output` (default `sweep_results.csv`).

```bash
python main.py infer --load checkpoints/v1 --weights 3,9,2,2 --output data/submission.csv
```
Loads a checkpoint, runs the ensemble (weights in BPR, GRU, LightGCN,
popularity order) over all users in `sample_submission.csv`, and writes the
Kaggle submission CSV.

```bash
python main.py all
```
Runs preprocess -> train (saving to `checkpoints/final`) -> infer with
weights `3,9,2,2`.

## Code structure

- `src/preprocess.py` - load raw data, dedupe, leave-one-out split
- `src/popularity.py` - popularity baseline + recommender wrapper
- `src/bpr.py` - BPR-MF model, training, recommender wrapper
- `src/gru4rec.py` - GRU4Rec model, training, recommender wrapper
- `src/lightgcn.py` - LightGCN model, sparse adjacency, training, recommender wrapper
- `src/ensemble.py` - score normalization, seen-item masking, weight sweep
- `src/evaluate.py` - Recall@K evaluation
- `src/train.py` - trains all models, runs default ensemble sweep, saves checkpoint
- `src/sweep.py` - hyperparameter and ensemble weight sweeps
- `src/inference.py` - loads checkpoint, generates Kaggle submission
- `main.py` - CLI entry point for the full pipeline
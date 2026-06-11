# Sequential Product Recommender

Next-item recommendation from user interaction sequences — the core problem
behind "customers who viewed this went on to view…". The project is built
around a **baseline-vs-model comparison** under a clean leave-one-out
protocol, so every number is directly comparable.

The point isn't to ship one model — it's to show that a sequential neural
model earns its complexity *only* where simpler baselines fall short, and to
measure exactly where that is.

## Methods compared

| Method        | Idea                                                        | Captures |
|---------------|-------------------------------------------------------------|----------|
| Popularity    | Recommend globally frequent items, ignore order             | nothing sequential |
| Markov(1)     | `P(next \| last item)` from transition counts                | first-order order |
| ItemKNN       | Co-occurrence within sessions, scored over recent context   | item affinity |
| GRU4Rec       | Embedding → GRU → softmax over items, next-item CE loss      | longer-range memory |

## Results (synthetic data, default run)

The bundled synthetic generator injects three signals — popularity noise, a
lag-1 dependency (next item depends on the last item), and a lag-2 dependency
(next item depends on the item *two steps back*). A first-order model
structurally can't see the lag-2 term; a model with memory can.

Leave-one-out test set, `python train.py`:

| Method     | Recall@10 | NDCG@10 | MRR   |
|------------|-----------|---------|-------|
| Popularity | 0.066     | 0.038   | 0.038 |
| Markov(1)  | 0.675     | 0.521   | 0.484 |
| ItemKNN    | 0.603     | 0.292   | 0.213 |
| GRU4Rec    | **0.783** | **0.629** | **0.580** |

Reading: Markov(1) is a genuinely strong baseline (it nails the lag-1 term),
and GRU4Rec's margin over it is precisely the lag-2 structure only a
memory-based model can use. That attribution — *why* the model wins — is the
result worth presenting, not the raw score.

## Quick start

```bash
pip install -r requirements.txt

python run_baselines.py        # baselines only (fast, no PyTorch needed to read)
python train.py                # train GRU4Rec + full comparison table
python train.py --epochs 30    # more epochs
```

### Using real data

The synthetic generator is the default so the repo runs out of the box. Swap
in real interactions via `--data`:

```bash
# Amazon Reviews 2023 (download a category .jsonl from the link in data/loader.py)
python train.py --data amazon --path Electronics.jsonl

# RetailRocket events.csv (Kaggle)
python train.py --data retailrocket --path events.csv

# any interactions CSV with user/item/timestamp columns
python train.py --data csv --path interactions.csv
```

`data/loader.py` documents where to download each dataset and applies k-core
filtering (drops rare items / short users) before splitting.

## Evaluation protocol

Standard **leave-one-out** for next-item recommendation. For each user
sequence `[i1 … i_{n-1}, i_n]`:

- **test**: input `[i1 … i_{n-1}]`, target `i_n`
- **val**: input `[i1 … i_{n-2}]`, target `i_{n-1}`
- **train**: next-item pairs from `[i1 … i_{n-2}]` only

Val/test targets are never training labels, so there's no leakage. Metrics are
ranked over the full item catalog (Recall@K, NDCG@K, MRR). On a large real
catalog you'd switch to sampled negatives for speed — noted in `src/metrics.py`.

## Repo structure

```
seqrec/
├── config.py                 # all hyperparameters in one dataclass
├── run_baselines.py          # fit + evaluate baselines
├── train.py                  # train GRU4Rec + full comparison
├── data/
│   ├── synthetic.py          # runs-out-of-the-box data generator
│   └── loader.py             # Amazon / RetailRocket / CSV loaders + k-core
└── src/
    ├── dataset.py            # sequence building + leave-one-out split
    ├── metrics.py            # Recall@K, NDCG@K, MRR + eval harness
    ├── baselines/            # popularity, markov, item_knn
    └── models/gru4rec.py     # PyTorch GRU4Rec
```

## An honest note for the writeup

On many *real* session datasets, well-tuned simple baselines (session kNN,
Markov) are surprisingly competitive with neural models — a documented finding
in the recommender-systems literature. If that happens on your data, that's a
legitimate, sophisticated result, not a failure: report it plainly. A rigorous
comparison that says "the simple baseline was within 2% and 50× cheaper" is a
stronger portfolio signal than an unverified claim that the neural model won.

## Possible extensions

- Swap GRU4Rec for a self-attention model (SASRec) and add it to the table.
- Add sampled-negative evaluation for large catalogs.
- Wrap the trained model in a small FastAPI `/recommend` endpoint + Dockerfile.

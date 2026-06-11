# Sequential Product Recommender

Next-item recommendation from user interaction sequences — the problem behind
*"customers who viewed this went on to view…"*.

- **What it does:** predicts the next item a user will interact with from their session history.
- **How it's built:** a fair baseline-vs-model comparison under a clean leave-one-out protocol, so every number is directly comparable.
- **The point:** show the neural model earns its complexity *only* where simpler baselines fall short — a strong baseline losing by a known margin beats an unexamined "the neural net won."

**Stack:** Python · PyTorch · NumPy. Baselines need no deep-learning dependencies.

## Methods compared

| Method      | Idea                                                      | Captures            |
|-------------|-----------------------------------------------------------|---------------------|
| Popularity  | Recommend globally frequent items, ignore order           | nothing sequential  |
| Markov(1)   | `P(next \| last item)` from transition counts             | first-order order   |
| ItemKNN     | Co-occurrence within sessions, scored over recent context | item affinity       |
| GRU4Rec     | Embedding → GRU → softmax over items, next-item CE loss    | longer-range memory |

All four use the same split, same metrics, and full-catalog ranking — that's what makes the comparison fair.

## Results on RetailRocket (real data)

> **Fill in with measured values** from
> `python train.py --data retailrocket --path events.csv`, then delete this line.

| Method     | Recall@10 | NDCG@10 | MRR  |
|------------|-----------|---------|------|
| Popularity | –         | –       | –    |
| Markov(1)  | –         | –       | –    |
| ItemKNN    | –         | –       | –    |
| GRU4Rec    | –         | –       | –    |

Add one line of interpretation once you have the numbers: where GRU4Rec's gain comes from, or — if the baselines stay close — that finding, stated plainly.

## Results on synthetic data (controlled benchmark)

The generator injects three known signals to isolate what each model can capture:

- **Popularity noise** — globally frequent items, no order.
- **Lag-1 dependency** — next item depends on the last item (a first-order model can see this).
- **Lag-2 dependency** — next item depends on the item *two steps back* (only a model with memory can see this).

Leave-one-out test set, `python train.py`:

| Method     | Recall@10 | NDCG@10   | MRR       |
|------------|-----------|-----------|-----------|
| Popularity | 0.066     | 0.038     | 0.038     |
| Markov(1)  | 0.675     | 0.521     | 0.484     |
| ItemKNN    | 0.603     | 0.292     | 0.213     |
| GRU4Rec    | **0.783** | **0.629** | **0.580** |

Markov(1) is a strong baseline (it nails the lag-1 term); GRU4Rec's margin over it is exactly the lag-2 structure only memory can use. *Why* the model wins is the point, not the raw score.

## Evaluation protocol

Standard **leave-one-out** for next-item recommendation. For each user sequence `[i1 … i_{n-1}, i_n]`:

- **test** — input `[i1 … i_{n-1}]`, target `i_n`
- **val** — input `[i1 … i_{n-2}]`, target `i_{n-1}`
- **train** — next-item pairs from `[i1 … i_{n-2}]` only

Val/test targets are never training labels, so there's no leakage. Metrics (Recall@K, NDCG@K, MRR) are ranked over the full catalog; sampled negatives are the standard speed-up for large catalogs (noted in `src/metrics.py`).

## Quick start

```bash
pip install -r requirements.txt

python run_baselines.py        # baselines only — fast, no PyTorch needed
python train.py                # GRU4Rec + full comparison (synthetic by default)
python train.py --epochs 30    # train longer
```

Swap in real interactions with `--data`:

```bash
python train.py --data retailrocket --path events.csv      # RetailRocket (Kaggle)
python train.py --data amazon --path Electronics.jsonl     # Amazon Reviews 2023
python train.py --data csv --path interactions.csv         # any user/item/timestamp CSV
```

`data/loader.py` documents where to get each dataset and applies k-core filtering before splitting.

## Repo structure

```
seqrec/
├── config.py            # all hyperparameters in one dataclass
├── run_baselines.py     # fit + evaluate baselines
├── train.py             # train GRU4Rec + full comparison
├── data/
│   ├── synthetic.py     # out-of-the-box data generator
│   └── loader.py        # Amazon / RetailRocket / CSV loaders + k-core
└── src/
    ├── dataset.py       # sequence building + leave-one-out split
    ├── metrics.py       # Recall@K, NDCG@K, MRR + eval harness
    ├── baselines/       # popularity, markov, item_knn
    └── models/gru4rec.py
```

## A note on interpreting results

- On many real session datasets, well-tuned simple baselines (session kNN, Markov) are surprisingly competitive with neural models — a documented finding in the literature.
- If that happens here, it's a legitimate, sophisticated result, not a failure.
- "The baseline came within 2% at 50× lower cost" is a stronger signal than an unexamined claim that the neural model won.

## Possible extensions

- Add a self-attention model (SASRec) to the comparison table.
- Sampled-negative evaluation for large catalogs.
- Wrap the trained model in a FastAPI `/recommend` endpoint + Dockerfile.

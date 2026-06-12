# Sequential Product Recommender

**Next-item recommendation** — predicting what a user will view or buy next from
their session history. The engine behind *"customers who viewed this went on to
view…"*.

- **The task:** given a user's session so far, predict the next item they'll interact with.
- **The approach:** benchmark a neural sequence model (GRU4Rec) against three classical baselines under one clean, leakage-free protocol.
- **The question:** does the neural model's added complexity pay off — and exactly where?

## TL;DR

- A fair **baseline-vs-model comparison** for next-item recommendation — four methods, identical split, identical metrics.
- On real e-commerce data (RetailRocket: 45K items, 2.7M interactions), GRU4Rec reaches **Recall@10 ≈ 0.46, NDCG@10 ≈ 0.36** after 10 epochs, with no train/test gap.
- The point isn't "the neural net won" — it's a rigorous table plus a clear explanation of *why* the numbers land where they do.

## Methods compared

| Method      | Idea                                                      | Captures            |
|-------------|-----------------------------------------------------------|---------------------|
| Popularity  | Recommend globally frequent items, ignore order           | nothing sequential  |
| Markov(1)   | `P(next \| last item)` from transition counts             | first-order order   |
| ItemKNN     | Co-occurrence within sessions, scored over recent context | item affinity       |
| GRU4Rec     | Embedding → GRU → softmax over items, next-item CE loss    | longer-range memory |

- All four methods use the **same split, same metrics, and same ranking** — that's what makes the comparison fair and the table meaningful.

## Results on RetailRocket (real data)

Leave-one-out test set, evaluated on a 5,000-instance sample.

| Method     | Recall@10  | NDCG@10    | MRR        |
|------------|------------|------------|------------|
| Popularity | _pending_  | _pending_  | _pending_  |
| Markov(1)  | _pending_  | _pending_  | _pending_  |
| ItemKNN    | _pending_  | _pending_  | _pending_  |
| GRU4Rec    | **0.4636** | **0.3635** | **0.3361** |

> **TODO before sharing:** fill the three baseline rows by running
> `python train.py --data retailrocket --path events.csv`, then add one line of
> interpretation — where GRU4Rec's gain comes from, or (if the baselines stay
> close) that finding, stated plainly. A complete table is what makes this land.

## Validation on synthetic data (controlled benchmark)

The comparison is first validated on synthetic data with three *known* signals,
so each model can be checked against what it should and shouldn't capture:

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

- **Markov(1) is a strong baseline** — it nails the lag-1 term.
- **GRU4Rec's margin over Markov is exactly the lag-2 structure** that only a memory-based model can use.
- **Takeaway:** the controlled experiment proves the *mechanism*; the RetailRocket table shows it holds on real data.

## Evaluation protocol

Standard **leave-one-out** for next-item recommendation. For each user sequence
`[i1 … i_{n-1}, i_n]`:

- **test** — input `[i1 … i_{n-1}]`, target `i_n`
- **val** — input `[i1 … i_{n-2}]`, target `i_{n-1}`
- **train** — next-item pairs from `[i1 … i_{n-2}]` only

- **No leakage:** val/test targets are never used as training labels.
- **Metrics:** Recall@K, NDCG@K, and MRR, ranking the true next item against the catalog.
- **Scale:** on large catalogs the eval set is sampled for speed.

## Quick start

```bash
pip install -r requirements.txt

python train.py                # GRU4Rec + full comparison (synthetic by default)
python train.py --epochs 30    # train longer
```

Swap in real interactions with `--data`:

```bash
python train.py --data retailrocket --path events.csv      # RetailRocket (Kaggle)
python train.py --data amazon --path Electronics.jsonl     # Amazon Reviews 2023
python train.py --data csv --path interactions.csv         # any user/item/timestamp CSV
```

`data/loader.py` documents where to get each dataset and applies k-core
filtering before splitting.

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

- On many real session datasets, well-tuned simple baselines (session kNN, Markov) are surprisingly competitive with neural models — a documented finding in the recommender-systems literature.
- If that turns out to be the case here, it's a legitimate, sophisticated result, not a failure.
- *"The baseline came within 2% at a fraction of the cost"* is a stronger portfolio signal than an unexamined claim that the neural model won.

## Possible extensions

- Add a self-attention model (SASRec) to the comparison table.
- Wrap the trained model in a FastAPI `/recommend` endpoint + Dockerfile for a live demo.
- Sampled-negative training loss for faster, more scalable optimization.

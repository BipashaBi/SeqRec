# Sequential Product Recommender

**Predicting what a user will look at next** — the idea behind
*"customers who viewed this also viewed…"*.

This project asks one simple question: to guess your next click, do you really
need a fancy neural network, or is a simple method just as good? It compares
four methods — three simple, one neural — fairly, on the same data, and reports
who wins and *why*.

## TL;DR

- Compares a neural network (**GRU4Rec**) against three simpler methods at predicting a user's next item.
- **The finding:** on real shopping data, the *simplest* method — basically a "what usually comes next" counter — **beats the neural network**. Your next click is mostly decided by the one item right before it, so the neural net's extra power isn't needed.
- On made-up data with deliberately hidden deeper patterns, the neural net wins instead. **Lesson: a complex model is only worth it when the data actually has complex patterns.**

## The four methods

| Method | What it does (in plain words) | How far back it looks |
|--------|-------------------------------|-----------------------|
| Popularity | Always suggests the most popular items, ignoring you | nothing |
| Markov(1)  | Suggests what usually comes right after your **last** item | 1 step |
| ItemKNN    | Suggests items that often show up alongside your recent ones | a few recent items |
| GRU4Rec    | A neural network that learns patterns across your **whole** session | whole session |

All four are tested the exact same way, so the comparison is fair.

## Results on real data (RetailRocket — a real online store's click logs)

**How to read the scores** (higher is better, the most is 1.0):

- **Recall@10** — how often the correct next item is somewhere in the top 10 suggestions.
- **NDCG@10** — same idea, but gives more credit for putting the right item *near the top*.
- **MRR** — on average, how near the top of the list the right item lands.

| Method     | Recall@10  | NDCG@10    | MRR        |
|------------|------------|------------|------------|
| Popularity | 0.0082     | 0.0038     | 0.0041     |
| **Markov(1)** | **0.7568** | **0.5042** | **0.4372** |
| ItemKNN    | 0.3276     | 0.1765     | 0.1461     |
| GRU4Rec    | 0.4992     | 0.3789     | 0.3461     |

**The simplest method wins:**

- **Markov(1) beats the neural network** by a wide margin. On this data, your next click is mostly decided by the single item you just looked at — so simply counting "what usually comes next" works best.
- **Popularity scores almost zero.** Suggesting generic popular items almost never matches what someone actually clicks next — the signal is personal, not generic.
- **Honest takeaway:** the neural network's ability to remember long histories doesn't help when the answer is just one step away. This is a well-known result in the field, not a mistake — and showing it is the point.

## Why test on made-up data too?

To prove the setup is correct, the same methods are run on data with three
patterns deliberately planted, so each method can be checked against what it
*should* be able to find:

- **Popular items** — some items simply appear a lot.
- **1-step pattern** — the next item depends on the last item.
- **2-step pattern** — the next item depends on the item *two back* (a simple method can't see this; a neural net can).

| Method     | Recall@10 | NDCG@10   | MRR       |
|------------|-----------|-----------|-----------|
| Popularity | 0.066     | 0.038     | 0.038     |
| Markov(1)  | 0.675     | 0.521     | 0.484     |
| ItemKNN    | 0.603     | 0.292     | 0.213     |
| GRU4Rec    | **0.783** | **0.629** | **0.580** |

- Here the **neural network wins** — because there's a 2-step pattern only it can catch.
- **This contrast is the whole point:** the neural net wins when there are deeper patterns to find, and loses to a simple method when there aren't. Match the model to the data — complexity should earn its place.

## How it's tested (fairly)

- For each user, we **hide their last action** and check whether the model predicts it, using only their earlier history.
- Nothing from the test is used during training, so there's **no cheating** and the scores are honest.
- Each score is measured by ranking the correct item against the full catalog.

## Quick start

```bash
pip install -r requirements.txt

python train.py                # neural net + all baselines (built-in demo data)
python train.py --epochs 30    # train longer
```

Use real data with `--data`:

```bash
python train.py --data retailrocket --path events.csv      # RetailRocket (Kaggle)
python train.py --data amazon --path Electronics.jsonl     # Amazon Reviews 2023
python train.py --data csv --path interactions.csv         # any user/item/time CSV
```

`data/loader.py` lists where to download each dataset, and drops very rare items
and very short sessions before testing.

## Repo structure

```
seqrec/
├── config.py            # all settings in one place
├── run_baselines.py     # run the simple methods only
├── train.py             # train the neural net + run the full comparison
├── data/
│   ├── synthetic.py     # the built-in demo data generator
│   └── loader.py        # loaders for RetailRocket / Amazon / any CSV
└── src/
    ├── dataset.py       # builds sequences + the train/test split
    ├── metrics.py       # the scoring (Recall, NDCG, MRR)
    ├── baselines/       # popularity, markov, item_knn
    └── models/gru4rec.py
```

## In short

- A fair contest between a neural recommender and simple baselines.
- On real data, the simple method wins; on data with deeper patterns, the neural net wins.
- The real takeaway: **match the model to the data — complexity has to earn its place.**

## Possible next steps

- Add another neural model (SASRec) to the comparison.
- Turn the trained model into a live API: send it a session, get recommendations back.
- Speed up training for very large catalogs.

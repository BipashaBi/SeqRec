"""
Train GRU4Rec and compare it against all baselines on the same split.

    python train.py                          # synthetic data (default)
    python train.py --data retailrocket --path events.csv --epochs 20

Reports Recall@K / NDCG@K / MRR on the held-out test set for every method,
so the headline result is the baseline-vs-model table.

Note: evaluation ranks the full item catalog per instance, which is O(N x V)
and prohibitive on large test sets. --eval_sample evaluates on a random sample
of instances (default 5000) for a fast, reliable estimate. Use --eval_sample 0
for the full test set.
"""
import argparse
import random
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset as TorchDataset

from config import CONFIG
from data.synthetic import generate_synthetic
from data import loader
from src.dataset import build_dataset
from src.metrics import evaluate, format_results
from src.baselines.popularity import PopularityRecommender
from src.baselines.markov import MarkovRecommender
from src.baselines.item_knn import ItemKNNRecommender
from src.models.gru4rec import GRU4Rec


# --------------------------- training data ---------------------------------
class NextItemDataset(TorchDataset):
    """Each training sequence -> (input = s[:-1], target = s[1:])."""
    def __init__(self, sequences):
        self.seqs = [s for s in sequences if len(s) >= 2]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        s = self.seqs[i]
        return torch.tensor(s[:-1]), torch.tensor(s[1:])


def collate(batch):
    inputs, targets = zip(*batch)
    x = pad_sequence(inputs, batch_first=True, padding_value=0)
    y = pad_sequence(targets, batch_first=True, padding_value=0)
    return x, y


# ------------------------------- helpers -----------------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataframe(args):
    if args.data == "synthetic":
        return generate_synthetic(seed=CONFIG.seed)
    if args.data == "amazon":
        df = loader.load_amazon_reviews(args.path)
    elif args.data == "retailrocket":
        df = loader.load_retailrocket(args.path)
    else:
        df = loader.from_csv(args.path)
    return loader.filter_core(df)


def run_baselines(ds):
    results = {}
    for m in (PopularityRecommender(), MarkovRecommender(), ItemKNNRecommender()):
        m.fit(ds.train_sequences, ds.n_items)
        results[m.name] = evaluate(m.score, ds.test_data, ds.n_items,
                                   CONFIG.k_list, CONFIG.exclude_seen)
    return results


def train_gru(ds, cfg, epochs):
    device = cfg.device
    model = GRU4Rec(ds.n_items, cfg.emb_dim, cfg.hidden_size,
                    cfg.num_layers, cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    loader_ = DataLoader(NextItemDataset(ds.train_sequences),
                         batch_size=cfg.batch_size, shuffle=True,
                         collate_fn=collate)

    best_ndcg, best_state, bad = -1.0, None, 0
    score_fn = lambda seq: model.score_sequence(seq, device)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for x, y in loader_:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)                       # (B, T, V)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        val = evaluate(score_fn, ds.val_data, ds.n_items,
                       cfg.k_list, cfg.exclude_seen)
        ndcg = val["NDCG@10"]
        print(f"epoch {epoch:2d} | loss {total / len(loader_.dataset):.4f} "
              f"| val NDCG@10 {ndcg:.4f} | {time.time() - t0:.1f}s")
        if ndcg > best_ndcg:
            best_ndcg, best_state, bad = ndcg, \
                {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                print("early stopping")
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def subsample_eval(ds, n, seed):
    """Randomly subsample val/test instances so full-catalog ranking is fast."""
    if not n:
        return
    if len(ds.val_data) > n:
        ds.val_data = random.Random(seed).sample(list(ds.val_data), n)
    if len(ds.test_data) > n:
        ds.test_data = random.Random(seed + 1).sample(list(ds.test_data), n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="synthetic",
                   choices=["synthetic", "amazon", "retailrocket", "csv"])
    p.add_argument("--path", default=None)
    p.add_argument("--epochs", type=int, default=CONFIG.epochs)
    p.add_argument("--eval_sample", type=int, default=5000,
                   help="evaluate on this many sampled instances (0 = full set)")
    args = p.parse_args()

    set_seed(CONFIG.seed)
    if torch.cuda.is_available():
        CONFIG.device = "cuda"

    df = load_dataframe(args)
    ds = build_dataset(df, CONFIG.min_seq_len, CONFIG.max_seq_len)
    subsample_eval(ds, args.eval_sample, CONFIG.seed)
    print(f"items={ds.n_items} | train_seqs={len(ds.train_sequences)} "
          f"| val={len(ds.val_data)} | test={len(ds.test_data)} "
          f"| device={CONFIG.device} | eval_sample={args.eval_sample}\n")

    print("Running baselines...")
    t0 = time.time()
    baseline_results = run_baselines(ds)
    print(f"baselines done in {time.time() - t0:.1f}s\n")

    print("Training GRU4Rec...")
    model = train_gru(ds, CONFIG, args.epochs)
    gru_results = evaluate(lambda s: model.score_sequence(s, CONFIG.device),
                           ds.test_data, ds.n_items,
                           CONFIG.k_list, CONFIG.exclude_seen)

    print("\nTest-set results (leave-one-out next-item):")
    for name, res in baseline_results.items():
        print(format_results(name, res))
    print(format_results("GRU4Rec", gru_results))


if __name__ == "__main__":
    main()
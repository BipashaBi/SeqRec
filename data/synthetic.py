"""
Synthetic interaction generator.

Produces a (user_id, item_id, timestamp) DataFrame that mimics real
session data so the whole pipeline runs without any download.

The dynamics are designed so each method has a clear, interpretable ceiling:

  * popularity noise  (prob 1 - p_lag1 - p_lag2)
        next item ~ global popularity (Zipf). Only the Popularity
        baseline benefits; everyone else treats it as noise.

  * lag-1 dependency  (prob p_lag1)
        next = f(last item). A first-order Markov model captures this
        exactly -> it's the strong, fair baseline.

  * lag-2 dependency  (prob p_lag2)
        next = g(item TWO steps back). A first-order model structurally
        cannot see this; a model with memory (GRU4Rec) can. This is the
        gap the sequential model is supposed to close.

So the expected ordering is Popularity < Markov(1) < GRU4Rec, and the
size of the GRU's win over Markov is governed by p_lag2.
"""
import numpy as np
import pandas as pd


def generate_synthetic(
    n_users: int = 5000,
    n_items: int = 1000,
    avg_seq_len: int = 12,
    p_lag1: float = 0.35,
    p_lag2: float = 0.45,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame with columns [user_id, item_id, timestamp]."""
    rng = np.random.default_rng(seed)

    ranks = np.arange(1, n_items + 1)
    base_pop = 1.0 / np.power(ranks, 0.8)
    base_pop /= base_pop.sum()
    all_items = np.arange(1, n_items + 1)

    # deterministic transition maps, indexed by item id (slot 0 unused)
    lag1_map = rng.integers(1, n_items + 1, size=n_items + 1)
    lag2_map = rng.integers(1, n_items + 1, size=n_items + 1)

    rows = []
    ts = 0
    for user in range(n_users):
        seq_len = max(3, int(rng.poisson(avg_seq_len)))
        cur = int(rng.choice(all_items, p=base_pop))
        prev = 0
        seq = [cur]
        for _ in range(seq_len - 1):
            u = rng.random()
            if prev != 0 and u < p_lag2:
                nxt = int(lag2_map[prev])          # depends on item 2 steps back
            elif u < p_lag2 + p_lag1:
                nxt = int(lag1_map[cur])           # depends on last item
            else:
                nxt = int(rng.choice(all_items, p=base_pop))  # popularity noise
            prev, cur = cur, nxt
            seq.append(cur)

        for it in seq:
            rows.append((user, it, ts))
            ts += 1  # global monotonically increasing timestamp

    return pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"])


if __name__ == "__main__":
    df = generate_synthetic()
    print(df.head())
    print(f"\n{len(df):,} interactions | "
          f"{df.user_id.nunique():,} users | "
          f"{df.item_id.nunique():,} items")

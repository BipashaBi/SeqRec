"""
First-order Markov baseline: score next items by P(next | last item),
estimated from transition counts in the training sequences. Falls back to
global popularity when the last item has no observed transitions.
"""
from collections import defaultdict
from typing import List

import numpy as np


class MarkovRecommender:
    name = "Markov(1)"

    def fit(self, train_sequences: List[List[int]], n_items: int):
        self.n_items = n_items
        self.trans = defaultdict(lambda: np.zeros(n_items + 1))
        self.pop = np.zeros(n_items + 1)
        for seq in train_sequences:
            for a, b in zip(seq, seq[1:]):
                self.trans[a][b] += 1.0
            for item in seq:
                self.pop[item] += 1.0
        return self

    def score(self, seq: List[int]) -> np.ndarray:
        if not seq:
            return self.pop
        last = seq[-1]
        row = self.trans.get(last)
        if row is not None and row.sum() > 0:
            return row
        return self.pop

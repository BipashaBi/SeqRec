"""
First-order Markov baseline: score next items by P(next | last item),
estimated from transition counts in the training sequences. Falls back to
global popularity when the last item has no observed transitions.

Transitions are stored sparsely (only observed last -> next pairs) instead of
a dense V x V matrix, so it scales to large catalogues without exhausting RAM.
"""
from collections import defaultdict, Counter
from typing import List

import numpy as np


class MarkovRecommender:
    name = "Markov(1)"

    def fit(self, train_sequences: List[List[int]], n_items: int):
        self.n_items = n_items
        # sparse transition counts: trans[a][b] = number of times a -> b
        self.trans = defaultdict(Counter)
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
        row_counts = self.trans.get(seq[-1])
        if not row_counts:
            return self.pop
        scores = np.zeros(self.n_items + 1)
        for b, c in row_counts.items():
            scores[b] = c
        return scores
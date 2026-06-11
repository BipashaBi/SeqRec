"""
Session co-occurrence item-kNN baseline. Items that co-occur within the
same session are considered similar; we score candidates by their summed
co-occurrence with the recent context (last few items). A strong, classic
session-based baseline that's often hard to beat.
"""
from collections import Counter, defaultdict
from typing import List

import numpy as np


class ItemKNNRecommender:
    name = "ItemKNN"

    def __init__(self, context_window: int = 3):
        self.context_window = context_window

    def fit(self, train_sequences: List[List[int]], n_items: int):
        self.n_items = n_items
        self.cooc = defaultdict(Counter)
        self.pop = np.zeros(n_items + 1)
        for seq in train_sequences:
            items = set(seq)
            for a in items:
                for b in items:
                    if a != b:
                        self.cooc[a][b] += 1
            for item in seq:
                self.pop[item] += 1.0
        return self

    def score(self, seq: List[int]) -> np.ndarray:
        scores = np.zeros(self.n_items + 1)
        context = seq[-self.context_window:] if seq else []
        for item in context:
            for j, c in self.cooc.get(item, {}).items():
                scores[j] += c
        if scores.sum() == 0:
            return self.pop
        return scores

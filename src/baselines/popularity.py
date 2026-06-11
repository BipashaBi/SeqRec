"""Most-popular baseline: recommend globally frequent items, ignoring order."""
from typing import List

import numpy as np


class PopularityRecommender:
    name = "Popularity"

    def fit(self, train_sequences: List[List[int]], n_items: int):
        self.n_items = n_items
        counts = np.zeros(n_items + 1)
        for seq in train_sequences:
            for item in seq:
                counts[item] += 1
        self.scores_ = counts
        return self

    def score(self, seq: List[int]) -> np.ndarray:
        return self.scores_

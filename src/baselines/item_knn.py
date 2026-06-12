"""
Session co-occurrence item-kNN baseline. Items that co-occur within the same
session are treated as similar; candidates are scored by their summed
co-occurrence with the recent context (last few items). A strong, classic
session-based baseline that's often hard to beat.

Co-occurrence is computed as a sparse item-item matrix (S^T S over a
session x item incidence matrix) and capped to the top-K neighbours per item,
so it scales to large catalogues without a dense V x V blow-up in time or memory.
"""
from typing import List

import numpy as np
from scipy.sparse import csr_matrix


class ItemKNNRecommender:
    name = "ItemKNN"

    def __init__(self, context_window: int = 3, topk: int = 500):
        self.context_window = context_window
        self.topk = topk

    def fit(self, train_sequences: List[List[int]], n_items: int):
        self.n_items = n_items
        V = n_items + 1

        # session x item binary incidence matrix
        rows, cols = [], []
        for s, seq in enumerate(train_sequences):
            for it in set(seq):
                rows.append(s)
                cols.append(it)
        S = csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(train_sequences), V),
        )

        # item-item co-occurrence counts = S^T S (sparse), drop self-pairs
        cooc = (S.T @ S).tocsr()
        cooc.setdiag(0.0)
        cooc.eliminate_zeros()

        self.cooc = self._topk_per_row(cooc, self.topk)
        self.pop = np.asarray(S.sum(axis=0)).ravel()
        return self

    @staticmethod
    def _topk_per_row(mat: csr_matrix, k: int) -> csr_matrix:
        """Keep only the k largest entries in each row, to bound memory."""
        mat = mat.tocsr()
        data, indices, indptr = mat.data, mat.indices, mat.indptr
        new_data, new_idx, new_indptr = [], [], [0]
        for r in range(mat.shape[0]):
            start, end = indptr[r], indptr[r + 1]
            row_data, row_idx = data[start:end], indices[start:end]
            if len(row_data) > k:
                top = np.argpartition(row_data, -k)[-k:]
                row_data, row_idx = row_data[top], row_idx[top]
            new_data.append(row_data)
            new_idx.append(row_idx)
            new_indptr.append(new_indptr[-1] + len(row_data))
        return csr_matrix(
            (np.concatenate(new_data) if new_data else np.zeros(0, np.float32),
             np.concatenate(new_idx) if new_idx else np.zeros(0, int),
             np.asarray(new_indptr)),
            shape=mat.shape,
        )

    def score(self, seq: List[int]) -> np.ndarray:
        scores = np.zeros(self.n_items + 1, dtype=np.float32)
        context = seq[-self.context_window:] if seq else []
        for item in context:
            if 0 <= item < self.cooc.shape[0]:
                row = self.cooc.getrow(item)
                scores[row.indices] += row.data
        if scores.sum() == 0:
            return self.pop
        return scores
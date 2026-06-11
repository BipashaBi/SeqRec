"""
Ranking metrics (Recall@K, NDCG@K, MRR) and a model-agnostic eval harness.

Every recommender exposes `score(input_seq) -> np.ndarray` of length
(n_items + 1), where index 0 is the padding slot. The harness ranks those
scores against the single held-out target and averages over all instances.
"""
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np


def metrics_from_scores(scores: np.ndarray, target: int,
                        k_list: List[int]) -> Dict[str, float]:
    """Metrics for one instance given a full score vector and the true item."""
    target_score = scores[target]
    # 1-based rank; ties resolved optimistically (strictly-greater count)
    rank = int(np.sum(scores > target_score)) + 1

    out = {}
    for k in k_list:
        hit = rank <= k
        out[f"Recall@{k}"] = 1.0 if hit else 0.0
        out[f"NDCG@{k}"] = (1.0 / np.log2(rank + 1)) if hit else 0.0
    out["MRR"] = 1.0 / rank
    return out


def evaluate(
    score_fn: Callable[[List[int]], np.ndarray],
    eval_data: List[Tuple[List[int], int]],
    n_items: int,
    k_list: List[int],
    exclude_seen: bool = False,
) -> Dict[str, float]:
    """Average metrics over an eval set. `score_fn` maps an input seq to scores."""
    agg = defaultdict(float)
    n = 0
    for seq, target in eval_data:
        scores = np.array(score_fn(seq), dtype=float).copy()
        scores[0] = -np.inf                       # never recommend padding
        if exclude_seen:
            seen = [i for i in set(seq) if i != target]
            if seen:
                scores[seen] = -np.inf
        m = metrics_from_scores(scores, target, k_list)
        for key, val in m.items():
            agg[key] += val
        n += 1
    return {key: val / n for key, val in agg.items()}


def format_results(name: str, results: Dict[str, float]) -> str:
    parts = " | ".join(f"{k}: {v:.4f}" for k, v in results.items())
    return f"{name:<14} {parts}"

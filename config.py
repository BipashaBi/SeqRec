"""Central configuration for the sequential recommender project."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # --- evaluation ---
    k_list: List[int] = field(default_factory=lambda: [5, 10, 20])
    exclude_seen: bool = False  # session-rec convention: allow repeat items as targets

    # --- data splitting ---
    min_seq_len: int = 3        # need >=3 to carve out train + val + test targets
    max_seq_len: int = 50       # truncate long histories (keep most recent)

    # --- GRU4Rec model ---
    emb_dim: int = 64
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.2

    # --- training ---
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 3           # early stopping on val NDCG@10
    seed: int = 42
    import torch
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # set to "cuda" if available


CONFIG = Config()

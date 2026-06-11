"""
Turn a raw [user_id, item_id, timestamp] DataFrame into:
  * contiguous item ids (0 = padding)
  * per-user chronological sequences
  * a leave-one-out split (last item = test, 2nd-last = val)

Leave-one-out protocol (standard for next-item recommendation):
  sequence = [i1, i2, ..., i_{n-1}, i_n]
    test  : input = [i1 .. i_{n-1}],  target = i_n
    val   : input = [i1 .. i_{n-2}],  target = i_{n-1}
    train : next-item pairs taken from [i1 .. i_{n-2}] only
            (val/test targets are never used as training labels -> no leakage)
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class Dataset:
    n_items: int                                   # number of real items (ids 1..n_items)
    train_sequences: List[List[int]]               # for fitting baselines / training
    val_data: List[Tuple[List[int], int]]          # (input_seq, target)
    test_data: List[Tuple[List[int], int]]         # (input_seq, target)
    item2idx: Dict[object, int]                    # original id -> contiguous id


def build_dataset(df: pd.DataFrame, min_seq_len: int = 3,
                  max_seq_len: int = 50) -> Dataset:
    # 1. contiguous item ids (reserve 0 for padding)
    unique_items = df["item_id"].unique()
    item2idx = {raw: i + 1 for i, raw in enumerate(unique_items)}
    df = df.assign(item_idx=df["item_id"].map(item2idx))

    # 2. chronological sequence per user
    df = df.sort_values(["user_id", "timestamp"])
    sequences = df.groupby("user_id")["item_idx"].apply(list)

    train_sequences, val_data, test_data = [], [], []
    for seq in sequences:
        # keep only the most recent max_seq_len interactions
        if len(seq) > max_seq_len:
            seq = seq[-max_seq_len:]
        if len(seq) < min_seq_len:
            continue
        train_part = seq[:-2]          # everything except val + test targets
        train_sequences.append(train_part)
        val_data.append((seq[:-2], seq[-2]))
        test_data.append((seq[:-1], seq[-1]))

    return Dataset(
        n_items=len(item2idx),
        train_sequences=train_sequences,
        val_data=val_data,
        test_data=test_data,
        item2idx=item2idx,
    )

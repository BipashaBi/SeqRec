"""
Loaders that turn real datasets into the unified
[user_id, item_id, timestamp] DataFrame the rest of the pipeline expects.

None of these download anything for you -- point them at a file you've
already downloaded. The synthetic generator is the default so the repo
runs out of the box; swap in one of these once you have real data.

Where to get data
-----------------
  * Amazon Reviews 2023 (McAuley Lab, UCSD):
      https://amazon-reviews-2023.github.io/
      Download a category's review file (JSON lines), e.g.
      `Electronics.jsonl`, then use `load_amazon_reviews(path)`.
  * RetailRocket (Kaggle):
      https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
      Use the `events.csv` file with `load_retailrocket(path)`.
"""
import pandas as pd


def from_csv(
    path: str,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """Generic loader for an interactions CSV with arbitrary column names."""
    df = pd.read_csv(path)
    df = df.rename(columns={user_col: "user_id",
                            item_col: "item_id",
                            time_col: "timestamp"})
    return df[["user_id", "item_id", "timestamp"]]


def load_amazon_reviews(path: str) -> pd.DataFrame:
    """Load an Amazon Reviews 2023 JSON-lines file (review-level)."""
    df = pd.read_json(path, lines=True)
    # field names in the 2023 dump
    df = df.rename(columns={"user_id": "user_id",
                            "parent_asin": "item_id",
                            "timestamp": "timestamp"})
    return df[["user_id", "item_id", "timestamp"]]


def load_retailrocket(path: str) -> pd.DataFrame:
    """Load RetailRocket events.csv (filters to 'view' events by default)."""
    df = pd.read_csv(path)
    df = df[df["event"] == "view"]
    df = df.rename(columns={"visitorid": "user_id",
                            "itemid": "item_id",
                            "timestamp": "timestamp"})
    return df[["user_id", "item_id", "timestamp"]]


def filter_core(df: pd.DataFrame, min_user: int = 3, min_item: int = 5) -> pd.DataFrame:
    """k-core filtering: drop rare items and short users (iterate to fixpoint)."""
    while True:
        n0 = len(df)
        item_counts = df["item_id"].value_counts()
        df = df[df["item_id"].isin(item_counts[item_counts >= min_item].index)]
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= min_user].index)]
        if len(df) == n0:
            return df

"""
Fit and evaluate all baselines on the test split.

    python run_baselines.py                 # synthetic data (default)
    python run_baselines.py --data retailrocket --path events.csv
"""
import argparse

from config import CONFIG
from data.synthetic import generate_synthetic
from data import loader
from src.dataset import build_dataset
from src.metrics import evaluate, format_results
from src.baselines.popularity import PopularityRecommender
from src.baselines.markov import MarkovRecommender
from src.baselines.item_knn import ItemKNNRecommender


def load_dataframe(args):
    if args.data == "synthetic":
        return generate_synthetic(seed=CONFIG.seed)
    if args.data == "amazon":
        df = loader.load_amazon_reviews(args.path)
    elif args.data == "retailrocket":
        df = loader.load_retailrocket(args.path)
    else:
        df = loader.from_csv(args.path)
    return loader.filter_core(df)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="synthetic",
                   choices=["synthetic", "amazon", "retailrocket", "csv"])
    p.add_argument("--path", default=None)
    args = p.parse_args()

    df = load_dataframe(args)
    ds = build_dataset(df, CONFIG.min_seq_len, CONFIG.max_seq_len)
    print(f"items={ds.n_items} | train_seqs={len(ds.train_sequences)} "
          f"| test={len(ds.test_data)}\n")

    models = [
        PopularityRecommender(),
        MarkovRecommender(),
        ItemKNNRecommender(),
    ]
    print("Test-set results (leave-one-out next-item):")
    for m in models:
        m.fit(ds.train_sequences, ds.n_items)
        res = evaluate(m.score, ds.test_data, ds.n_items,
                       CONFIG.k_list, CONFIG.exclude_seen)
        print(format_results(m.name, res))


if __name__ == "__main__":
    main()

# Sequential Product Recommender

A recommendation system that predicts the next product a user is likely to interact with based on their browsing session.

The project compares a deep learning model (**GRU4Rec**) against three traditional recommendation approaches on the same dataset.

## Models Compared

- **Popularity** – recommends the most popular items.
- **Markov(1)** – predicts the next item based on the last viewed item.
- **ItemKNN** – recommends items frequently viewed together.
- **GRU4Rec** – a recurrent neural network that learns from full user sessions.

## Results

### RetailRocket Dataset

| Model | Recall@10 | NDCG@10 | MRR |
|---------|---------|---------|---------|
| Popularity | 0.0082 | 0.0038 | 0.0041 |
| **Markov(1)** | **0.7568** | **0.5042** | **0.4372** |
| ItemKNN | 0.3276 | 0.1765 | 0.1461 |
| GRU4Rec | 0.4992 | 0.3789 | 0.3461 |

**Key Finding:** A simple Markov model outperformed the neural network because user behavior in this dataset is largely determined by the most recently viewed item.

### Synthetic Dataset

| Model | Recall@10 | NDCG@10 | MRR |
|---------|---------|---------|---------|
| Popularity | 0.066 | 0.038 | 0.038 |
| Markov(1) | 0.675 | 0.521 | 0.484 |
| ItemKNN | 0.603 | 0.292 | 0.213 |
| **GRU4Rec** | **0.783** | **0.629** | **0.580** |

**Key Finding:** GRU4Rec performs best when longer-term user behavior patterns exist, showing the value of sequence-aware neural models.

## Evaluation

Models are evaluated using:

- Recall@10
- NDCG@10
- Mean Reciprocal Rank (MRR)

A leave-one-out strategy is used, where the final item in each session is hidden and predicted from previous interactions.

## Tech Stack

- Python
- PyTorch
- NumPy
- SciPy

## Run Locally

```bash
pip install -r requirements.txt

# Synthetic dataset
python train.py

# RetailRocket dataset
python train.py --data retailrocket --path events.csv

# Train longer
python train.py --epochs 30

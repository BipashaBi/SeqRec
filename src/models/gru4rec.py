"""
GRU4Rec: a session-based sequential recommender.

  item ids -> embedding -> GRU -> linear head over the item vocabulary.

Trained with next-item cross-entropy at every timestep (padding ignored).
At inference we take the hidden state after the last input item and rank
all items by the output logits.
"""
import torch
import torch.nn as nn


class GRU4Rec(nn.Module):
    def __init__(self, n_items: int, emb_dim: int = 64, hidden_size: int = 128,
                 num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.n_items = n_items
        self.embedding = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.gru = nn.GRU(
            emb_dim, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, n_items + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) padded item ids
        emb = self.dropout(self.embedding(x))      # (B, T, E)
        out, _ = self.gru(emb)                      # (B, T, H)
        logits = self.head(self.dropout(out))       # (B, T, n_items+1)
        return logits

    @torch.no_grad()
    def score_sequence(self, seq, device="cpu"):
        """Return a numpy score vector (n_items+1) for a single input seq."""
        self.eval()
        x = torch.tensor([seq], dtype=torch.long, device=device)
        logits = self.forward(x)            # (1, T, n_items+1)
        return logits[0, -1].cpu().numpy()  # logits after the last item

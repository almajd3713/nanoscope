import torch
import torch.nn as nn
import torch.nn.functional as F


def window(emb, k):
    """For every position t, the embeddings of tokens t-k+1 .. t (zeros before the start of
    the text), side by side: (batch, time, d) -> (batch, time, k * d)."""
    padded = F.pad(emb, (0, 0, k - 1, 0))  # pad the time axis on the left
    return padded.unfold(1, k, 1).flatten(2).contiguous()  # (batch, time, d, k) -> flatten


class MyMLP(nn.Module):
    def __init__(self, vocab_size, d_model=32, window_size=8, hidden=256):
        super().__init__()
        self.window_size = window_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.up = nn.Linear(window_size * d_model, hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, idx):
        context = window(self.token_embedding(idx), self.window_size)
        return self.head(torch.relu(self.up(context)))

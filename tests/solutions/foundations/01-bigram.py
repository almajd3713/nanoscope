import torch.nn as nn


class MyBigram(nn.Module):
    def __init__(self, vocab_size, d_model=32):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        return self.head(self.token_embedding(idx))

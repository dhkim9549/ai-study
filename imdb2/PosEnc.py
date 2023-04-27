import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):  #@save
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = torch.unsqueeze(X, 0)
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        X = torch.squeeze(X)
        return self.dropout(X)


"""
pos_enc = PositionalEncoding(4, 0, 16)
x = torch.zeros(16, 4)
x = pos_enc(x)
print(x)
"""




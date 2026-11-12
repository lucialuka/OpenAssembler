"""Small attention/FFN block for full OA-native graph and gradient validation.

This is unmasked attention over supplied features, not a complete language model.
Shapes deliberately include tails and multiple heads to exercise native layouts.
"""
import torch


class AttentionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(8, 8)
        self.k = torch.nn.Linear(8, 8)
        self.v = torch.nn.Linear(8, 8)
        self.projection = torch.nn.Linear(8, 8)
        self.norm = torch.nn.LayerNorm(8)
        self.up = torch.nn.Linear(8, 16)
        self.down = torch.nn.Linear(16, 8)
        self.gelu = torch.nn.GELU(approximate='tanh')

    def forward(self, x):
        q = self.q(x).reshape(2, 7, 2, 4).permute(0, 2, 1, 3)
        k = self.k(x).reshape(2, 7, 2, 4).permute(0, 2, 1, 3)
        v = self.v(x).reshape(2, 7, 2, 4).permute(0, 2, 1, 3)
        probabilities = torch.softmax((q @ k.transpose(-2, -1)) * 0.5, dim=-1)
        context = (probabilities @ v).permute(0, 2, 1, 3).reshape(2, 7, 8)
        hidden = self.norm(self.projection(context) + x)
        return self.down(self.gelu(self.up(hidden))) + hidden


def create_model():
    return AttentionBlock()

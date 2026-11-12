"""Explicit mixed-precision native Tensor Core training example."""
import torch


class ResidualMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(128, 256)
        self.down = torch.nn.Linear(256, 128)

    def forward(self, x):
        return self.down(torch.relu(self.up(x))) + x


def create_model():
    return ResidualMLP()

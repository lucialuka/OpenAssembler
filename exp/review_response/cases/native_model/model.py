"""A residual MLP captured on CPU and compiled entirely by the OA backend."""
import torch


class ResidualMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(8, 16)
        self.down = torch.nn.Linear(16, 8)

    def forward(self, x):
        return self.down(torch.relu(self.up(x))) + x


def create_model():
    return ResidualMLP()

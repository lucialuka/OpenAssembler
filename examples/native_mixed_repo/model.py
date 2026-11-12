"""Python host model alongside independently compiled native source languages."""
import torch


def create_model():
    return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8))

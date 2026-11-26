"""Small FP32 CNN using native convolution, spatial mean and a linear head."""
import torch


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(2, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 3, 3, stride=2, padding=1),
            torch.nn.Sigmoid(),
        )
        self.head = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))


def create_model():
    return Classifier()

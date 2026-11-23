import torch

class ReductionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.project=torch.nn.Linear(7,5)
    def forward(self,x):
        y=torch.sigmoid(self.project(x))
        return y.mean(dim=1)+y.sum(dim=1)*.0625

def create_model():return ReductionNet()

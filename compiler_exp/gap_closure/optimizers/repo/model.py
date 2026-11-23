import torch
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__();self.first=torch.nn.Linear(7,5);self.second=torch.nn.Linear(5,3)
    def forward(self,x):return self.second(torch.sigmoid(self.first(x)))
def create_model():return Net()

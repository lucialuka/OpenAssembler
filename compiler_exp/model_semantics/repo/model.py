"""Small causal language block used for complete OA repository verification."""
import torch
from torch import nn
from torch.nn import functional as F

class LanguageBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(11,8)
        self.norm=nn.LayerNorm(8)
        self.up=nn.Linear(8,16)
        self.down=nn.Linear(16,8)
        self.readout=nn.Linear(8,11)
    def forward(self,tokens):
        x=self.embedding(tokens)
        x=x+F.scaled_dot_product_attention(x,x,x,is_causal=True)
        y=self.norm(x)
        x=x+self.down(F.gelu(self.up(y),approximate='tanh'))
        return self.readout(x).reshape(-1,11)

def create_model():return LanguageBlock()

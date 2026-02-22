import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
    
    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        var = x.var(-1,keepdim=True)
        x = (x-mean)/torch.sqrt(var+1e-13)
        y = self.gamma*x + self.beta
        return y
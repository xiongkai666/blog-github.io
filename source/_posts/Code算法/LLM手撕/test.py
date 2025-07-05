import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

def selfAttention(nn.Module):
    def __init__(self, input_dim, dim_k, dim_v):
        super(selfAttention, self).__init__()

        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)

        self.norm = sqrt(dim_k)

    def forward(self, x):

        Q = self.q(x)

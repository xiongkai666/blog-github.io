---
title: 大模型各种算子手撕（实现）
date: 2025-04-28 17:40:38
tags: 
- Python
- Pytorch
- LLM
categories: 面试手撕
---
## 注意力机制的代码实现-
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class SelfAttention(nn.Module):
    def __init__(self, input_dim, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)
        
        self.norm = sqrt(dim_k)
    
    def forward(self, x):
        Q = self.q(x) # [bs, seq_len, dim_k]
        K = self.k(x)
        V = self.v(x)
        atten = nn.Softmax(dim=-1)(torch.bmm(Q, K.permute(0, 2, 1)) / self.norm)
        #等价于atten = F.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.norm, dim=-1)
        output = torch.bmm(atten, V) # torch.matmul
        return output

if __name__ == "__main__":
    X = torch.randn(4, 3, 2)
    self_attention = SelfAttention(2, 4, 5)
    res = self_attention(X)
    print(res)
```

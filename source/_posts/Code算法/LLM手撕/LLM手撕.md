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
import math

# 1. 单头注意力机制（ScaledDotProductAttention）
class DotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -torch.inf)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights


# 2. 多头注意力（MultiHeadAttention）
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model # 输入维度，等价于hidden_size
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # head_dim = d_k表示每个头的维度

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = DotProductAttention(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        # 将每个头的维度拆分出来，形状：(batch_size, seq_len, num_heads, d_k)
        # 转置第1和第2维度，得到形状：(batch_size, num_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            #添加两个unsqueeze操作，使mask形状为(batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(1)

        output, attn_weights = self.attention(Q, K, V, mask)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)

        return output, attn_weights


# 自注意力机制函数（SelfAttention）
class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.multihead_attn = MultiHeadAttention(d_model, num_heads)

    def forward(self, x, mask=None):
        output, attn_weights = self.multihead_attn(x, x, x, mask)
        return output, attn_weights


# 示例使用
if __name__ == "__main__":
    
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    x = torch.randn(batch_size, seq_len, d_model)
    mask = None

    # 点积注意力
    dot_attn = DotProductAttention(d_k=d_model)
    dot_output, dot_attn_weights = dot_attn(x, x, x, mask)
    print("Dot Product Attention Output Shape:", dot_output.shape)

    # 多头注意力
    multihead_attn = MultiHeadAttention(d_model, num_heads)
    multihead_output, multihead_attn_weights = multihead_attn(x, x, x, mask)
    print("Multi - Head Attention Output Shape:", multihead_output.shape)

    # 自注意力
    self_attn = SelfAttention(d_model, num_heads)
    self_output, self_attn_weights = self_attn(x, mask)
    print("Self Attention Output Shape:", self_output.shape)
    
```

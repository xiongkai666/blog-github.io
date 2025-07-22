import torch
import torch.nn as nn
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
if __name__ == "__main__":
    batch_size = 2
    sequence_length = 3
    hidden_size = 4
    x = torch.randn(batch_size, sequence_length, hidden_size)
    rms_norm = RMSNorm(hidden_size)
    output = rms_norm(x)
    print(output.shape)
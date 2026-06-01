import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear_0 = nn.Linear(0, 64)
        self.Linear_1 = nn.Linear(0, 64)
        self.Linear_2 = nn.Linear(0, 64)
        self.Softmax_0 = nn.Softmax(dim=-1)
        self.Linear_3 = nn.Linear(0, 64)

    def forward(self, x):
        query = x
        key = x
        value = x
        q_proj = self.Linear_0(query)
        k_proj = self.Linear_1(key)
        v_proj = self.Linear_2(value)
        x = torch.matmul(q_proj, k_proj)
        attn_weights = self.Softmax_0(x)
        x = torch.matmul(attn_weights, v_proj)
        x = self.Linear_3(x)
        return x

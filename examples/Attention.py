import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear_0 = nn.Linear(64, 64)
        self.Linear_1 = nn.Linear(64, 64)
        self.Linear_2 = nn.Linear(64, 64)
        self.Softmax_0 = nn.Softmax(dim=-1)
        self.Linear_3 = nn.Linear(64, 64)

    def forward(self, query, key, value):
        # query: input  # [512, 64]
        # key: input  # [512, 64]
        # value: input  # [512, 64]
        q_proj = self.Linear_0(query)  # [512, 64]
        k_proj = self.Linear_1(key)  # [512, 64]
        v_proj = self.Linear_2(value)  # [512, 64]
        x = torch.matmul(q_proj, k_proj)  # [512, 64]
        attn_weights = self.Softmax_0(x)  # [512, 64]
        x = torch.matmul(attn_weights, v_proj)  # [512, 64]
        x = self.Linear_3(x)  # [512, 64]
        return x  # [512, 64]

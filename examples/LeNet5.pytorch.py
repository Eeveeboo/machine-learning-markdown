import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv2d_0 = nn.Conv2d(1, 6, 5, stride=1, padding=0)
        self.ReLU_0 = nn.ReLU()
        self.MaxPool_0 = nn.MaxPool2d(2, stride=2)
        self.Conv2d_1 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.ReLU_1 = nn.ReLU()
        self.MaxPool_1 = nn.MaxPool2d(2, stride=2)
        self.Flatten_0 = nn.Flatten()
        self.Linear_0 = nn.Linear(256, 120)
        self.ReLU_2 = nn.ReLU()
        self.Linear_1 = nn.Linear(120, 84)
        self.ReLU_3 = nn.ReLU()
        self.Linear_2 = nn.Linear(84, 10)

    def forward(self, x):
        # x: input  # [1, 28, 28]
        x = self.Conv2d_0(x)  # [6, 24, 24]
        x = self.ReLU_0(x)  # [6, 24, 24]
        x = self.MaxPool_0(x)  # [6, 12, 12]
        x = self.Conv2d_1(x)  # [16, 8, 8]
        x = self.ReLU_1(x)  # [16, 8, 8]
        x = self.MaxPool_1(x)  # [16, 4, 4]
        x = self.Flatten_0(x)  # [256]
        x = self.Linear_0(x)  # [120]
        x = self.ReLU_2(x)  # [120]
        x = self.Linear_1(x)  # [84]
        x = self.ReLU_3(x)  # [84]
        x = self.Linear_2(x)  # [10]
        return x  # [10]

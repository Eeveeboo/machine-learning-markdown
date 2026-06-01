import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv2d_0 = nn.Conv2d(0, 6, 5, stride=1, padding=0)
        self.ReLU_0 = nn.ReLU()
        self.MaxPool_0 = nn.MaxPool2d(2, stride=2)
        self.Conv2d_1 = nn.Conv2d(0, 16, 5, stride=1, padding=0)
        self.ReLU_1 = nn.ReLU()
        self.MaxPool_1 = nn.MaxPool2d(2, stride=2)
        self.Flatten_0 = nn.Flatten()
        self.Linear_0 = nn.Linear(0, 120)
        self.ReLU_2 = nn.ReLU()
        self.Linear_1 = nn.Linear(0, 84)
        self.ReLU_3 = nn.ReLU()
        self.Linear_2 = nn.Linear(0, 10)

    def forward(self, x):
        # x: input
        x = self.Conv2d_0(x)
        x = self.ReLU_0(x)
        x = self.MaxPool_0(x)
        x = self.Conv2d_1(x)
        x = self.ReLU_1(x)
        x = self.MaxPool_1(x)
        x = self.Flatten_0(x)
        x = self.Linear_0(x)
        x = self.ReLU_2(x)
        x = self.Linear_1(x)
        x = self.ReLU_3(x)
        x = self.Linear_2(x)
        return x

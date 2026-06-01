import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv2d_0 = nn.Conv2d(0, 64, 1, stride=1, padding=0)
        self.ReLU_0 = nn.ReLU()
        self.Conv2d_1 = nn.Conv2d(0, 96, 1, stride=1, padding=0)
        self.ReLU_1 = nn.ReLU()
        self.Conv2d_2 = nn.Conv2d(0, 128, 3, stride=1, padding=1)
        self.ReLU_2 = nn.ReLU()
        self.Conv2d_3 = nn.Conv2d(0, 16, 1, stride=1, padding=0)
        self.ReLU_3 = nn.ReLU()
        self.Conv2d_4 = nn.Conv2d(0, 32, 5, stride=1, padding=2)
        self.ReLU_4 = nn.ReLU()
        self.MaxPool_0 = nn.MaxPool2d(3, stride=1)
        self.Conv2d_5 = nn.Conv2d(0, 32, 1, stride=1, padding=0)
        self.ReLU_5 = nn.ReLU()

    def forward(self, x):
        b4 = x
        x = self.Conv2d_0(b4)
        out1 = self.ReLU_0(x)
        x = self.Conv2d_1(b4)
        x = self.ReLU_1(x)
        x = self.Conv2d_2(x)
        out2 = self.ReLU_2(x)
        x = self.Conv2d_3(b4)
        x = self.ReLU_3(x)
        x = self.Conv2d_4(x)
        out3 = self.ReLU_4(x)
        x = self.MaxPool_0(b4)
        x = self.Conv2d_5(x)
        out4 = self.ReLU_5(x)
        x = torch.cat([out1, out2, out3, out4], dim=0)
        return x

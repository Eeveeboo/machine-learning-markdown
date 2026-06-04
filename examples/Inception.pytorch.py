import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv2d_0 = nn.Conv2d(192, 64, 1, stride=1, padding=0)
        self.ReLU_0 = nn.ReLU()
        self.Conv2d_1 = nn.Conv2d(192, 96, 1, stride=1, padding=0)
        self.ReLU_1 = nn.ReLU()
        self.Conv2d_2 = nn.Conv2d(96, 128, 3, stride=1, padding=1)
        self.ReLU_2 = nn.ReLU()
        self.Conv2d_3 = nn.Conv2d(192, 16, 1, stride=1, padding=0)
        self.ReLU_3 = nn.ReLU()
        self.Conv2d_4 = nn.Conv2d(16, 32, 5, stride=1, padding=2)
        self.ReLU_4 = nn.ReLU()
        self.MaxPool_0 = nn.MaxPool2d(3, stride=1)
        self.Conv2d_5 = nn.Conv2d(192, 32, 1, stride=1, padding=0)
        self.ReLU_5 = nn.ReLU()

    def forward(self, b4):
        # b4: input  # [192, 28, 28]
        x = self.Conv2d_0(b4)  # [64, 28, 28]
        out1 = self.ReLU_0(x)  # [64, 28, 28]
        x = self.Conv2d_1(b4)  # [96, 28, 28]
        x = self.ReLU_1(x)  # [96, 28, 28]
        x = self.Conv2d_2(x)  # [128, 28, 28]
        out2 = self.ReLU_2(x)  # [128, 28, 28]
        x = self.Conv2d_3(b4)  # [16, 28, 28]
        x = self.ReLU_3(x)  # [16, 28, 28]
        x = self.Conv2d_4(x)  # [32, 28, 28]
        out3 = self.ReLU_4(x)  # [32, 28, 28]
        x = self.MaxPool_0(b4)  # [192, 28, 28]
        x = self.Conv2d_5(x)  # [32, 28, 28]
        out4 = self.ReLU_5(x)  # [32, 28, 28]
        x = torch.cat([out1, out2, out3, out4], dim=0)  # [256, 28, 28]
        return x  # [256, 28, 28]

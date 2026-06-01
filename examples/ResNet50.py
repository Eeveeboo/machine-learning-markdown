import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv2d_0 = nn.Conv2d(0, 64, 1, stride=1, padding=0)
        self.BatchNorm_0 = nn.BatchNorm1d(0)
        self.ReLU_0 = nn.ReLU()
        self.Conv2d_1 = nn.Conv2d(0, 64, 3, stride=1, padding=1)
        self.BatchNorm_1 = nn.BatchNorm1d(0)
        self.ReLU_1 = nn.ReLU()
        self.Conv2d_2 = nn.Conv2d(0, 256, 1, stride=1, padding=0)
        self.BatchNorm_2 = nn.BatchNorm1d(0)
        self.Conv2d_3 = nn.Conv2d(0, 256, 1, stride=1, padding=0)
        self.BatchNorm_3 = nn.BatchNorm1d(0)
        self.ReLU_2 = nn.ReLU()

    def forward(self, x):
        skip = x
        x = self.Conv2d_0(skip)
        x = self.BatchNorm_0(x)
        x = self.ReLU_0(x)
        x = self.Conv2d_1(x)
        x = self.BatchNorm_1(x)
        x = self.ReLU_1(x)
        x = self.Conv2d_2(x)
        main_out = self.BatchNorm_2(x)
        x = self.Conv2d_3(skip)
        skip_out = self.BatchNorm_3(x)
        x = main_out + skip_out
        x = self.ReLU_2(x)
        return x

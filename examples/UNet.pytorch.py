import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv2d_0 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.ReLU_0 = nn.ReLU()
        self.MaxPool_0 = nn.MaxPool2d(2, stride=2)
        self.Conv2d_1 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.ReLU_1 = nn.ReLU()
        self.MaxPool_1 = nn.MaxPool2d(2, stride=2)
        self.Conv2d_2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.ReLU_2 = nn.ReLU()
        self.TransposedConv2d_0 = nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0)
        self.Conv2d_3 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.ReLU_3 = nn.ReLU()
        self.TransposedConv2d_1 = nn.ConvTranspose2d(64, 32, 2, stride=2, padding=0)
        self.Conv2d_4 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.ReLU_4 = nn.ReLU()
        self.Conv2d_5 = nn.Conv2d(32, 1, 1, stride=1, padding=0)

    def forward(self, x):
        # x: input  # [1, 64, 64]
        x = self.Conv2d_0(x)  # [32, 64, 64]
        enc1_skip = self.ReLU_0(x)  # [32, 64, 64]
        x = self.MaxPool_0(enc1_skip)  # [32, 32, 32]
        x = self.Conv2d_1(x)  # [64, 32, 32]
        enc2_skip = self.ReLU_1(x)  # [64, 32, 32]
        x = self.MaxPool_1(enc2_skip)  # [64, 16, 16]
        x = self.Conv2d_2(x)  # [128, 16, 16]
        bottleneck = self.ReLU_2(x)  # [128, 16, 16]
        up1 = self.TransposedConv2d_0(bottleneck)  # [64, 32, 32]
        x = torch.cat([up1, enc2_skip], dim=0)  # [128, 32, 32]
        x = self.Conv2d_3(x)  # [64, 32, 32]
        dec1 = self.ReLU_3(x)  # [64, 32, 32]
        up2 = self.TransposedConv2d_1(dec1)  # [32, 64, 64]
        x = torch.cat([up2, enc1_skip], dim=0)  # [64, 64, 64]
        x = self.Conv2d_4(x)  # [32, 64, 64]
        x = self.ReLU_4(x)  # [32, 64, 64]
        x = self.Conv2d_5(x)  # [1, 64, 64]
        return x  # [1, 64, 64]

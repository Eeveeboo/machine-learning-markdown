import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv2d_0 = nn.Conv2d(0, 32, 3, stride=1, padding=1)
        self.ReLU_0 = nn.ReLU()
        self.MaxPool_0 = nn.MaxPool2d(2, stride=2)
        self.Conv2d_1 = nn.Conv2d(0, 64, 3, stride=1, padding=1)
        self.ReLU_1 = nn.ReLU()
        self.MaxPool_1 = nn.MaxPool2d(2, stride=2)
        self.Conv2d_2 = nn.Conv2d(0, 128, 3, stride=1, padding=1)
        self.ReLU_2 = nn.ReLU()
        self.TransposedConv2d_0 = nn.ConvTranspose2d(0, 64, 2, stride=2, padding=0)
        self.Conv2d_3 = nn.Conv2d(0, 64, 3, stride=1, padding=1)
        self.ReLU_3 = nn.ReLU()
        self.TransposedConv2d_1 = nn.ConvTranspose2d(0, 32, 2, stride=2, padding=0)
        self.Conv2d_4 = nn.Conv2d(0, 32, 3, stride=1, padding=1)
        self.ReLU_4 = nn.ReLU()
        self.Conv2d_5 = nn.Conv2d(0, 1, 1, stride=1, padding=0)

    def forward(self, x):
        # x: input
        x = self.Conv2d_0(x)
        enc1_skip = self.ReLU_0(x)
        x = self.MaxPool_0(enc1_skip)
        x = self.Conv2d_1(x)
        enc2_skip = self.ReLU_1(x)
        x = self.MaxPool_1(enc2_skip)
        x = self.Conv2d_2(x)
        bottleneck = self.ReLU_2(x)
        up1 = self.TransposedConv2d_0(bottleneck)
        x = torch.cat([up1, enc2_skip], dim=0)
        x = self.Conv2d_3(x)
        dec1 = self.ReLU_3(x)
        up2 = self.TransposedConv2d_1(dec1)
        x = torch.cat([up2, enc1_skip], dim=0)
        x = self.Conv2d_4(x)
        x = self.ReLU_4(x)
        x = self.Conv2d_5(x)
        return x

import pytest
import torch


def test_forward():
    model = UNet()
    model.eval()
    x = torch.randn(1, 1, 64, 64)
    output = model(=)
    assert output.shape == torch.Size([1, 1, 64, 64])

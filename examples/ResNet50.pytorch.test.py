import pytest
import torch


def test_forward():
    model = ResNet50()
    model.eval()
    x = torch.randn(1, 256, 56, 56)
    output = model(=)
    assert output.shape == torch.Size([1, 256, 56, 56])

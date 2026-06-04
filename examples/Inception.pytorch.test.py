import pytest
import torch


def test_forward():
    model = Inception()
    model.eval()
    x = torch.randn(1, 192, 28, 28)
    output = model(=)
    assert output.shape == torch.Size([1, 256, 28, 28])

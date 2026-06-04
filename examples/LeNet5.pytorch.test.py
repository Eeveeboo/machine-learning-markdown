import pytest
import torch


def test_forward():
    model = LeNet5()
    model.eval()
    x = torch.randn(1, 1, 28, 28)
    output = model(=)
    assert output.shape == torch.Size([1, 10])

import pytest
import torch


def test_forward():
    model = Attention()
    model.eval()
    x = torch.randn(1, 512, 64)
    x2 = torch.randn(1, 512, 64)
    x3 = torch.randn(1, 512, 64)
    output = model(=, =, =)
    assert output.shape == torch.Size([1, 512, 64])

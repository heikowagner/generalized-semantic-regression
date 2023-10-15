from RiskBERT import poissonLoss
import torch


def test_poissonLoss():
    expected = torch.tensor(219.9123)
    xbeta = torch.Tensor([1, 1, 2, 3, 5, 6, 7])
    y = torch.Tensor([4, 3, 4, 3, 7, 8, 5])
    actual = poissonLoss(xbeta, y)
    torch.testing.assert_close(expected, actual)

from RiskBERT import poissonLoss, gammaLoss, paretoLoss
import torch


def test_poissonLoss():
    expected = torch.tensor(219.9123)
    xbeta = torch.Tensor([1, 1, 2, 3, 5, 6, 7])
    y = torch.Tensor([4, 3, 4, 3, 7, 8, 5])
    actual = poissonLoss(xbeta, y)
    torch.testing.assert_close(expected, actual)

def test_gammaLoss():
    expected = torch.tensor(-3028.1316)
    xbeta = torch.Tensor([42.3, 3.7, 1.1, 39, 93.1, 99.4, 39.8])
    y = torch.Tensor([45.3, 3.4, 4.1, 30, 73.1, 89.4, 59.8])
    actual = gammaLoss(xbeta, y)
    torch.testing.assert_close(expected, actual)

def test_paretoLoss():
    expected = torch.tensor(3.4031083583831787)
    xbeta = torch.Tensor([42.3, 3.7, 1.1, 39, 93.1, 99.4, 39.8])
    y = torch.Tensor([45.3, 3.4, 4.1, 30, 73.1, 89.4, 59.8])
    actual = paretoLoss(xbeta, y)
    torch.testing.assert_close(expected, actual)

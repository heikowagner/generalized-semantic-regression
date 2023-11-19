#
# %%
import torch


def generalExponentialLoss(xbeta, y, b, a=lambda x: x, phi=1):
    """Generates the neg. log liklyhood loss of for
    targets distributed by the exponential family.

    f(y|theta, u) = exp((y theta - b(phi))/a(phi)+ c(y, phi) ))
    phi  float
        dispersion paramter
    a
        function
    b
        function

    """
    return -torch.mean(torch.div(torch.sub(torch.mul(y, xbeta), b(xbeta)), a(phi)))


# y = torch.randn(128, 20)
# xbeta = torch.randn(128, 20)
# generalExponentialLoss(xbeta, y, b=torch.exp)

# %%
def poissonLoss(xbeta, y):
    """Loss function for Poisson model."""
    # loss = -torch.mean(y * xbeta - torch.exp(xbeta))
    loss = generalExponentialLoss(xbeta=xbeta, y=y, b=torch.exp)
    return loss


# y = torch.randn(128, 20)
# xbeta = torch.randn(128, 20)
# poissonLoss(xbeta, y)

# %%
def gammaLoss(xbeta, y):
    """Loss function for Gamma model with dispersion parameter phi=1.
    .. math::
        \[l(Y|X, \zeta)= \sum_{i=1}^m \sum_{j=1}^{N_i} \frac{- y_{ij}exp(\zeta^T X_i)^{-1} -exp(\zeta^T X_i)}{\phi}
    """
    a = lambda x: x
    b = lambda x: -torch.log(x)
    loss = generalExponentialLoss(xbeta=xbeta, y=y, b=b, a=a)
    return loss


# %%
def paretoLoss(xbeta, y):
    """Loss function for Poisson model.
    The MLE estimator xbeta is always min(y). Using this loss function is
    an inefficient way to fit the BERT.
    """
    a = 1
    loss = -torch.mean(torch.log(xbeta) - torch.mul((a + 1), torch.log(y)))
    return loss

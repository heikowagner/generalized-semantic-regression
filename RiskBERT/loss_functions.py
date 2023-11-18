#
# %%
import torch

def generalExponentialLoss(xbeta, y , b, a=lambda x: x, phi=1):
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
    return -torch.mean ( torch.div( torch.sub(torch.mul(y, xbeta),b(xbeta)), a(phi)))

y = torch.randn(128, 20)
xbeta = torch.randn(128, 20)
generalExponentialLoss(xbeta, y, b=torch.exp)

# %%
def poissonLoss(xbeta, y):
    """Loss function for Poisson model."""
    #loss = -torch.mean(y * xbeta - torch.exp(xbeta))
    loss = generalExponentialLoss(xbeta, y, b=torch.exp)
    return loss

y = torch.randn(128, 20)
xbeta = torch.randn(128, 20)
poissonLoss(xbeta, y)

# %%
def gammaLoss(xbeta, y):
    """Loss function for Poisson model.
    .. math::
        \[l(Y|X, \zeta)= \sum_{i=1}^m \sum_{j=1}^{N_i} \frac{- y_{ij}exp(\zeta^T X_i)^{-1} -exp(\zeta^T X_i)}{\phi}
    """
    a= lambda x:x
    b=lambda x: -torch.log(x)
    loss = generalExponentialLoss(xbeta, y, b=b ,a=a)
    return loss

y = torch.absolute( torch.randn(128, 20) )
xbeta = torch.absolute( torch.randn(128, 20) )
gammaLoss(xbeta, y)

# %%
def paretoLoss(xbeta, x_m, y):
    """Loss function for Poisson model."""
    #loss = -torch.mean(y * xbeta - torch.exp(xbeta))
    b=torch.log
    #loss = - nln xbeta - n xbeta ln xm+(xbeta + 1)* torch.mean(np.log(y))

    return loss
#ℓ(α,δ)=nlogδ−nδlogα−(δ+1)∑i=1nlogxi for α≥min{x1,…,xn}.
# https://math.stackexchange.com/questions/3174150/find-the-maximum-likelihood-estimator-for-pareto-distribution-and-a-unbiased-est
theta = log theta..

#theta alpha^theta/ x^(theta +1)
# %%

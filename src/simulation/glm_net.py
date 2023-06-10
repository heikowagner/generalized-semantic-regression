# %%
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
from torch import nn
import torch
import matplotlib.pyplot as plt
# %% 
# %%
N=5000
x1 = np.random.normal(size=N)           # some continuous variables 
x2 = np.random.normal(size=N)

z = np.exp(1 + 2*x1 + 3*x2) 
Y= np.random.poisson(lam=z, size=N) #lam is lamda_i
# %%
columns = ["x1", "x2"]
X = pd.DataFrame(data=zip(x1,x2))
X["intercept"]=1
model = sm.GLM(Y, X, family=sm.families.Poisson()) 
result = model.fit()

# %%
result.summary()
# %%
result.predict()

# %%
net = nn.Linear(3, 1, bias=False)

def poissonLoss(xbeta, y):
    """Custom loss function for Poisson model."""
    loss=-torch.mean(y*xbeta - torch.exp(xbeta))
    return loss


#loss = nn.PoissonNLLLoss()
loss = poissonLoss

trainer = torch.optim.SGD(net.parameters(), lr=0.0001)

# %%
# creating tensor from targets_df 
Y_tensor = torch.tensor(Y)
X_tensor = torch.tensor(X.values).float()
#data_utils.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())

# %%
num_epochs = 100
 
for epoch in range(num_epochs):
    my_model = net(X_tensor) 
    l = loss(my_model ,Y_tensor)
    print(l)
    trainer.zero_grad() #sets gradients to zero
    l.backward() # back propagation
    trainer.step() # parameter update

l = loss(net(X_tensor), Y_tensor)
print(f'epoch {epoch + 1}, loss {l:f}')

# %%

#Results
m = net.weight.data
# print('error in estimating m:', true_m - m.reshape(true_m.shape))
#c = net.bias.data
# print('error in estimating c:', true_c - c)
print(m)

# %%
m = nn.Linear(2, 1, bias=False)
input = torch.randn(1, 2)
output = m(input)
print(input)
print(m.weight.data)
print(output)

# %%
torch.sum( input*m.weight.data )
# %%
print(z)
print( torch.exp(net(X_tensor)) )


# %%
## Test with linear Model
# Creating a function f(X) with a slope of -5
# X = torch.arange(-5, 5, 0.1).view(-1, 1)
func = torch.matmul(X_tensor,torch.tensor([-5,0.3,2])) #-5 * X

#%%
# Plot the line in red with grids
plt.plot(X_tensor.numpy(), func.numpy(), 'r', label='func')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid('True', color='y')
plt.show()

# %%

...
# Adding Gaussian noise to the function f(X) and saving it in Y
Y = func + 0.4 * torch.randn(X_tensor.size()[0])


net = nn.Linear(3, 1, bias=False)

#loss = nn.PoissonNLLLoss()
loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# %%
print("The parameters: ", list(net.parameters()))

# %%
num_epochs = 200
 
for epoch in range(num_epochs):
    my_model = net(X_tensor) 
    l = loss(my_model ,Y)
    print(l)
    trainer.zero_grad() #sets gradients to zero
    l.backward() # back propagation
    trainer.step() # parameter update

l = loss(net(X_tensor), Y)
print(f'epoch {epoch + 1}, loss {l:f}')

print(net.weight.data)
# %%


import torch
from torch.autograd import Variable
 
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))
 
 
class LinearRegressionModel(torch.nn.Module):
 
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out
 
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
 
# our model
our_model = LinearRegressionModel()
 
criterion = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.01)
 
for epoch in range(500):
 
    # Forward pass: Compute predicted y by passing
    # x to the model
    pred_y = our_model(x_data)
 
    # Compute and print loss
    loss = criterion(pred_y, y_data)
 
    # Zero gradients, perform a backward pass,
    # and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))
 
new_var = Variable(torch.Tensor([[4.0]]))
pred_y = our_model(new_var)
print("predict (after training)", 4, our_model(new_var).item())

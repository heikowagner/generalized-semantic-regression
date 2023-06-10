# %%
# Importing libraries and packages
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
torch.manual_seed(42)

m =400

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# %%
from datasets import load_dataset


# Creating the dataset class
class Data(Dataset):
    # Constructor
    def __init__(self):
        self.x = torch.zeros(m, 2)
        self.x[:, 0] = torch.randn(m) # torch.arange(-2, 2, 0.1)
        self.x[:, 1] = torch.randn(m) #torch.arange(-2, 2, 0.1)
        self.w = torch.tensor([[1.0], [2.5]])
        dataset = load_dataset("glue", "ax")

        score1=model.encode(dataset['test']['premise'])
        #scorem1=-model.encode(dataset['test']['hypothesis'])
        scorem1=-model.encode(dataset['test']['premise'])

        # But we keep the negative input as sentence
        sentences = dataset['test']['premise'] + dataset['test']['hypothesis']

        self.sentence_data={
            "Embeddings" :[*score1, *scorem1],
            "Sentences": sentences
        }

        sentence_embeddings = pd.DataFrame(self.sentence_data)
        sentence_sample = sentence_embeddings.sample(m, replace=True)
        embeddings=torch.tensor(np.array( list(sentence_sample["Embeddings"])))
        # the mult will be added to the scores

        print(embeddings.shape)

        embed_scores= torch.randn(len(sentence_embeddings["Embeddings"][0]),1)

        self.b = 1
        self.func = torch.mm(self.x, self.w) + torch.mm(embeddings, embed_scores) +self.b    
        self.y = torch.poisson( torch.exp(self.func) )
        self.len = self.x.shape[0]
    # Getter
    def __getitem__(self, index):          
        return self.x[index], self.y[index] 
    # getting data length
    def __len__(self):
        return self.len
 
# Creating dataset object
data_set = Data()
 
# Creating a custom Multiple Linear Regression Model
class MultipleLinearRegression(torch.nn.Module):
    # Constructor
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    # Prediction
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
 
# Creating the model object
MLR_model = MultipleLinearRegression(2,1)
# defining the model optimizer
optimizer = torch.optim.SGD(MLR_model.parameters(), lr=0.0001)
# defining the loss criterion
criterion = torch.nn.PoissonNLLLoss() #torch.nn.MSELoss()
# Creating the dataloader
train_loader = DataLoader(dataset=data_set, batch_size=2)
 
# Train the model
Loss = []
epochs = 100
for epoch in range(epochs):
    for x,y in train_loader:
        y_pred = MLR_model(x)
        loss = criterion(y_pred, y)
        Loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   
    print(f"epoch = {epoch}, loss = {loss}")
print("Done training!")
 
# Plot the graph for epochs and loss
plt.plot(Loss)
plt.xlabel("Iterations ")
plt.ylabel("total loss ")
plt.show()

# %%
print( list(MLR_model.parameters()) )
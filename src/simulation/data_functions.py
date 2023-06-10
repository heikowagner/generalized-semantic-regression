# %%
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
# %%

def sentence_generator():
    dataset = load_dataset("glue", "ax")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # We use the sentence model to generate embeddings. The loaded sentences  are of the form premise is the sentence while hypothesis is the negation
    # Since we assume that the negation has the inverse impact on the risk, for simulation, we do not use the embedding of the negation rather than the negative
    # Embedding of the original sentence. 
    score1=model.encode(dataset['test']['premise'])
    scorem1=-model.encode(dataset['test']['premise'])

    # However as input we memory the positive as well as the negation sentences.
    senetences = dataset['test']['premise'] + dataset['test']['hypothesis']

    data={
        "Embeddings" :[*score1, *scorem1],
        "Sentences": senetences
    }
    return pd.DataFrame(data)
# %%
def generate_training(N=5000, seed=123):
    np.random.seed(seed)
    sentence_embeddings = sentence_generator()
    # We assume that apart from the free text sentence our model has 2 covariates x1, x2
    x1 = np.random.normal(size=N)
    x2 = np.random.normal(size=N)
    sentence_sample = sentence_embeddings.sample(N, replace=True)
    embeddings=pd.DataFrame( list(sentence_sample["Embeddings"]) )
    embed_scores= np.random.normal(size=len(sentence_embeddings["Embeddings"][0]))
    lambda_i = np.exp(1 + 2*x1 + 3*x2 + np.matmul(embeddings,embed_scores))

    X = pd.DataFrame(data=zip(x1,x2), columns=["x1", "x2"])
    Y= np.random.poisson(lam=lambda_i, size=N)
    return [Y,X, sentence_sample["Sentences"]]

# my_sample = generate_training() 
        
# %%
#my_sample 
# %%
class Data(Dataset):
    # Constructor
    def __init__(self, N=5000):
        self.x = torch.zeros(N, 2)
        self.x[:, 0] = torch.randn(N) # torch.arange(-2, 2, 0.1)
        self.x[:, 1] = torch.randn(N) #torch.arange(-2, 2, 0.1)
        # scores
        self.w = torch.tensor([[1.0], [3.0]])

        sentence_embeddings = sentence_embeddings = sentence_generator()
        self.sentence_sample = sentence_embeddings.sample(N, replace=True).reset_index()
        embeddings=torch.tensor(np.array( list(self.sentence_sample["Embeddings"])))
        embed_scores= torch.randn(len(sentence_embeddings["Embeddings"][0]),1)
        self.b = 1 #intercept
        self.lambda_i = torch.mm(self.x, self.w) + torch.mm(embeddings, embed_scores) +self.b    
        self.y = torch.poisson( torch.exp(self.lambda_i) )
        self.len = self.x.shape[0]
    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.sentence_sample["Sentences"][index]
    # getting data length
    def __len__(self):
        return self.len

# my_sample2 = Data()

#%%
# my_sample2.__getitem__(1)
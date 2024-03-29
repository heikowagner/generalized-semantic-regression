# %%
import numpy as np
import torch
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from RiskBERT import DataConstructor

# %%


def sentence_generator(weight=1):
    dataset = load_dataset("glue", "ax")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # We use the sentence model to generate embeddings. The loaded sentences  are of the form premise is the sentence while hypothesis is the negation
    # Since we assume that the negation has the inverse impact on the risk, for simulation, we do not use the embedding of the negation rather than the negative
    # Embedding of the original sentence.
    score1 = weight * model.encode(dataset["test"]["premise"])
    scorem1 = -weight * model.encode(dataset["test"]["premise"])
    # scorem1=model.encode(dataset['test']['hypothesis'])

    # However as input we memory the positive as well as the negation sentences.
    senetences = dataset["test"]["premise"] + dataset["test"]["hypothesis"]

    data = {"Embeddings": [*score1, *scorem1], "Sentences": senetences}
    return pd.DataFrame(data)


# %%
def generate_training(N=5000, seed=123):
    np.random.seed(seed)
    sentence_embeddings = sentence_generator()
    # We assume that apart from the free text sentence our model has 2 covariates x1, x2
    x1 = np.random.normal(size=N)
    x2 = np.random.normal(size=N)
    sentence_sample = sentence_embeddings.sample(N, replace=True)
    embeddings = pd.DataFrame(list(sentence_sample["Embeddings"]))
    embed_scores = np.random.normal(size=len(sentence_embeddings["Embeddings"][0]))
    lambda_i = np.exp(1 + 2 * x1 + 3 * x2 + np.matmul(embeddings, embed_scores))

    X = pd.DataFrame(data=zip(x1, x2), columns=["x1", "x2"])
    Y = np.random.poisson(lam=lambda_i, size=N)
    return [Y, X, sentence_sample["Sentences"]]


# %%
class Data(DataConstructor):
    # Constructor
    def __init__(self, N=5000, num_sentences=None, scores=torch.tensor([[1.0], [3.0]]), intercept=0, weigth=1):
        if num_sentences is None:
            num_sentences = [1] * N
        self.x = torch.zeros(N, len(scores))
        for i in range(0, len(scores)):
            self.x[:, i] = torch.randn(N)  # torch.arange(-2, 2, 0.1)
        # scores
        self.w = scores

        sentence_embeddings = sentence_generator(weigth)

        total_num_sentences = sum(num_sentences)
        self.sentence_sample = sentence_embeddings.sample(N * total_num_sentences, replace=True).reset_index()

        # We have to pack embeddings and scores according to the number of sentences
        i = 0
        m = 0
        embeddings = [None] * N
        sentences = [None] * N
        for j in num_sentences:
            sentences[m] = list(self.sentence_sample["Sentences"][i : i + j])
            embeddings[m] = np.sum(self.sentence_sample["Embeddings"][i : i + j], axis=0)
            i = i + j
            m = m + 1
        self.embeddings = torch.tensor(embeddings)
        self.embed_scores = torch.rand(len(sentence_embeddings["Embeddings"][0]), 1)
        self.sentence_sample = sentences

        self.b = intercept
        self.lambda_i = torch.mm(self.x, self.w) + torch.mm(self.embeddings, self.embed_scores) + self.b
        self.y = torch.poisson(torch.exp(self.lambda_i))
        self.len = self.x.shape[0]
        self.num_sentences = num_sentences

        self.labels = self.y
        self.covariates = self.x
        self.sentences = self.sentence_sample

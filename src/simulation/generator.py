# %%
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
from torch import nn
# %% 
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# %%
from datasets import load_dataset
dataset = load_dataset("glue", "ax")

# %%
# As scores we use the same embeddings for the negativ but with a minus
# this corresponds to a negativ score impling the opposit effect

score1=model.encode(dataset['test']['premise'])
#scorem1=-model.encode(dataset['test']['hypothesis'])
scorem1=-model.encode(dataset['test']['premise'])

# But we keep the negative input as sentence
senetences = dataset['test']['premise'] + dataset['test']['hypothesis']

#%%
data={
    "Embeddings" :[*score1, *scorem1],
    "Sentences": senetences
}
#%%
data
# %%
sentence_embeddings = pd.DataFrame(data)
sentence_embeddings

# %%
N=5000
x1 = np.random.normal(size=N)           # some continuous variables 
x2 = np.random.normal(size=N)
embed_scores= np.random.normal(size=len(sentence_embeddings["Embeddings"][0]))

# %%
# Draw random sentences
np.random.seed(123)
sentence_sample = sentence_embeddings.sample(N, replace=True)

# %%
import torch

# %%
embeddings = torch.tensor(np.array( list(sentence_sample["Embeddings"])))
#torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))

# %%

embed_scores= torch.randn(len(sentence_embeddings["Embeddings"][0]),1)
torch.mm( embeddings, embed_scores) 

#%%
embeddings=pd.DataFrame( list(sentence_sample["Embeddings"]) )

# %%
np.matmul(embeddings,embed_scores)
# %%
z = np.exp(1 + 2*x1 + 3*x2 + np.matmul(embeddings,embed_scores)) 
Y= np.random.poisson(lam=z, size=N) #lam is lamda_i
# %%
columns = ["x1", "x2"]
X = pd.DataFrame(data=zip(x1,x2))
                 
model = sm.GLM(Y, X, family=sm.families.Poisson()) 
result = model.fit()

# %%
result.summary()
# %%
result.predict()
# %%
import matplotlib.pyplot as plt

plt.scatter(Y, result.predict())

# %%
# But we keep the negative input as sentence
def glm_w_embdding(Y, X, sentences,senctence_model):
    senctence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = senctence_model.encode(sentences)
    X2 = pd.concat([X, pd.DataFrame( list( embeddings ) )], axis=1)
    return sm.GLM(Y, X2, family=sm.families.Poisson()) 

senetences = pd.DataFrame([*dataset['test']['premise'],*dataset['test']['hypothesis']])
s_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model = glm_w_embdding(Y, X, list(sentence_sample["Sentences"]), s_model)

# %%
result = model.fit()
result.summary()
# %%
plt.scatter(Y, result.predict())

# %%
def poissonLoss(xbeta, y):
    """Custom loss function for Poisson model."""
    loss=torch.mean(torch.exp(xbeta)-y*xbeta)
    return loss

# %%


# PyTorch models inherit from torch.nn.Module
class GLMNet(nn.Module):
    def __init__(self):
        super(GLMNet, self).__init__()
        self.lin = nn.Linear(3, 1)

    def forward(self, x):
        x = self.lin(x)
        return x


net = GLMNet()
#net = nn.Linear(3, 1)

#Initialize model params
 
#net.weight.data.normal_(0, 0.01)
 
#net.bias.data.fill_(0)
# loss = poissonLoss
#loss = nn.PoissonNLLLoss()
loss = nn.L1Loss()

trainer = torch.optim.SGD(net.parameters(), lr=0.00003)

# %%

import torch

# creating tensor from targets_df 
Y_tensor = torch.tensor(Y)
X["intercept"]=1
X_tensor = torch.tensor(X.values).float()
#data_utils.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())

# %%

m = nn.Tanh()
input = torch.randn(7)
output = m(input)

print("This is the input:",input)
print("This is the output:",output)

my_o = torch.exp

# %%
my_model = my_o(m(net(X_tensor)))

# %%
num_epochs = 50
 
for epoch in range(num_epochs):
    #my_model = my_o(m(net(X_tensor)))
    my_model = net(X_tensor)
    l = loss(my_model ,Y_tensor)
    print(l)
    trainer.zero_grad() #sets gradients to zero
    l.backward() # back propagation
    trainer.step() # parameter update
    #l = loss(net(features), labels)
 
print(f'epoch {epoch + 1}, loss {l:f}')

# %%

#Results
m = net.weight.data
# print('error in estimating m:', true_m - m.reshape(true_m.shape))
c = net.bias.data
# print('error in estimating c:', true_c - c)
print(m)
#nn.functional.poisson_nll_los


## All above is how to estimate a classical GLM using the torch neuronal net library.
# %%
from transformers import BertPreTrainedModel, AutoConfig, AutoModel
import torch

#class ExtendedTransformer(BertPreTrainedModel):
#    def __init__(self, config):
#        super().__init__(config)
#        self.bert = AutoModel.from_pretrained('bert-base-uncased')
#        self.linear = torch.nn.Linear(self.config.hidden_size, 128, bias=False)
#        self.post_init()

#https://discuss.huggingface.co/t/finetune-pretrained-bert-for-custom-regression-task/18562/8
from transformers import BertTokenizer, BertModel

class RiskBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = BertModel.from_pretrained(config)
        self.config = AutoConfig.from_pretrained(config)
        self.output = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids,
        covariates,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
    ):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        sequence_output = outputs.last_hidden_state
        outputs = self.output(sequence_output+covariates)

        # if labels, then we are training
        loss = None
        if labels is not None:
            # add custom neg log likelyhood here
            loss_fn = poissonLoss # nn.MSELoss()
            loss = loss_fn(outputs, labels)
            
        return {
            "loss": loss,
            "logits": outputs
        }



model = RiskBertModel("bert-base-uncased")

# %%
model_name = "roberta-base"
cfg = AutoConfig.from_pretrained(model_name)
cfg.update({
 "num_labels": 5
})

model = get_pretrained(cfg, model_name)

# %%
model = BertModel.from_pretrained("bert-base-uncased")
# %%
model
# %%

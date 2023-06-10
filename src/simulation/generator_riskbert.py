# %%
# Importing libraries and packages
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertPreTrainedModel, AutoConfig, AutoModel, AutoTokenizer

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

# %%

class RiskBertModel(BertPreTrainedModel):
    def __init__(self, config, input_dim):
        super().__init__(config)
        self.backbone = AutoModel.from_config(config)
        #self.output = nn.Linear(config.hidden_size, config.num_labels)
        self.output = torch.nn.Linear(input_dim+config.hidden_size, 1)

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
            loss_fn = torch.nn.PoissonNLLLoss()
            loss = loss_fn(outputs, labels)
            
        return {
            "loss": loss,
            "logits": outputs
        }


# %%
config = AutoConfig.from_pretrained('bert-base-uncased')
MLR_model = RiskBertModel(config,2)

optimizer = torch.optim.SGD(MLR_model.parameters(), lr=0.0001)
# defining the loss criterion
criterion = torch.nn.PoissonNLLLoss() #torch.nn.MSELoss()
# Creating the dataloader
train_loader = DataLoader(dataset=data_set, batch_size=2)

#%%
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# %%

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


# %%
model = AutoModel.from_pretrained('bert-base-uncased')
#https://huggingface.co/docs/transformers/training
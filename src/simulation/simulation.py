# %%
import torch
from transformers import PreTrainedModel, BertPreTrainedModel, AutoModel, AutoConfig, AutoTokenizer, BertModel
import matplotlib.pyplot as plt
from transformers import AdamW
from tqdm.auto import tqdm
import pandas as pd
from torch.utils.data import DataLoader

from models import glmModel, RiskBertModel
from utils import trainer, evaluate_model, evaluate_model_glm, print_params, visualize_model
from data_functions import Data

# %%
# Set device to gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% 
# simulate data
model_dataset = Data(20000, scores=torch.tensor([[0.2],[0.4]]), weigth=5)
plt.plot(model_dataset.y)

# %%
# choose pretrained model
pre_model= "distilbert-base-uncased"

# %%

glm_model = glmModel(2).to(device)

glm_model, Total_Loss_glm, Validation_Loss_glm =  trainer(model =glm_model, 
        model_dataset = model_dataset, 
        epochs=150, 
        evaluate_fkt=evaluate_model_glm,
        batch_size=500,
        tokenizer=  AutoTokenizer.from_pretrained(pre_model), 
        optimizer = torch.optim.SGD(glm_model.parameters(), lr=0.001)
        )
#%%
model = RiskBertModel(model=pre_model, input_dim=2, dropout=0.4, freeze_bert=True, mode="CLS").to(device)
model, Total_Loss, Validation_Loss = trainer(model =model, 
        model_dataset = model_dataset, 
        epochs=100,
        batch_size=1000,
        evaluate_fkt=evaluate_model,
        tokenizer=  AutoTokenizer.from_pretrained(pre_model), 
        optimizer =  torch.optim.SGD(model.parameters(), lr=0.001)
        )

 # %%
import gc
torch.cuda.empty_cache()
gc.collect()
# %%

model_unfreeze = RiskBertModel(model=pre_model, input_dim=2, dropout=0.4, freeze_bert=False, mode="CLS").to(device)
model_unfreeze, Total_Loss_unfreeze, Validation_Loss_unfreeze = trainer(model =model_unfreeze, 
        model_dataset = model_dataset, 
        epochs=100,
        batch_size=250,
        evaluate_fkt=evaluate_model,
        tokenizer=  AutoTokenizer.from_pretrained(pre_model), 
        optimizer =  torch.optim.SGD(model_unfreeze.parameters(), lr=0.001)
        )

# %%
visualize_model(model_unfreeze, model_dataset, AutoTokenizer.from_pretrained(pre_model))
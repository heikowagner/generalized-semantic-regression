# %%
import torch
from transformers import PreTrainedModel, BertPreTrainedModel, AutoModel, AutoConfig, AutoTokenizer, BertModel
import matplotlib.pyplot as plt
from transformers import AdamW
from tqdm.auto import tqdm
import pandas as pd
from torch.utils.data import DataLoader

from RiskBERT import glmModel, RiskBertModel
from RiskBERT import trainer, evaluate_model, evaluate_model_glm, print_params, visualize_model
from RiskBERT.simulation.data_functions import Data


# %%
# Set device to gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# simulate data
model_dataset = Data(20000, scores=torch.tensor([[0.2], [0.4]]), weigth=5)
plt.plot(model_dataset.y)

# %%
# choose pretrained model
pre_model = "distilbert-base-uncased"

# %%

glm_model = glmModel(2)

glm_model, Total_Loss_glm, Validation_Loss_glm, Test_Loss_glm = trainer(
    model=glm_model,
    model_dataset=model_dataset,
    epochs=5,
    evaluate_fkt=evaluate_model_glm,
    batch_size=500,
    tokenizer=AutoTokenizer.from_pretrained(pre_model),
    optimizer=torch.optim.SGD(glm_model.parameters(), lr=0.001),
    device=device,
)
#%%
model = RiskBertModel(model=pre_model, input_dim=2, dropout=0.4, freeze_bert=True, mode="CLS")
model, Total_Loss, Validation_Loss, Test_Loss = trainer(
    model=model,
    model_dataset=model_dataset,
    epochs=5,
    batch_size=1000,
    evaluate_fkt=evaluate_model,
    tokenizer=AutoTokenizer.from_pretrained(pre_model),
    optimizer=torch.optim.SGD(model.parameters(), lr=0.001),
    device=device,
)

# %%
import gc

torch.cuda.empty_cache()
gc.collect()
# %%

model_unfreeze = RiskBertModel(model=pre_model, input_dim=2, dropout=0.4, freeze_bert=False, mode="CLS")
model_unfreeze, Total_Loss_unfreeze, Validation_Loss_unfreeze, Test_Loss_unfreeze = trainer(
    model=model_unfreeze,
    model_dataset=model_dataset,
    epochs=5,
    batch_size=250,
    evaluate_fkt=evaluate_model,
    tokenizer=AutoTokenizer.from_pretrained(pre_model),
    optimizer=torch.optim.SGD(model_unfreeze.parameters(), lr=0.001),
    device=device,
)

# %%
visualize_model(model_unfreeze, model_dataset, AutoTokenizer.from_pretrained(pre_model), device)

# %%
Total_Loss_glm, Validation_Loss_glm
Total_Loss, Validation_Loss
Total_Loss_unfreeze, Validation_Loss_unfreeze
# %%
Total_Loss_glm[-1:],
Validation_Loss_glm[-1:]
Total_Loss[-1:],
Validation_Loss[-1:]
Total_Loss_unfreeze[-1:],
Validation_Loss_unfreeze[-1:]

# %%
fig, ax = plt.subplots()
plt.plot([l for l in Total_Loss_glm], label="Train Loss glm model")
plt.plot([l for l in Validation_Loss_glm], label="Validation Loss glmmodel")
plt.plot([l for l in Total_Loss], label="Train Loss freezed model")
plt.plot([l for l in Validation_Loss], label="Validation Loss  freezed model")
plt.plot([l for l in Total_Loss_unfreeze], label="Train Loss full model")
plt.plot([l for l in Validation_Loss_unfreeze], label="Validation Loss full model")
ax.legend()
plt.xlabel("Iterations ")
plt.ylabel("total loss ")
# %%
import pandas as pd

classes = ["GLM", "BERT freezed", "BERT full"]
df = pd.DataFrame(
    [
        [Total_Loss_glm[-1:][0], Validation_Loss_glm[-1:][0], Test_Loss_glm],
        [Total_Loss[-1:][0], Validation_Loss[-1:][0], Test_Loss],
        [Total_Loss_unfreeze[-1:][0], Validation_Loss_unfreeze[-1:][0], Test_Loss_unfreeze],
    ],
    classes,
    ["Loss", "Validation Loss", "Test Loss"],
)
df
# %%
# Save the models

# torch.save(model_unfreeze, './model_unfreeze')
# torch.save(model, './model')
# torch.save(glm_model, './glm_model')

# %%

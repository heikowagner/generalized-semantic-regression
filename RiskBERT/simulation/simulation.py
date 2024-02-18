# %%
import torch
from transformers import PreTrainedModel, BertPreTrainedModel, AutoModel, AutoConfig, AutoTokenizer, BertModel
import matplotlib.pyplot as plt
from transformers import AdamW
from tqdm.auto import tqdm
import pandas as pd
from torch.utils.data import DataLoader

from RiskBERT import glmModel, RiskBertModel
from RiskBERT import trainer, evaluate_model, print_params, visualize_model
from RiskBERT.simulation.data_functions import SimulatedData


# %%
# Set device to gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%


from RiskBERT.utils import DataConstructor

pre_model = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(pre_model)
my_data = DataConstructor(
    sentences=[["Dies ist ein Test"], ["Hallo Welt", "RiskBERT ist das beste BERT Modell"]],
    covariates=[[1, 5], [2, 6]],
    labels=[1, 6],
    tokenizer=tokenizer,
)

model = RiskBertModel(model=pre_model, input_dim=2, dropout=0.4, freeze_bert=True, mode="CLS")
model.to(device)
# %%
model(**my_data.prepare_for_model())

# %%
glm_model = glmModel(2).to(device)
evaluate_model(glm_model, my_data.prepare_for_model())


# %%
# %%
# simulate data
model_dataset = SimulatedData(1000, scores=torch.tensor([[0.2], [0.4]]), weigth=5, tokenizer=tokenizer)
plt.plot(model_dataset.labels)

# my_data_2 = DataConstructor(
#    sentences=model_dataset.sentences,
#    covariates=model_dataset.covariates.tolist(),
#    labels=model_dataset.labels.tolist(),
#    tokenizer=tokenizer,
# )

# %%
model_dataset.prepare_for_model()

# %%
model_dataset.__getitem__(1)


# %%
print(model_dataset.__getitem__(1)["input_ids"])
print(model_dataset.prepare_for_model()["input_ids"])

# %%
# choose pretrained model
pre_model = "distilbert-base-uncased"

# %%
i = 0
data_loader = DataLoader(model_dataset, batch_size=200, shuffle=True)


# %%

train_features = next(iter(data_loader))
# %%

print(train_features)
# %%
evaluate_model(model=model.to(device), dataset=train_features)


# %%
for dataset in data_loader:
    print(dataset)
    print(i)
    i = i + 1
    loss = evaluate_model(model=model.to(device), dataset=dataset)

# %%
data_loader

# %%

glm_model = glmModel(2).to(device)
# %%
glm_model, Total_Loss_glm, Validation_Loss_glm, Test_Loss_glm = trainer(
    model=glm_model,
    model_dataset=model_dataset,
    epochs=5,
    batch_size=1000,
    # tokenizer=AutoTokenizer.from_pretrained(pre_model),
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
# Predict from the model

from RiskBERT.utils import DataConstructor

pre_model = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(pre_model)
my_data = DataConstructor(
    sentences=[["Dies ist ein Test"], ["Hallo Welt", "RiskBERT ist das beste BERT Modell"]],
    covariates=[[1, 5], [2, 6]],
    tokenizer=tokenizer,
)


model = RiskBertModel(model=pre_model, input_dim=2, dropout=0.4, freeze_bert=True, mode="CLS")
# %%
model(**my_data.prepare_for_model())

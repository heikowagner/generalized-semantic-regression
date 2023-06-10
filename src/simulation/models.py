# %%
import torch
from transformers import PreTrainedModel, BertPreTrainedModel, AutoModel, AutoConfig, AutoTokenizer

# %%
class RiskBertModel(PreTrainedModel):
    def __init__(self, model, input_dim):
        super().__init__(AutoConfig.from_pretrained(model))
        self.backbone = AutoModel.from_pretrained(model)
        config = AutoConfig.from_pretrained(model)
        #self.output = nn.Linear(config.hidden_size, config.num_labels)
        self.intertmed= torch.nn.Linear(config.hidden_size,1, bias=False)
        self.output = torch.nn.Linear(input_dim+1, 1)

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
        intermed = self.intertmed(sequence_output)
        outputs = self.output(torch.cat( (torch.mean(intermed).reshape(1) , covariates), axis=0 ))

        # if labels, then we are training
        loss = None
        if labels is not None:
            # add custom neg log likelyhood here
            loss_fn = torch.nn.PoissonNLLLoss()
            loss = loss_fn(outputs, labels)
            
        return {
            "loss": loss,
            "lambda": outputs
        }
    
model = RiskBertModel("bert-base-uncased", 2)
 # %%

# %% 
# Lower learning rates are often better for fine-tuning transformers
#model.compile(optimizer=Adam(3e-5))
from data_functions import Data

model_dataset = Data()
# %%

sentence = model_dataset.__getitem__(1)[2]
covariates = model_dataset.__getitem__(1)[0]  #covariates are the first tensor!
label = model_dataset.__getitem__(1)[1]

# %%

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    model_dataset, (int( len(model_dataset)*0.7 ), int( len(model_dataset)*0.2) , int(len(model_dataset)*0.1))
)

# %%
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer(str( sentence ), return_tensors="pt")

# %% Put dataset into RiskBert model
fit = model(**inputs, covariates= covariates, labels=label)


# %%
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=1)


# Train the model
Loss = []
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
epochs = 100
for epoch in range(epochs):
    for x,y, sentence_sample in train_dataset:

        inputs = tokenizer(str( sentence_sample ), return_tensors="pt")
        #print(x)
        #print(y)
        #print(inputs)
        y_pred = model(**inputs, covariates=x, labels=y)
        loss = y_pred['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   
    print(f"epoch = {epoch}, loss = {loss}")
print("Done training!")
 
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
def preprocess_function(input_dataset):
    tokenized = tokenizer(str( input_dataset[2]), truncation=True)
    tokenized + {"covariates": input_dataset[0] } + {"labels": input_dataset[1] }
    return tokenized

# %%
preprocess_function(model_dataset.__getitem__(1))

# %%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, my [MASK] is cute", return_tensors="pt")

# %%
with torch.no_grad():
    lambda = model(**inputs, covariates= covariates)['lambda']
#outputs = model(**inputs)
lambda




# %%
from transformers import TrainingArguments, Trainer
from data_functions import Data

training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
)

trainer.train()


# %%
model_test = AutoModel.from_pretrained("bert-base-uncased")

# %%

from transformers import AutoTokenizer, BertForPreTraining, BertForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForPreTraining.from_pretrained("bert-base-uncased")

model = BertForMaskedLM.from_pretrained("bert-base-uncased")

#model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, my [MASK] is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits
#outputs = model(**inputs)

mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
tokenizer.decode(predicted_token_id)
# %%

#use labels for loss ->

#labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
# mask labels of non-[MASK] tokens
#labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

#outputs = model(**inputs, labels=labels)
#round(outputs.loss.item(), 2)
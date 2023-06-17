# %%
import torch
from transformers import PreTrainedModel, BertPreTrainedModel, AutoModel, AutoConfig, AutoTokenizer
import matplotlib.pyplot as plt
from transformers import AdamW
from tqdm.auto import tqdm

# %%
class glmModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(glmModel, self).__init__()
        self.output = torch.nn.Linear(input_dim, 1)

    def forward(
        self,
        covariates,
        labels=None,
    ):
        
        lambda_i = self.output(covariates)
        # if labels, then we are training
        loss = None
        if labels is not None:
            # add custom neg log likelyhood here
            loss_fn = torch.nn.PoissonNLLLoss()
            loss = loss_fn(lambda_i, labels)
            
        return {
            "loss": loss,
            "lambda": lambda_i
        }


class RiskBertModel(PreTrainedModel):
    def __init__(self, model, input_dim, dropout=0.5, freeze_bert=False):
        super(RiskBertModel, self).__init__(AutoConfig.from_pretrained(model))
        self.backbone = AutoModel.from_pretrained(model)
        self.dropout = torch.nn.Dropout(dropout)
        config = AutoConfig.from_pretrained(model)
        #self.output = nn.Linear(config.hidden_size, config.num_labels)
        self.intertmed= torch.nn.Linear(config.hidden_size,1, bias=False)
        self.output = torch.nn.Linear(input_dim+1, 1)
        if freeze_bert:
            for param in self.backbone.parameters():
                param.requires_grad = False

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
            #return_dict=False
        )
        sequence_output = outputs.last_hidden_state
        cls_representation = sequence_output[:,0,:]

        #dropped_outputs = self.dropout(sequence_output)
        dropped_outputs = self.dropout(cls_representation)
        intermed = self.intertmed(dropped_outputs)
        if len(intermed)==1:
            lambda_i = self.output(torch.cat( (intermed.reshape(1) , covariates), axis=0 ))
        else:
            lambda_i = self.output(torch.cat( (intermed , covariates), 1 ))

        #lambda_i = self.output(torch.cat( (torch.mean(intermed).reshape(1) , covariates), axis=0 ))

        # if labels, then we are training
        loss = None
        if labels is not None:
            # add custom neg log likelyhood here
            loss_fn = torch.nn.PoissonNLLLoss()
            loss = loss_fn(lambda_i, labels)
            
        return {
            "loss": loss,
            "lambda": lambda_i
        }
    
model = RiskBertModel("bert-base-uncased", 2, freeze_bert=False)

 # %%

# %% 
# Lower learning rates are often better for fine-tuning transformers
#model.compile(optimizer=Adam(3e-5))
from data_functions import Data

model_dataset = Data(50000)
# %%

sentence = model_dataset.__getitem__(100)[2]
covariates = model_dataset.__getitem__(100)[0]  #covariates are the first tensor!
label = model_dataset.__getitem__(100)[1]

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
loss = fit['loss']
print(loss)
# %%
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=5000)

# %%

# at beginning of the script
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)
# Train the model
Loss = []
#optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
# activate training mode
model.train()

from transformers import get_linear_schedule_with_warmup
epochs = 1
total_steps = len(train_loader) * epochs

# %%
epochs = 30
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

dataloader = DataLoader(train_dataset,
                        batch_size=100,
                        shuffle=True)

total_steps = len(dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

for epoch in range(epochs):
    Loss=[] # reset loss for each epoch
    Test_Loss=[]
    total_loss = 0
    total_loss_test = 0
    model.train() # set model in train mode
    #model.eval()
    for x,y, sentence_sample in dataloader:
        inputs =  tokenizer( ["[CLS]" + str( sentence ) + "[SEP]" for sentence in sentence_sample], return_tensors="pt",padding=True ).to(device)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(**inputs, covariates=x, labels=y)
        loss = y_pred['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        Loss.append(loss)
        total_loss += loss.item()
    #for x,y, sentence_sample in test_dataset:
    #    with torch.no_grad():        
    #        inputs = tokenizer(str( sentence_sample ), return_tensors="pt")
    #        inputs.to(device)
    #        x, y = x.to(device), y.to(device)
    #        y_pred = model(**inputs, covariates=x, labels=y)
    #        loss = y_pred['loss']
    #        Test_Loss.append(loss)
    #        total_loss_test += loss.item()
    print(f"epoch = {epoch}, loss = {loss}")
    print(f"epoch = {epoch}, total loss = {total_loss/ len(train_dataset)}")
    print(f"epoch = {epoch}, test loss = {total_loss_test/ len(test_dataset)}")
print("Done training!")


 
# %%
# Plot the graph for epochs and loss
plt.plot([l.cpu().detach().numpy() for l in Loss])
plt.xlabel("Iterations ")
plt.ylabel("total loss ")
plt.show()

#[p for p in model.parameters()]
# %%

#2nd and third param are the params of the glm
for param in model.output.parameters():
    print(param)


#%%
## Fit an ordinary glm

glm_model = glmModel(2)


optimizer = AdamW(glm_model.parameters(),
                  lr = 2e-2, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-4 # args.adam_epsilon  - default is 1e-8.
                )

dataloader = DataLoader(train_dataset,
                        batch_size=5000,
                        shuffle=True)

epochs = 100
for epoch in range(epochs):
    Loss=[] # reset loss for each epoch
    Test_Loss=[]
    total_loss = 0
    total_loss_test = 0
    glm_model.to(device)
    glm_model.train() # set model in train mode
    for x,y, sentence_sample in dataloader:
        x, y = x.to(device), y.to(device)
        print(x)
        optimizer.zero_grad()
        y_pred = glm_model(covariates=x, labels=y)
        loss = y_pred['loss']
        loss.backward()
        optimizer.step()
        Loss.append(loss)
        total_loss += loss.item()
    print(f"epoch = {epoch}, loss = {loss}")
    print(f"epoch = {epoch}, total loss = {total_loss/ len(train_dataset)}")
    print(f"epoch = {epoch}, test loss = {total_loss_test/ len(test_dataset)}")
print("Done training!")

# %%
# Plot the graph for epochs and loss
plt.plot([l.cpu().detach().numpy() for l in Loss])
plt.xlabel("Iterations ")
plt.ylabel("total loss ")
plt.show()

# true parameter are 1 and 3, we
for param in glm_model.parameters():
    print(param)
# %%
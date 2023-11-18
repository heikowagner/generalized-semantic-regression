# %%
import torch
from transformers import BertPreTrainedModel, AutoConfig, BertModel
from .loss_functions import poissonLoss

class glmModel(torch.nn.Module):
    def __init__(self, input_dim, cnt_hidden_layer=0, loss_fn=poissonLoss):
        super(glmModel, self).__init__()
        self.output = torch.nn.Linear(input_dim, 1)

        self.hidden_layer = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim) for i in range(cnt_hidden_layer)])
        self.activation_functions = torch.nn.ModuleList([torch.nn.Tanh() for i in range(cnt_hidden_layer)])

        self.loss_fn = loss_fn

    def forward(
        self,
        covariates,
        labels=None,
    ):
        for i, l in enumerate(self.hidden_layer):
            covariates = self.hidden_layer[i](covariates)
            covariates = self.activation_functions[i](covariates)

        log_lambda_i = self.output(covariates)

        # if labels, then we are training
        loss = None
        if labels is not None:
            loss = self.loss_fn(log_lambda_i, labels)

        return {"loss": loss, "lambda": log_lambda_i}


# %%

# https://stackoverflow.com/questions/64156202/add-dense-layer-on-top-of-huggingface-bert-model
class RiskBertModel(BertPreTrainedModel):
    def __init__(self, model, input_dim, dropout=0.5, freeze_bert=False, mode="CLS", hidden_layer=1, loss_fn=poissonLoss):
        super(RiskBertModel, self).__init__(AutoConfig.from_pretrained(model))
        self.backbone = BertModel.from_pretrained(model)
        config = AutoConfig.from_pretrained(model)
        self.relu = self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.output = glmModel(input_dim=config.hidden_size + input_dim, cnt_hidden_layer=hidden_layer)
        self.mode = mode
        self.loss_fn = loss_fn
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
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        if self.mode == "CLS":
            sequence_output = outputs.last_hidden_state
            bert_outputs = sequence_output[:, 0, :].view(-1, 768)
        else:
            bert_outputs = outputs.pooler_output

        dropped_outputs = self.dropout(bert_outputs)
        dropped_outputs = self.relu(dropped_outputs)
        glm_model = self.output(torch.cat((covariates, dropped_outputs), 1), labels, self.loss_fn)

        return glm_model

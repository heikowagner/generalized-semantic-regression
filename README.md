# generalized-semantic-regression
RiskBERT is a significant step forward, making it easier than ever to incorporate text fragments into various applications, such as insurance frequency and severity models, or other GLM-based models. Feel free to explore and utilize RiskBERT for your text analysis needs.

To learn more about the RiskBERT implementation read this article: https://www.thebigdatablog.com/generalized-semantic-regression-using-contextual-embeddings/

Example: 
`pip install RiskBERT`

```
from transformers import AutoTokenizer
import torch
from RiskBERT import glmModel, RiskBertModel
from RiskBERT import trainer, evaluate_model
from RiskBERT.simulation.data_functions import Data

# Set device to gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_dataset = Data(20000, scores=torch.tensor([[0.2],[0.4]]), weigth=5)

pre_model= "distilbert-base-uncased"
model = RiskBertModel(model=pre_model, input_dim=2, dropout=0.4, freeze_bert=True, mode="CLS")
model, Total_Loss, Validation_Loss, Test_Loss = trainer(model =model, 
        model_dataset = model_dataset, 
        epochs=100,
        batch_size=1000,
        evaluate_fkt=evaluate_model,
        tokenizer=  AutoTokenizer.from_pretrained(pre_model), 
        optimizer =  torch.optim.SGD(model.parameters(), lr=0.001),
        device = device
        )
```

# Upload to pip
```
python -m pip install build twine
python -m build
twine check dist/*
twine upload dist/*`
````


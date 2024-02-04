from transformers import AutoTokenizer
from RiskBERT import RiskBertModel
from RiskBERT.utils import DataConstructor


def test_models():
    pre_model = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(pre_model)
    model = RiskBertModel(model=pre_model, input_dim=2, dropout=0.4, freeze_bert=True, mode="CLS")
    # Predict from the model
    my_data = DataConstructor(
        sentences=[["Dies ist ein Test"], ["Hallo Welt", "RiskBERT ist das Beste"]],
        covariates=[[1, 5], [2, 6]],
        # labels =[0,3],
        tokenizer=tokenizer,
    )

    model(**my_data.prepare_for_model())

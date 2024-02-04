# %%
from RiskBERT.utils import DataConstructor
from transformers import AutoTokenizer
from torch import tensor


def test_DataConstructor():
    pre_model = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(pre_model)
    my_data = DataConstructor(
        sentences=[["Dies ist ein Test"], ["Hallo Welt", "RiskBERT ist das Beste"]],
        covariates=[[1, 5], [2, 6]],
        labels=[0, 3],
        tokenizer=tokenizer,
    )

    result = {
        "input_ids": tensor(
            [
                [101, 8289, 21541, 16417, 3231, 102, 0, 0],
                [101, 2534, 2080, 2057, 7096, 102, 0, 0],
                [101, 3891, 8296, 21541, 8695, 2190, 2063, 102],
            ]
        ),
        "attention_mask": tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]]),
        "covariates": tensor([[1.0, 5.0], [2.0, 6.0]]),
        "labels": tensor([0.0, 3.0]),
        "num_sentences": [1, 2],
    }

    assert str(my_data.prepare_for_model()) == str(result)

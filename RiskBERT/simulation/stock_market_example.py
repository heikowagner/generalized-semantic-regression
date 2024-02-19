# %%
import requests
import pandas as pd
import yahooquery as yq
import matplotlib.pyplot as plt
import numpy as np
from RiskBERT import normalLoss
from RiskBERT import RiskBertModel
from RiskBERT import trainer, evaluate_model
from RiskBERT import DataConstructor
import torch
from transformers import AutoTokenizer
import datetime
import numpy as np

refit = True
# %%
## AAPL.US example on Apple

# The eodhd Api only allows 1000 records per call. Therfore we looping the calls until the API denies the request.
i = 0
start_from = datetime.datetime.today().strftime("%Y-%m-%d")

try:
    appl_news = pd.read_parquet("./aapl_news.parquet")
except:
    while True:
        try:
            url = f"https://eodhd.com/api/news?s=AAPL.US&offset=0&limit=1000&to={start_from}&api_token=demo&fmt=json"
            data = requests.get(url).json()
            if i == 0:
                appl_news = pd.DataFrame(data)
            else:
                appl_news = appl_news._append(pd.DataFrame(data), ignore_index=True)
            start_from = str(min(pd.to_datetime(appl_news["date"]).dt.date))
            print(start_from)

            if min(pd.to_datetime(appl_news["date"]).dt.date) <= datetime.date.fromisoformat("2016-02-19"):
                break

            i = i + 1

        except Exception as e:
            print(e)
            break
    # This takes a while, therefore we will write to a paquet to be faster next time
    appl_news.to_parquet("./aapl_news.parquet")


# %%
# We use the yahoo finance api to get open and close prices

end = max(appl_news["date"])
start = min(appl_news["date"])

tq = yq.Ticker("AAPL")
stock_data = tq.history(start=start, end=end)
# %%
# Join Data
appl_news["daydate"] = pd.to_datetime(appl_news["date"]).dt.date

# %%
stock_with_news = stock_data.merge(appl_news, left_on="date", right_on="daydate", how="left")

# %%

# To determine the correct distribution to be used in RiskBERT we plot a basic histogram
plt.hist(np.log(stock_data["close"]) - np.log(stock_data["open"]), bins=50, color="skyblue", edgecolor="black")

# -> The distribution looks pretty "normal" no further tests needed. This is what to be expected theoretically which should not be surprising for the frequent reader of my blog (see https://www.thebigdatablog.com/does-my-stock-trading-strategy-work/)
# %%
# We thus use the normalLoss as Loss function for RiskBERT
# %%
# Prepare the Data adding additional features to be used by RiskBERT

stock_with_news = stock_with_news.dropna()
stock_with_news["label"] = np.log(stock_with_news["close"]) - np.log(stock_with_news["open"])
stock_with_news["daycounter"] = (stock_with_news["daydate"] - min(stock_with_news["daydate"])).apply(lambda x: x.days)
stock_with_news["num_symbols"] = stock_with_news["symbols"].apply(lambda x: len(x))

# %%
# Set device to gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pre_model = "distilbert-base-uncased"
model = RiskBertModel(model=pre_model, input_dim=1, dropout=0.2, freeze_bert=True, mode="CLS", loss_fn=normalLoss)
tokenizer = AutoTokenizer.from_pretrained(pre_model)
# %%
covariates = np.array([stock_with_news["num_symbols"]]).T

my_data = DataConstructor(
    sentences=[[x] for x in stock_with_news["title"]],
    covariates=covariates,
    labels=[[x] for x in stock_with_news["label"]],
    tokenizer=tokenizer,
    device=device,
)

# %%
model.to(device)
# %%

if refit:
    fitted_model, Total_Loss, Validation_Loss, Test_Loss = trainer(
        model=model,
        model_dataset=my_data,
        epochs=100,
        batch_size=2000,
        evaluate_fkt=evaluate_model,
        tokenizer=tokenizer,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.001),
        device=device,
    )
    torch.save(fitted_model, "./fitted_model.mod")
else:
    fitted_model = torch.load("./fitted_model.mod")

# %%
from torch.utils.data import DataLoader

batch_size = 5000
batches = DataLoader(my_data, batch_size=batch_size, shuffle=False)
preds = []
labels = []
for batch in batches:
    inputs = tokenizer.batch_encode_plus(
        [item for row in batch[2] for item in row],
        return_tensors="pt",
        padding=True,
        truncation=True,
        # max_length=50,
        add_special_tokens=True,
    ).to(device)
    labels = labels + batch[1].tolist()
    preds = preds + fitted_model(**inputs, covariates=batch[0], num_sentences=batch[4])["lambda"].tolist()

# %%
preds = np.array(preds)
labels = np.array(labels)
# %%
R_squared = -sum((preds - np.mean(labels)) ** 2) / sum((labels - np.mean(labels)) ** 2)

# %%

from sklearn.metrics import r2_score

r2_score(labels, preds)

# %%

# %%
import requests
import pandas as pd
import yahooquery as yq
import matplotlib.pyplot as plt
import numpy as np
from RiskBERT import normalLoss
from RiskBERT import RiskBertModel, glmModel
from RiskBERT import trainer
from RiskBERT import DataConstructor
import torch
from transformers import AutoTokenizer
import datetime
import numpy as np

import torch.nn as nn

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
# Get Dow Jones data to remove market trends
end = max(appl_news["date"])
start = min(appl_news["date"])

tq = yq.Ticker("^DJI")
dow_data = tq.history(start=start, end=end)
dow_data["log_diff"] = np.log(dow_data["close"]) - np.log(dow_data["open"])

# %%
# Join Data
appl_news["daydate"] = pd.to_datetime(appl_news["date"]).dt.date

# %%
stock_with_news = stock_data.merge(appl_news, left_on="date", right_on="daydate", how="left").merge(
    dow_data, left_on="daydate", right_on="date", suffixes=("", "_dow")
)

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
stock_with_news_grp = stock_with_news.groupby("daycounter").title.apply(list)

single_day = (
    pd.DataFrame(stock_with_news_grp)
    .merge(stock_with_news, left_on="daycounter", right_on="daycounter", how="left")
    .drop_duplicates("daycounter")
)

# %%
# Set device to gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
covariates = np.array(
    [
        stock_with_news["num_symbols"],
        stock_with_news["sentiment"].apply(lambda x: x["neg"]),
        stock_with_news["sentiment"].apply(lambda x: x["neu"]),
        stock_with_news["sentiment"].apply(lambda x: x["pos"]),
        # stock_with_news["log_diff"],
    ]
).T

pre_model = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(pre_model)

my_data = DataConstructor(
    sentences=[[x] for x in stock_with_news["title"]],
    covariates=covariates,
    labels=[[x] for x in stock_with_news["label"]],
    tokenizer=tokenizer,
)

# %%

glm_model = glmModel(input_dim=covariates.shape[1], loss_fn=nn.MSELoss(), cnt_hidden_layer=0)

glm_model.to(device)
glm_model, Total_Loss_glm, Validation_Loss_glm, Test_Loss_glm = trainer(
    model=glm_model,
    model_dataset=my_data,
    epochs=5000,
    batch_size=int(len(covariates) / 8),
    tokenizer=None,
    optimizer=torch.optim.Adam(glm_model.parameters(), lr=0.001),
    device=device,
    num_workers=0,
)

# %%
model = RiskBertModel(
    model=pre_model, input_dim=covariates.shape[1], dropout=0.2, freeze_bert=False, mode="CLS", loss_fn=nn.MSELoss()
)  # normalLoss)

model.to(device)
# %%

if refit:
    fitted_model, Total_Loss, Validation_Loss, Test_Loss = trainer(
        model=model,
        model_dataset=my_data,
        epochs=5000,
        batch_size=150,
        tokenizer=tokenizer,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        device=device,
        update_freq=1,
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
data = pd.DataFrame()
sentence = []
fitted_model.eval()  # set model in eval mode
with torch.no_grad():
    for batch in batches:
        inputs = tokenizer.batch_encode_plus(
            [item for row in batch[2] for item in row],
            return_tensors="pt",
            padding=True,
            truncation=True,
            # max_length=50,
            add_special_tokens=True,
        ).to(device)
        # data = data._append(pd.DataFrame(batch[2][0]).merge(pd.DataFrame(labels), left_index=True, right_index=True).merge(pd.DataFrame(preds), left_index=True, right_index=True))
        labels = labels + batch[1].tolist()
        sentence = sentence + [item for row in batch[2] for item in row]
        preds = preds + fitted_model(**inputs, covariates=batch[0], num_sentences=batch[4])["lambda"].tolist()

# %%
preds = np.array(preds)
labels = np.array(labels)
# %%
mse = np.mean((preds - labels) ** 2)

# %%
# We have to beat at least the standart deviation
sigma = np.mean((labels - np.mean(labels)) ** 2)

# %%
import matplotlib.pyplot as plt

plt.plot([l for l in Validation_Loss], label="Validation Loss")
plt.xlabel("Iterations ")

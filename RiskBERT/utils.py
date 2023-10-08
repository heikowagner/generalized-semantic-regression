# %%
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
import torchvision
from torchview import draw_graph

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, tokenizer, x, y, sentence_sample, device):
    inputs = tokenizer.batch_encode_plus(
        sentence_sample, return_tensors="pt", padding=True, truncation=True, max_length=50, add_special_tokens=True
    ).to(device)
    y_pred = model(**inputs, covariates=x, labels=y)
    loss = y_pred["loss"]
    return loss


def evaluate_model_glm(model, x, y, **kwargs):
    y_pred = model(covariates=x, labels=y)
    loss = y_pred["loss"]
    return loss


def trainer(model, model_dataset, epochs, evaluate_fkt, tokenizer, batch_size=100, optimizer=None, seed=123, device="cuda"):

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        model_dataset,
        (int(len(model_dataset) * 0.7), int(len(model_dataset) * 0.2), int(len(model_dataset) * 0.1)),
        generator=torch.Generator().manual_seed(seed),
    )

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)

    epochs = epochs
    train_batches = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_batches = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    test_batches = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total_steps = len(train_batches) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps  # Default value in run_glue.py
    )
    Loss = []
    Total_Loss = []
    Validation_Loss = []
    model.to(device)
    for epoch in range(epochs):
        # reset total loss for each epoch
        total_loss = 0
        model.train()  # set model in train mode
        for x, y, sentence_sample, embeddingsx in train_batches:
            optimizer.zero_grad()
            loss = evaluate_fkt(
                model=model, tokenizer=tokenizer, x=x.to(device), y=y.to(device), sentence_sample=sentence_sample, device=device
            )
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            Loss.append(loss)
            total_loss += loss.item()

        val_loss = 0
        model.eval()  # set model in eval mode

        with torch.no_grad():
            for x, y, sentence_sample, embeddingsx in val_batches:
                v_loss = evaluate_fkt(
                    model=model,
                    tokenizer=tokenizer,
                    x=x.to(device),
                    y=y.to(device),
                    sentence_sample=sentence_sample,
                    device=device,
                )
                val_loss += v_loss.item()

        print(f"epoch = {epoch}, loss = {loss}")
        print(f"epoch = {epoch}, loss = {v_loss}")
        print(f"epoch = {epoch}, total loss = {total_loss/ len(train_batches)}")
        print(f"epoch = {epoch}, validation loss = {val_loss/ len(val_batches)}")
        Total_Loss.append(total_loss / len(train_batches))
        Validation_Loss.append(val_loss / len(val_batches))
    print("Done training!")

    test_loss = 0
    model.eval()  # set model in eval mode

    with torch.no_grad():
        for x, y, sentence_sample, embeddingsx in test_batches:
            t_loss = evaluate_fkt(
                model=model, tokenizer=tokenizer, x=x.to(device), y=y.to(device), sentence_sample=sentence_sample, device=device
            )
            test_loss += t_loss.item()
    Test_Loss = test_loss / len(test_batches)

    print(f"epoch = {epoch}, Test loss = {test_loss}")
    # Plot the graph for epochs and loss

    plt.plot([l for l in Total_Loss], label="Train Loss")
    plt.plot([l for l in Validation_Loss], label="Validation Loss")
    plt.xlabel("Iterations ")
    plt.ylabel("total loss ")
    plt.show()

    # 2nd and third param are the params of the glm
    for param in model.output.parameters():
        print(param)

    return model, Total_Loss, Validation_Loss, Test_Loss


def print_params(model):
    params = list(model.named_parameters())

    print("The BERT model has {:} different named parameters.\n".format(len(params)))

    print("==== Embedding Layer ====\n")

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print("\n==== First Transformer ====\n")

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print("\n==== Output Layer ====\n")

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


def visualize_model(model, model_dataset, tokenizer, device):
    x = model_dataset.x[:2, :].to(device)
    inputs = inputs = tokenizer.batch_encode_plus(
        ["Hallo", "Welt"], return_tensors="pt", padding=True, truncation=True, max_length=50, add_special_tokens=True
    ).to(device)
    model_grp = draw_graph(
        model, input_data=inputs["input_ids"], expand_nested=True, covariates=x, attention_mask=inputs["attention_mask"]
    )
    model_grp.visual_graph


def visualize_attention(model, tokenizer, sentences=["This is not a test"], view="model"):
    inputs = tokenizer.batch_encode_plus(
        sentences, return_tensors="pt", padding=True, truncation=True, max_length=50, add_special_tokens=True
    ).to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    outputs = model.backbone(**inputs, output_attentions=True)  # Run model
    attention = outputs[-1]  # Retrieve attention from model outputs

    if view == "model":
        return model_view(attention, tokens)  # Display model view
    else:
        return head_view(attention, tokens)

# %%
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torchview import draw_graph
from torch.utils.data import Dataset
import numpy as np

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class collate:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def collate_fn(self, data):
        try:
            x, y, sentence_sample, embed, num_sentences, input_ids, attention_mask = data
        except ValueError:
            x, y, sentence_sample, embed, num_sentences, input_ids, attention_mask = zip(*data)
        # x= np.array(x)
        if self.tokenizer:
            pass
        #            inputs = self.tokenizer.batch_encode_plus(
        #                [item for row in sentence_sample for item in row],
        #                return_tensors="pt",
        #                padding=True,
        #                truncation=True,
        #                # max_length=50,
        #                add_special_tokens=True,
        #            )
        else:
            return {
                "input_ids": None,
                "attention_mask": None,
                "covariates": torch.Tensor(x),
                "labels": torch.Tensor(y),
                "num_sentences": num_sentences,
            }
        #        if y:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "covariates": torch.Tensor(x),
            "labels": torch.Tensor(y),
            "num_sentences": num_sentences,
        }


def trainer(
    model,
    model_dataset,
    epochs,
    tokenizer=None,
    batch_size=100,
    optimizer=None,
    seed=123,
    device="cuda",
    num_workers=4,
    update_freq=100,
    **kwargs,
):
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        model_dataset,
        (
            int(len(model_dataset) * 0.7),
            int(len(model_dataset) * 0.2),
            len(model_dataset) - int(len(model_dataset) * 0.7) - int(len(model_dataset) * 0.2),
        ),
        generator=torch.Generator().manual_seed(seed),
    )

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)

    epochs = epochs
    col = collate(tokenizer=tokenizer)
    train_batches = MultiEpochsDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=col.collate_fn, num_workers=num_workers, pin_memory=True
    )
    val_batches = MultiEpochsDataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=col.collate_fn, num_workers=num_workers, pin_memory=True
    )
    test_batches = MultiEpochsDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=col.collate_fn, num_workers=num_workers, pin_memory=True
    )
    # we could use collate_fn to do the tokeization
    total_steps = len(train_batches) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=5, num_training_steps=total_steps  # Default value in run_glue.py
    )
    Total_Loss = []
    Validation_Loss = []
    for epoch in range(epochs):
        # reset total loss for each epoch
        total_loss = 0
        model.train()  # set model in train mode
        for batch in train_batches:
            if batch["input_ids"] is None:
                batch = {
                    "covariates": batch["covariates"].to(device, non_blocking=True),
                    "num_sentences": batch["num_sentences"],
                    "labels": batch["labels"].to(device, non_blocking=True),
                }
            else:
                batch = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "covariates": batch["covariates"].to(device, non_blocking=True),
                    "num_sentences": batch["num_sentences"],
                    "labels": batch["labels"].to(device, non_blocking=True),
                }
            optimizer.zero_grad()
            loss = model(**batch)["loss"]
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        val_loss = 0
        model.eval()  # set model in eval mode

        with torch.no_grad():
            for batch in val_batches:
                if batch["input_ids"] is None:
                    batch = {
                        "covariates": batch["covariates"].to(device, non_blocking=True),
                        "num_sentences": batch["num_sentences"],
                        "labels": batch["labels"].to(device, non_blocking=True),
                    }
                else:
                    batch = {
                        "input_ids": batch["input_ids"].to(device),
                        "attention_mask": batch["attention_mask"].to(device),
                        "covariates": batch["covariates"].to(device, non_blocking=True),
                        "num_sentences": batch["num_sentences"],
                        "labels": batch["labels"].to(device, non_blocking=True),
                    }
                v_loss = model(**batch)["loss"]
                val_loss += v_loss.item()

        if (epoch + 1) % update_freq == 0:
            print(f"epoch = {epoch+1}/{epochs}, train loss = {loss}")
            print(f"epoch = {epoch+1}/{epochs}, loss = {v_loss}")
            print(f"epoch = {epoch+1}/{epochs}, total loss = {total_loss/ len(train_batches)}")
            print(f"epoch = {epoch+1}/{epochs}, validation loss = {val_loss/ len(val_batches)}")
        Total_Loss.append(total_loss / len(train_batches))
        Validation_Loss.append(val_loss / len(val_batches))
    print("Done training!")

    test_loss = 0
    model.eval()  # set model in eval mode

    with torch.no_grad():
        for batch in test_batches:
            if batch["input_ids"] is None:
                batch = {
                    "covariates": batch["covariates"].to(device, non_blocking=True),
                    "num_sentences": batch["num_sentences"],
                    "labels": batch["labels"].to(device, non_blocking=True),
                }
            else:
                batch = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "covariates": batch["covariates"].to(device, non_blocking=True),
                    "num_sentences": batch["num_sentences"],
                    "labels": batch["labels"].to(device, non_blocking=True),
                }
            t_loss = model(**batch)["loss"]
            test_loss += t_loss.item()
    Test_Loss = test_loss / len(test_batches)

    print(f"Test loss = {test_loss}")
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
    )
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    outputs = model.backbone(**inputs, output_attentions=True)  # Run model
    attention = outputs[-1]  # Retrieve attention from model outputs

    if view == "model":
        return model_view(attention, tokens)  # Display model view
    else:
        return head_view(attention, tokens)


class _RepeatSampler(object):
    """Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class DataConstructor(Dataset):
    def __init__(self, sentences, covariates, labels=None, tokenizer=None, prepare_cache=True):
        self.covariates = np.array(covariates)  # torch.Tensor(covariates).to(device)
        if labels:
            self.labels = np.array(labels)  # torch.Tensor(labels).to(device)
            self.has_label = True
        else:
            self.labels = None
            self.has_label = False
        self.tokenizer = tokenizer
        self.len = len(covariates)
        self.embeddings = np.array([0] * len(covariates))
        self.num_sentences = np.array([len(sentence) for sentence in sentences])

        # Pad sentences
        max_sentences = max(self.num_sentences)
        self.sentences = [np.pad(s, (0, max_sentences - len(s)), "constant", constant_values="") for s in sentences]
        self.sentences = np.array(self.sentences)

        self.prepare_cache = prepare_cache
        if prepare_cache:
            cache = self.tokenizer.batch_encode_plus(
                [item for row in self.sentences for item in row],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=50,
                add_special_tokens=True,
            )
            self.cached_inputs_ids = cache["input_ids"]
            self.cached_attention_mask = cache["attention_mask"]

        else:
            self.cached_inputs_ids = [0 for i in self.sentences]
            self.cached_attention_mask = [0 for i in self.sentences]

        # Todo: Add checks! len(covariates)=len(sentences)=len(labels) etc.

    def prepare_for_model(self, index=None, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        inputs = self.tokenizer.batch_encode_plus(
            [item for row in self.sentences for item in row],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=50,
            add_special_tokens=True,
        ).to(device)
        if not self.has_label:
            return {**inputs, "covariates": torch.Tensor(self.covariates).to(device), "num_sentences": self.num_sentences}
        else:
            return {
                **inputs,
                "covariates": self.covariates.to(device),
                "labels": self.labels.to(device),
                "num_sentences": self.num_sentences,
            }

    # Getter
    def __getitem__(self, index):
        return (
            self.covariates[index],
            self.labels[index],
            self.sentences[index],
            self.embeddings[index],
            self.num_sentences[index],
            self.cached_inputs_ids[index],
            self.cached_attention_mask[index],
        )

    def __getitems__(self, index_list):
        return self.__getitem__(index_list)

    # getting data length
    def __len__(self):
        return self.len

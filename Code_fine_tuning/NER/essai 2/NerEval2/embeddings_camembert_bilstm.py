import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import CamembertTokenizerFast, CamembertModel
from torchcrf import CRF


LR = 1e-3
BATCH = 4
EPOCHS = 3


# Load WikiAnn (French)
dataset = load_dataset("wikiann", "fr")

# Label mapping
label_list = dataset["train"].features["ner_tags"].feature.names
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")


# Dataset tokenized word_ids

def encode_batch(batch):
    tokenized = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        return_attention_mask=True
    )

    # store word_ids for reconstruction
    tokenized["word_ids"] = [
        tokenized.word_ids(i) for i in range(len(batch["tokens"]))
    ]

    # convert labels to tensor
    tokenized["labels"] = batch["ner_tags"]
    return tokenized


dataset = dataset.map(encode_batch, batched=True)
dataset.set_format(type="torch",
                   columns=["input_ids", "attention_mask", "labels"])



# Collate function

def collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    attention_mask = []
    word_ids = []
    labels = []

    for item in batch:
        pad = max_len - len(item["input_ids"])

        input_ids.append(torch.cat([item["input_ids"], torch.ones(pad, dtype=torch.long)]))
        attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad, dtype=torch.long)]))

        # word_ids stored separately
        word_ids.append(item["word_ids"] + [None] * pad)

        labels.append(torch.tensor(item["labels"]))

    return {
        "input_ids": torch.stack(input_ids).to(DEVICE),
        "attention_mask": torch.stack(attention_mask).to(DEVICE),
        "word_ids": word_ids,
        "labels": labels
    }


train_loader = DataLoader(dataset["train"], batch_size=BATCH, shuffle=True, collate_fn=collate_fn)


# CamemBERT Model (gelé) et LSTM-CRF

class CamembertLSTMCRF(nn.Module):
    def __init__(self, num_labels):
        super().__init__()

        self.camembert = CamembertModel.from_pretrained(
            "camembert-base",
            output_hidden_states=True
        )

        # freeze transformer
        for p in self.camembert.parameters():
            p.requires_grad = False

        hidden_size = self.camembert.config.hidden_size

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(256, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, word_ids, labels=None):
        out = self.camembert(input_ids, attention_mask=attention_mask)

        # average last 4 layers
        last4 = torch.stack(out.hidden_states[-4:], dim=0)
        emb = last4.mean(0)  # (B, T, H)

        # convert subwords → words
        batch_word_embs = []
        batch_word_labels = []

        for i in range(len(word_ids)):
            mapping = {}
            for tok_idx, w_id in enumerate(word_ids[i]):
                if w_id is not None:
                    mapping.setdefault(w_id, []).append(tok_idx)

            # average subwords
            word_embs = []
            for w_id in sorted(mapping.keys()):
                idxs = mapping[w_id]
                word_embs.append(emb[i, idxs].mean(0))

            batch_word_embs.append(torch.stack(word_embs))

            if labels is not None:
                batch_word_labels.append(labels[i][:len(word_embs)])

        # pad word sequences
        max_words = max(x.size(0) for x in batch_word_embs)
        padded_embs = []
        padded_labels = []

        for i, w_emb in enumerate(batch_word_embs):
            pad = max_words - w_emb.size(0)
            padded_embs.append(torch.cat([w_emb, torch.zeros(pad, w_emb.size(1)).to(DEVICE)], dim=0))

            if labels is not None:
                padded_labels.append(torch.cat([
                    batch_word_labels[i],
                    torch.full((pad,), -1, dtype=torch.long).to(DEVICE)
                ]))

        padded_embs = torch.stack(padded_embs)
        if labels is not None:
            padded_labels = torch.stack(padded_labels)

        # LSTM → linear → CRF
        lstm_out, _ = self.lstm(padded_embs)
        emissions = self.fc(lstm_out)

        if labels is not None:
            mask = padded_labels != -1
            loss = -self.crf(emissions, padded_labels, mask=mask)
            return loss
        else:
            mask = torch.ones(emissions.size()[:2], dtype=torch.bool).to(DEVICE)
            return self.crf.decode(emissions, mask=mask)


# Training
model = CamembertLSTMCRF(num_labels=len(label_list)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

model.train()
for epoch in range(EPOCHS):
    for batch in train_loader:
        loss = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["word_ids"],
            batch["labels"]
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Loss:", loss.item())

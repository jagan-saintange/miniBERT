import torch
import torch.nn as nn
from transformers import CamembertModel, CamembertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchcrf import CRF  # Ensure you have `torchcrf` installed

# Step 1: Load Dataset and Tokenizer
dataset = load_dataset("wikiann", "fr")
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

# Step 2: Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Step 3: Prepare the Labels
label_list = dataset['train'].features['ner_tags'].feature
label_to_id = {label: i for i, label in enumerate(label_list.feature)}
def align_labels_with_tokens(examples):
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = examples.word_ids(i)
        label_ids = [label_to_id[label] if word_id is not None else -100 for word_id in word_ids]
        labels.append(label_ids)
    examples['labels'] = labels
    return examples

# Align the labels with tokens
tokenized_datasets = tokenized_datasets.map(align_labels_with_tokens, batched=True)

# Step 4: Create the Feature Extraction Model
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.camembert = CamembertModel.from_pretrained("camembert-base")

    def forward(self, input_ids, attention_mask):
        outputs = self.camembert(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # Get last hidden states

# Step 5: Create LSTM+CRF Model
class LSTMCRF(nn.Module):
    def __init__(self, num_labels, hidden_dim=256, embedding_dim=768):
        super(LSTMCRF, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        features = self.feature_extractor(input_ids, attention_mask)
        lstm_out, _ = self.lstm(features)
        logits = self.hidden2tag(lstm_out)

        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask.bool(), reduction='mean')
            return loss
        else:
            return self.crf.decode(logits, mask=attention_mask.bool())

# Step 6: Initialize model, optimizer, etc.
model = LSTMCRF(num_labels=len(label_list)).to("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Step 7: Prepare DataLoader
train_loader = DataLoader(tokenized_datasets["train"], batch_size=16, shuffle=True)

# Step 8: Training function
def train_model(model, data_loader):
    model.train()
    for batch in data_loader:
        input_ids = batch['input_ids'].to("cuda" if torch.cuda.is_available() else "cpu")
        attention_mask = batch['attention_mask'].to("cuda" if torch.cuda.is_available() else "cpu")
        labels = batch['labels'].to("cuda" if torch.cuda.is_available() else "cpu")

        optimizer.zero_grad()
        loss = model(input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()

# Step 9: Train the model (example for a few epochs)
for epoch in range(3):  # Adjust the number of epochs as needed
    train_model(model, train_loader)

# Step 10: Save your trained model
torch.save(model.state_dict(), "lstm_crf_ner_model.pth")

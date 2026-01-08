
import torch
from transformers import CamembertTokenizer, CamembertForTokenClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# load French WikiANN

# Load the French part of WikiANN dataset
dataset = load_dataset("wikiann", "fr")

# Load CamemBERT tokenizer
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

# Get label information
label_list = dataset["train"].features["ner_tags"].feature.names
num_labels = len(label_list)
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for i, label in enumerate(label_list)}

# Define a function to tokenize and align labels
MAX_LEN = 128 # Max length i reduce for fast training

def tokenize_and_align_labels(examples):
    all_input_ids = []
    all_labels = []
    all_attention_masks = []

    for tokens, ner_tags in zip(examples["tokens"], examples["ner_tags"]):
        input_ids = [tokenizer.cls_token_id]
        label_ids = [-100]

        for word, label in zip(tokens, ner_tags):
            word_tokens = tokenizer.tokenize(word)
            word_token_ids = tokenizer.convert_tokens_to_ids(word_tokens)

            input_ids.extend(word_token_ids)
            label_ids.extend([label] * len(word_token_ids))

        input_ids.append(tokenizer.sep_token_id)
        label_ids.append(-100)

        # ---- TRUNCATION ----
        if len(input_ids) > MAX_LEN:
            input_ids = input_ids[:MAX_LEN]
            label_ids = label_ids[:MAX_LEN]

        # ---- PADDING ----
        attention_mask = [1] * len(input_ids)
        padding_length = MAX_LEN - len(input_ids)

        input_ids += [tokenizer.pad_token_id] * padding_length
        label_ids += [-100] * padding_length
        attention_mask += [0] * padding_length

        all_input_ids.append(input_ids)
        all_labels.append(label_ids)
        all_attention_masks.append(attention_mask)

    return {
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": all_attention_masks,
    }

# Tokenize and align the dataset
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Load CamemBERT for token classification
model = CamembertForTokenClassification.from_pretrained("camembert-base", num_labels=num_labels)

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4, # Smaller than 16 so it doesn't take hours
    per_device_eval_batch_size=4,
    num_train_epochs=30,
    weight_decay=0.0,  # No specific regularization
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the final model
trainer.save_model("./camembert-ner")

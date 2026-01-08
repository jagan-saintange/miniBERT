import random
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset, load_dataset

# Config
model_base = "camembert-base"
num_labels = 8  # adjust to actual number of labels in WikiANN (use dataset features)
seed = 42
random.seed(seed)

# Load WikiANN French subset
raw = load_dataset("wikiann", "fr")
# Use train split and create a small validation split if needed
dataset_train = raw["train"]
dataset_valid = raw["validation"] if "validation" in raw else dataset_train.train_test_split(test_size=0.2, seed=seed)["test"]
dataset_test = raw["test"] if "test" in raw else dataset_valid

# Helper: extract tokens and ner tags for compatibility with your pipeline
def convert_example(example):
    # wikiann provides "tokens" and "ner_tags" already
    return {"tokens": example["tokens"], "ner_tags": example["ner_tags"]}

train_examples = dataset_train.map(convert_example)
valid_examples = dataset_valid.map(convert_example)
test_examples = dataset_test.map(convert_example)

# If label list needed:
label_list = dataset_train.features["ner_tags"].feature.names if hasattr(dataset_train.features["ner_tags"].feature, "names") else None
if label_list:
    num_labels = len(label_list)

# Tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(model_base, num_labels=num_labels)

# Tokenize and align labels (keeps your function behavior)
def tokenize_and_align_labels(batch, label_all_tokens=True):
    tokenized_inputs = tokenizer(batch["tokens"], truncation=True, padding=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(batch["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

train_dataset = train_examples.map(tokenize_and_align_labels, batched=True, remove_columns=train_examples.column_names)
valid_dataset = valid_examples.map(tokenize_and_align_labels, batched=True, remove_columns=valid_examples.column_names)
test_dataset = test_examples.map(tokenize_and_align_labels, batched=True, remove_columns=test_examples.column_names)

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0,
    logging_dir='./logs',
    seed=seed,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
test_dataset.save_to_disk("./test_dataset")

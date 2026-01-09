from datasets import load_dataset
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))

from transformers import (
    CamembertTokenizer,
    CamembertForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

tokenizer = CamembertTokenizer(vocab_file="models/spm.model")
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"

dataset = load_dataset(
    "text",
    data_files={"train": "Oscar/shards/*.txt"}
)

dataset = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True,max_length=512),
    batched=True,
    remove_columns=["text"]
)

model = CamembertForMaskedLM.from_pretrained("model_init")
model.resize_token_embeddings(len(tokenizer))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="camembert_repro",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    warmup_steps=10_000,
    weight_decay=0.01,
    logging_steps=500,
    save_strategy="steps",
    save_steps=25_000,
    save_total_limit=3,
    fp16=True,
    dataloader_num_workers=4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()

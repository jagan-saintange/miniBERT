from datasets import load_dataset
from transformers import (
    CamembertTokenizer,
    CamembertForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

tokenizer = CamembertTokenizer(
    vocab_file="models/spm.model"
)

dataset = load_dataset(
    "text",
    data_files={"train": "Oscar/shards/*.txt"}
)

dataset = dataset.map(
    lambda x: tokenizer(x["text"]),
    batched=True,
    remove_columns=["text"]
)

model = CamembertForMaskedLM.from_pretrained("model_init")

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
    save_steps=5_000,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()

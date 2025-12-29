from datasets import load_dataset
from transformers import CamembertTokenizer
from transformers import CamembertForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import evaluate
import numpy as np
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))


dataset = load_dataset("xnli", "fr")


tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

def tokenize(batch):
    return tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(
    ["premise", "hypothesis"]
)
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch")



model = CamembertForSequenceClassification.from_pretrained(
    "camembert-base",
    num_labels=3
)



training_args = TrainingArguments(
    output_dir="./camembert-xnli",
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=2500,          # ≈ 30 evals sur 3 epochs
    save_steps=2500,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    seed=42,
    fp16=True
)



accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# Ne garder que 100000 lignes par split

train_small = dataset["train"]#.select(range(100000))


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_small,
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model("./camembert-xnli-best")
tokenizer.save_pretrained("./camembert-xnli-best")

trainer.evaluate(dataset["test"])

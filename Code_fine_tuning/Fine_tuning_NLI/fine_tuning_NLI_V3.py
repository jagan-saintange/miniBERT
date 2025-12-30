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
        batch["premise"], #Les phrases prémises 
        batch["hypothesis"], #Les phrases hypothèses qu'on juge 
        truncation=True,
        padding="max_length",
        max_length=256
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(["premise", "hypothesis"]) #On retire les colonnes catégories car génantes pour l'entrainement
dataset = dataset.rename_column("label", "labels") #On renome label en labels car hugginface utilise l'appelation labels
dataset.set_format("torch")



model = CamembertForSequenceClassification.from_pretrained(
    "camembert-base",
    num_labels=3 #ici, 3 car pour NLI, on veut une classification en 3 classes (Confirmation,négation,neutre)
)



training_args = TrainingArguments(
    output_dir="./camembert-xnli",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=2000,
    save_steps=2000,

    learning_rate=1e-5,
    warmup_steps=7432,
    lr_scheduler_type="polynomial",

    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,

    num_train_epochs=10,  # borne haute, pas objectif réel
    max_steps=123873,

    weight_decay=0.1,
    fp16=True,

    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,

    logging_steps=200,
    seed=42,
) #Choix réalisé en cherchant les hyperparametres de RoBERTa,Source : https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/config/finetuning/mnli.yaml




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

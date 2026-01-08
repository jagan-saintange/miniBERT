from datasets import load_dataset
import torch

from transformers import (
    CamembertTokenizerFast,
    CamembertForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))



tokenizer = CamembertTokenizerFast(
    vocab_file="models/spm.model",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>"
)

dataset = load_dataset(
    "text",
    data_files={"train": "Oscar/shards/*.txt"}
)

# 1. Tokenisation initiale sans troncature
def tokenize_function(examples):
    return tokenizer(examples["text"], add_special_tokens=True)

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    num_proc=8 # Accélère le traitement CPU
)

# 2. Groupage par blocs de 512 (Packing)
def group_texts(examples):
    block_size = 512
    # Concatène tous les textes
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # On arrondit au multiple de 512 inférieur
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
        
    # Découpage en blocs de 512
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # Pour le MLM, on a besoin des labels (qui sont une copie des input_ids au départ)
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=8
)
model = CamembertForMaskedLM.from_pretrained("model_init")
model.resize_token_embeddings(len(tokenizer))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="camembert_MLM_repro_bonus",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=2500,
    weight_decay=0.01,
    logging_steps=100,
    save_strategy="steps",
    save_steps=5000,
    save_total_limit=3,
    fp16=False,
    bf16=True,
    dataloader_num_workers=4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    data_collator=data_collator,
    tokenizer=tokenizer
)



trainer.train(resume_from_checkpoint=True)
trainer.save_model("camembert_MLM_repro_final_bonus")
tokenizer.save_pretrained("camembert_MLM_repro_final_bonus")

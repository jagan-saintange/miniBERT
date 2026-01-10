"""
Projet MLA 2025-2026 : Reproduction de l'architecture CamemBERT 
Objectif : Pré-entraînement Masked Language Modeling (MLM) sur corpus OSCAR.
"""

from datasets import load_dataset
import torch

# --- CONFIGURATION MATÉRIELLE ---
# Optimisation des calculs matriciels sur architectures Ampere/Ada Lovelace 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Validation des ressources GPU 
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

# --- INITIALISATION DU TOKENIZER ---
# Chargement de notre modèle SentencePiece spécifique à CamemBERT 
tokenizer = CamembertTokenizer(vocab_file="models/spm.model")
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"

# --- PRÉPARATION DES DONNÉES ---
# Chargement du corpus partitionné pour une gestion efficace de la mémoire (Shards)
dataset = load_dataset(
    "text",
    data_files={"train": "Oscar/shards/*.txt"}
)

# Tokenisation avec troncature à la longueur maximale supportée par l'architecture (512 tokens)
dataset = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, max_length=512),
    batched=True,
    remove_columns=["text"]
)

# --- INITIALISATION DU MODÈLE ---
# Chargement des poids initiaux et redimensionnement des embeddings pour correspondre au tokenizer
model = CamembertForMaskedLM.from_pretrained("model_init")
model.resize_token_embeddings(len(tokenizer))

# Configuration du Data Collator pour la tâche de complétion (MLM) avec probabilité de 15%
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# --- HYPERPARAMÈTRES D'ENTRAÎNEMENT ---
training_args = TrainingArguments(
    output_dir="camembert_MLM_repro",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    # Gradient Accumulation pour simuler un batch de 64 (8x8) malgré les limites de mémoire VRAM 
    gradient_accumulation_steps=8, 
    learning_rate=1e-4,
    warmup_steps=10_000,
    weight_decay=0.01,
    logging_steps=500,
    save_strategy="steps",
    save_steps=25_000,
    save_total_limit=3,
    # Utilisation de la précision mixte (FP16) pour accélérer l'entraînement sur GPU 
    fp16=True, 
    dataloader_num_workers=4
)

# --- EXÉCUTION DE L'EXPÉRIMENTATION ---
# L'objet Trainer centralise la logique de boucle d'entraînement et de logging 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Lancement du processus d'apprentissage 
trainer.train()
trainer.save_model("./camembert_MLM_repro_best")
tokenizer.save_pretrained("./camembert_MLM_repro_best")

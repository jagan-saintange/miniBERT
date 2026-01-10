"""
Projet de Machine Learning Avancé (MLA) - Sorbonne Université 
Objectif : Reproduction des résultats expérimentaux d'un modèle type CamemBERT via Masked Language Modeling (MLM).
Ce script implémente une stratégie de 'Packing' pour optimiser le débit (throughput) sur GPU.
"""

from datasets import load_dataset
import torch
from transformers import (
    CamembertTokenizerFast,
    CamembertForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# --- OPTIMISATION DU MATÉRIEL (Hardware Acceleration) ---
# Optimisation des calculs matriciels sur architectures Ampere/Ada Lovelace
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"CUDA disponible : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Utilisation du GPU : {torch.cuda.get_device_name(0)}")

# --- PRÉPARATION DU TOKENIZER ET DES DONNÉES ---
# Initialisation du tokenizer SentencePiece rapide avec gestion des tokens spéciaux.
tokenizer = CamembertTokenizerFast(
    vocab_file="models/spm.model",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>"
)

# Chargement du corpus au format texte brut (ex: extraits de la base OSCAR).
dataset = load_dataset(
    "text",
    data_files={"train": "Oscar/shards/*.txt"}
)

# Étape 1 : Tokenisation initiale sans troncature pour conserver l'intégralité du texte avant le packing.
def tokenize_function(examples):
    return tokenizer(examples["text"], add_special_tokens=True)

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    num_proc=4 # Parallélisation du traitement CPU pour accélérer la préparation.
)

# Étape 2 : Stratégie de 'Packing' (regroupement par blocs).
# Permet de maximiser l'efficacité computationnelle en évitant les séquences trop courtes et le padding excessif.
def group_texts(examples):
    block_size = 512
    # Concaténation de tous les tokens du batch en un flux continu.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # Ajustement pour obtenir des blocs uniformes de taille block_size.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
        
    # Découpage du flux concaténé en séquences de 512 tokens.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # Pour le MLM, les labels cibles sont initialement une copie exacte des input_ids.
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=8
)

# --- INITIALISATION DE L'ARCHITECTURE ---
# Chargement du modèle CamemBERT à partir d'un checkpoint initial.
model = CamembertForMaskedLM.from_pretrained("model_init")
# Ajustement de la couche d'embeddings en fonction de la taille réelle du dictionnaire du tokenizer.
model.resize_token_embeddings(len(tokenizer))

# Configuration du collator pour la tâche de complétion (MLM) avec 15% de tokens masqués.
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# --- CONFIGURATION DES HYPERPARAMÈTRES D'ENTRAÎNEMENT ---
# Ces paramètres sont définis pour assurer une reproductibilité optimale des résultats de l'article de référence.
training_args = TrainingArguments(
    output_dir="camembert_MLM_repro_bonus",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=32, # Taille de batch par GPU.
    gradient_accumulation_steps=2,  # Simule un batch effectif plus large pour stabiliser le gradient.
    learning_rate=1e-4,             # Taux d'apprentissage pour la phase de pré-entraînement.
    warmup_steps=2500,              # Montée en puissance progressive pour éviter l'instabilité initiale.
    weight_decay=0.01,              # Régularisation pour prévenir le surapprentissage.
    logging_steps=100,
    save_strategy="steps",
    save_steps=5000,
    save_total_limit=3,             # Conservation des 3 meilleurs checkpoints uniquement.
    fp16=False,
    bf16=True,                      # Utilisation du Bfloat16 pour une meilleure stabilité numérique sur GPU récents.
    dataloader_num_workers=4
)

# --- GESTION DE LA BOUCLE D'ENTRAÎNEMENT ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Lancement de l'entraînement avec reprise possible à partir du dernier checkpoint (fault tolerance).
trainer.train(resume_from_checkpoint=True)

# Sauvegarde finale des artefacts pour la phase de démonstration.
trainer.save_model("camembert_MLM_repro_final_bonus")
tokenizer.save_pretrained("camembert_MLM_repro_final_bonus")

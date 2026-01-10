"""
Projet MLA 2025-2026 - Sorbonne Université
Phase d'Évaluation : Fine-tuning de CamemBERT sur la tâche XNLI (Cross-lingual NLI).
Objectif : Classifier la relation logique entre une prémisse et une hypothèse en 3 classes.
"""

from datasets import load_dataset
from transformers import (
    CamembertTokenizer, 
    CamembertForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
import evaluate
import numpy as np
import torch

# --- DIAGNOSTIC MATÉRIEL ---
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

# --- PRÉPARATION DES DONNÉES (XNLI FR) ---
# Chargement du dataset de Natural Language Inference (NLI) en français
dataset = load_dataset("xnli", "fr")

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

def tokenize(batch):
    """
    Tokenisation par paire de phrases.
    Le tokenizer concatène prémisse et hypothèse avec un séparateur spécifique.
    """
    return tokenizer(
        batch["premise"], 
        batch["hypothesis"], 
        truncation=True,
        padding="max_length",
        max_length=256 # Longueur optimisée pour les paires de phrases NLI
    )

# Mapping et nettoyage du dataset pour le format PyTorch
dataset = dataset.map(tokenize, batched=True)
# Suppression des colonnes de texte brut pour ne conserver que les tensors d'entrée
dataset = dataset.remove_columns(["premise", "hypothesis"])
# Harmonisation du nom de la colonne cible avec les attentes du Trainer
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch")

# --- INITIALISATION DU MODÈLE DE CLASSIFICATION ---
# Chargement des poids pré-entraînés avec une tête de classification (Linear layer)
model = CamembertForSequenceClassification.from_pretrained(
    "camembert-base",
    num_labels=3 # Classes : Entailment (0), Neutral (1), Contradiction (2)
)

# --- HYPERPARAMÈTRES DE FINE-TUNING ---
# Configuration basée sur les recommandations de RoBERTa/CamemBERT pour MNLI/XNLI
training_args = TrainingArguments(
    output_dir="./camembert-xnli",
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=2000,
    save_steps=2000,

    learning_rate=1e-5,
    warmup_steps=7432,           # Montée en puissance lente pour préserver les poids pré-entraînés
    lr_scheduler_type="polynomial",

    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,

    num_train_epochs=10, 
    max_steps=123873,            # Limite de pas de calcul pour la convergence

    weight_decay=0.1,            # Régularisation L2 forte pour éviter le surapprentissage
    fp16=True,                   # Accélération via précision mixte

    load_best_model_at_end=True, # Early stopping : conserver le meilleur modèle sur la validation
    metric_for_best_model="accuracy",
    greater_is_better=True,

    logging_steps=200,
    seed=42,
)

# --- MÉTRIQUES D'ÉVALUATION ---
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """Calcul de la précision (accuracy) lors des phases de validation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# --- ENTRAÎNEMENT ET ÉVALUATION FINALE ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Lancement de la phase d'apprentissage supervisé
trainer.train()

# Persistance du modèle fine-tuné et du tokenizer
trainer.save_model("./camembert-xnli-best")
tokenizer.save_pretrained("./camembert-xnli-best")

# Évaluation finale sur le jeu de test (généralisation)
print("Évaluation sur le dataset de test :")
trainer.evaluate(dataset["test"])

# fine-tuning Camembert pour NER named enntity recognition sur un corpus en français
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer # on utilise notamment la fonction trainer
import torch

# 1. Chargement du modèle CamemBERT-base et du tokenizer [4, 5]
model_base = "camembert-base"
# On définit num_labels à 8 (7 types d'entités du FTB + 1 pour 'O') [6]
tokenizer = AutoTokenizer.from_pretrained(model_base)
model = AutoModelForTokenClassification.from_pretrained(model_base, num_labels=8)

# 2. Configuration de l'entraînement [1]
# Les auteurs utilisent Adam et sélectionnent le meilleur modèle sur 30 époques.
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Sélection sur l'ensemble de validation [1]
    learning_rate=5e-5,           # À ajuster via grid search selon les sources [1]
    per_device_train_batch_size=16,
    num_train_epochs=30,          # Maximum 30 époques [1]
    weight_decay=0,               # Pas de régularisation spécifique mentionnée pour le NER [1]
    logging_dir='./logs',
)

# 3. Initialisation du Trainer
# Note : Vous devez fournir vos propres objets 'train_dataset' et 'eval_dataset'
# formatés pour la classification de tokens (ex: French Treebank) [6]
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset, 
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# 4. Lancement du fine-tuning
trainer.train()
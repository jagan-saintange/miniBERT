# Generated from: camembert Parisstories.ipynb
# Converted at: 2026-01-09T23:23:33.809Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from datasets import load_dataset
from transformers import (
    CamembertTokenizerFast, 
    CamembertForTokenClassification, 
    DataCollatorForTokenClassification,
    TrainingArguments, 
    Trainer
)

import numpy as np
import evaluate

# 1. Chargement du dataset  (Universal Dependencies French)
# Autorise explicitement l'utilisation du script de chargement
from datasets import Dataset, DatasetDict
from conllu import parse_incr

def load_conllu_to_hf(filepath):
    # On ouvre le fichier .conllu local
    with open(filepath, "r", encoding="utf-8") as data_file:
        data_list = []
        for tokenlist in parse_incr(data_file):
            tokens = [token["form"] for token in tokenlist]
            upos = [token["upos"] for token in tokenlist]
            data_list.append({"tokens": tokens, "upos": upos})
    return Dataset.from_list(data_list)

# Charger parisstories (via le projet Universal Dependencies)
dataset = DatasetDict({
    "train": load_conllu_to_hf("fr_parisstories-ud-train.conllu"),
    "validation": load_conllu_to_hf("fr_parisstories-ud-dev.conllu"),
    "test": load_conllu_to_hf("fr_parisstories-ud-test.conllu")
})

# Liste standard des 17 étiquettes Universal POS (UPOS)
label_list = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", 
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"
]

# Création des dictionnaires de correspondance
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

# Fonction pour transformer les textes ("NOUN", "VERB") en chiffres (7, 15)
def encode_labels(example):
    # On remplace chaque étiquette par son ID. 
    # Si une étiquette est inconnue ou vide (_), on met -100 (sera ignoré par le modèle)
    example["labels"] = [label2id.get(l, -100) for l in example["upos"]]
    return example

# Appliquer la conversion sur tout le dataset
dataset = dataset.map(encode_labels)

#  Chargement du Tokenizer (version Fast obligatoire pour l'alignement)
model_name = "camembert-base"
tokenizer = CamembertTokenizerFast.from_pretrained(model_name)

#  Fonction de prétraitement (Alignement des tokens et labels)
def tokenize_and_align_labels(examples):
    # On tokenise les textes
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    
    labels = []
    for i, label in enumerate(examples["upos"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # Tokens spéciaux ([CLS], [SEP]) -> -100
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # C'est le début d'un nouveau mot
                # ON CONVERTIT ICI : on cherche l'ID du label (ex: 'DET' -> 5)
                label_name = label[word_idx]
                label_id = label2id.get(label_name, -100) # -100 si inconnu
                label_ids.append(label_id)
            else:
                # C'est un subword (ex: 'ait' dans 'mangeait') -> -100
                label_ids.append(-100)
            previous_word_idx = word_idx
            
        labels.append(label_ids)

    # On crée la colonne 'labels' qui contient uniquement des ENTIERS
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# On relance le map (pensez à redéfinir label2id avant si nécessaire)
tokenized_gsd = dataset.map(tokenize_and_align_labels, batched=True)

#  Chargement du modèle pour la classification de tokens
model = CamembertForTokenClassification.from_pretrained(
    model_name, 
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# Métriques avec 'evaluate'
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

print("Modèle et données partut prêts pour l'entraînement !")

from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer

#  Le collator s'occupe de mettre les phrases à la même longueur (padding)
data_collator = DataCollatorForTokenClassification(tokenizer)

#  Configuration de l'entraînement
training_args = TrainingArguments(
    output_dir="./results_camembert_pos_partut",
    eval_strategy="epoch",            # Changé ici : 'eval_strategy' au lieu de 'evaluation_strategy'
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir='./logs',
    load_best_model_at_end=True,      # Pour garder le meilleur modèle (pas le dernier)
    metric_for_best_model="accuracy",
    # --- AJOUTEZ CES DEUX LIGNES ---
    logging_steps=20,       # Affiche la loss tous les 10 steps au lieu de 500
    logging_first_step=True # Affiche la loss dès le tout premier step
    # -------------------------------# Basé sur l'accuracy
)


#  Initialisation du Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_gsd["train"],
    eval_dataset=tokenized_gsd["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics # La fonction qu'on a définie plus haut
) 
import shutil
# Supprime tous les dossiers 'checkpoint-XXX' dans le répertoire de sortie
[shutil.rmtree(os.path.join(training_args.output_dir, d)) for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint")]

print(" Nettoyage terminé : tous les checkpoints ont été supprimés.")
#  C'EST PARTI !
trainer.train()
# On utilise le split 'test' qui a été tokenisé
results = trainer.evaluate(tokenized_gsd["test"])

print("Résultats sur le jeu de test :", results)

trainer.save_model("./mon_modele_final_TOP6")

import shutil
# Supprime tous les dossiers 'checkpoint-XXX' dans le répertoire de sortie
[shutil.rmtree(os.path.join(training_args.output_dir, d)) for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint")]

print("✨ Nettoyage terminé : tous les checkpoints ont été supprimés.")

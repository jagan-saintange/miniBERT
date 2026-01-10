"""
Projet de Machine Learning Avancé (MLA) - Sorbonne Université
Phase 1 : Initialisation de l'architecture du modèle
Ce script prépare le squelette du modèle (poids aléatoires) à partir d'une configuration spécifique.
"""

from transformers import CamembertConfig, CamembertForMaskedLM
import os

# --- GESTION DE L'ENVIRONNEMENT ---
# Création du répertoire de destination pour l'artefact initial si inexistant
os.makedirs("model_init", exist_ok=True)

# --- CONFIGURATION DE L'ARCHITECTURE ---
# Chargement de la configuration (hyperparamètres du modèle : couches, têtes, dimensions)
config = CamembertConfig.from_pretrained("config")

# --- INSTANCIATION DU MODÈLE ---
# Initialisation d'un modèle CamemBERT pour le Masked Language Modeling (MLM).
# Note : Les poids sont ici initialisés aléatoirement (cold start) conformément à la config.
model = CamembertForMaskedLM(config)

# --- PERSISTANCE DES DONNÉES ---
# Sauvegarde locale de l'architecture et des poids initiaux.
# Ce checkpoint servira de point de départ reproductible pour l'entraînement.
model.save_pretrained("model_init")

print("Architecture CamemBERT initialisée avec succès dans le dossier 'model_init'.")

"""
Projet MLA 2025-2026 - Sorbonne Université
Script de validation du Tokenizer (Unit Test)
Objectif : Vérifier la cohérence de la segmentation SentencePiece et la persistance des artefacts.
"""

from transformers import CamembertTokenizer
import os

# --- GESTION DES CHEMINS ---
# Définition du répertoire de travail vers les artefacts de modélisation
# Note : S'assurer que le chemin correspond à l'arborescence locale du serveur
os.chdir(r"/home/silver/models/")

# --- CHARGEMENT ET SAUVEGARDE ---
# Initialisation du tokenizer à partir du modèle SentencePiece (.model) généré précédemment
tokenizer = CamembertTokenizer(
    vocab_file="spm.model"
)

# Exportation au format Hugging Face (génère tokenizer_config.json, vocab.txt, etc.)
tokenizer.save_pretrained("tokenizer")

# --- TESTS DE SEGMENTATION (Sanity Check) ---
# Échantillons de test pour observer le comportement de l'algorithme Unigram sur le français
texts = [
    "Le groupe 11 est très efficace.",
    "CamemBERT small va adorer ces tokens.",
    "Miam Miam Scrountch scrountch."
]

print("Vérification de la tokenisation :\n" + "="*30)

for t in texts:
    print(f"Texte original : {t}")
    # Décomposition en sous-unités (sub-tokens)
    tokens = tokenizer.tokenize(t)
    print(f"Tokens générés : {tokens}")
    print(f"Nombre de tokens : {len(tokens)}\n")

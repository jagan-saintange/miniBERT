"""
Projet MLA 2025-2026 - Sorbonne Université
Phase de Prétraitement : Apprentissage du Tokenizer SentencePiece
Objectif : Génération d'un vocabulaire robuste pour la langue française via l'algorithme Unigram.
"""

import sentencepiece as spm
from pathlib import Path

# --- CONFIGURATION DES CHEMINS ET SORTIES ---
# Référence au corpus massif de 4Go 
CORPUS = "/home/silver/Oscar/corpus_4go.txt"  
OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

model_prefix = str(OUT_DIR / "spm")
# Taille du vocabulaire alignée sur les standards CamemBERT (32k tokens)
vocab_size = 32000
# Utilisation du modèle 'unigram' pour une meilleure flexibilité de segmentation sémantique
model_type = "unigram"   
character_coverage = 1.0

# --- ENTRAÎNEMENT DU MODÈLE DE TOKENISATION ---
spm.SentencePieceTrainer.Train(
    input=CORPUS,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    model_type=model_type,
    character_coverage=character_coverage,
    # Optimisation pour les ressources du Master : gestion de corpus volumineux
    train_extremely_large_corpus=True,
    # Échantillonnage de 5 millions de phrases pour équilibrer temps de calcul et représentativité
    input_sentence_size=5_000_000,
    shuffle_input_sentence=True,
    # Désactivation de la limite stricte pour permettre une meilleure convergence du vocabulaire
    hard_vocab_limit=False
)

# --- VALIDATION DES ARTEFACTS ---
# Génération des fichiers .model et .vocab indispensables pour l'initialisation de CamembertTokenizerFast
print("Entraînement terminé :", OUT_DIR / "spm.model", OUT_DIR / "spm.vocab")

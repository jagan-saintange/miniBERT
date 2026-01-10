"""
Projet MLA 2025-2026 - Sorbonne Université
Script de définition de l'hyper-paramétrage (Configuration)
Objectif : Assurer la cohérence structurelle entre le tokenizer SentencePiece et l'architecture CamemBERT.
"""

from transformers import CamembertConfig, CamembertTokenizerFast
import os

# 1. Synchronisation avec les primitives du Tokenizer
tokenizer = CamembertTokenizerFast(vocab_file="models/spm.model")

os.makedirs("config", exist_ok=True)

# 2. Instanciation de la configuration du modèle (Hyper-paramètres)
# Les valeurs choisies ici déterminent la complexité du réseau de neurones à reproduire.
config = CamembertConfig(
    # Utilisation de la taille dynamique (incluant souvent les jetons réservés de SentencePiece)
    vocab_size=len(tokenizer), 
    
    # Alignement des identifiants de jetons spéciaux pour garantir l'intégrité de la logique MLM
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    
    # Architecture Base : 768 dimensions cachées et 12 couches (Similaire à BERT-Base)
    max_position_embeddings=514,     # 512 + 2 jetons de structure
    hidden_size=768,                 # Dimension du vecteur de représentation
    num_hidden_layers=12,            # Profondeur du transformeur
    num_attention_heads=12,          # Nombre de têtes d'attention (Multi-head attention)
    intermediate_size=3072,          # Dimension de la couche Feed-Forward
    hidden_act="gelu",               # Fonction d'activation non-linéaire standard
    hidden_dropout_prob=0.1,         # Régularisation par dropout
    attention_probs_dropout_prob=0.1,
    layer_norm_eps=1e-5,             # Stabilité numérique de la Layer Normalization
    initializer_range=0.02,          # Écart-type pour l'initialisation des poids
    
    # Contrainte structurelle CamemBERT/RoBERTa
    type_vocab_size=1 
)

# 3. Sérialisation de la configuration
config.save_pretrained("config")

# Vérification post-configuration : Indispensable pour la reproductibilité expérimentale
print(f"Vocab size détecté: {config.vocab_size}")
print(f"BOS ID: {config.bos_token_id}, EOS ID: {config.eos_token_id}, PAD ID: {config.pad_token_id}")

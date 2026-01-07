from transformers import CamembertConfig, CamembertTokenizerFast
import os

# 1. Charger le tokenizer pour récupérer les vraies valeurs du spm.model
tokenizer = CamembertTokenizerFast(vocab_file="models/spm.model")

os.makedirs("config", exist_ok=True)

config = CamembertConfig(
    # Utiliser la taille réelle (souvent 32005 et non 32000 avec SentencePiece)
    vocab_size=len(tokenizer), 
    
    # Synchronisation des IDs spéciaux (évite les avertissements et le cache invalide)
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    
    max_position_embeddings=514,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    layer_norm_eps=1e-5,
    initializer_range=0.02,
    
    # Spécificité CamemBERT/RoBERTa : pas de type_vocab_size (on met 1)
    type_vocab_size=1 
)

config.save_pretrained("config")

# Vérification
print(f"Vocab size détecté: {config.vocab_size}")
print(f"BOS ID: {config.bos_token_id}, EOS ID: {config.eos_token_id}, PAD ID: {config.pad_token_id}")
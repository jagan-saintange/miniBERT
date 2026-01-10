"""
Projet MLA 2025-2026 - Sorbonne Université
Script de Sharding : Segmentation du corpus massif pour l'entraînement distribué.
Cette étape est cruciale pour permettre un chargement paresseux (lazy loading) des données.
"""

import os

# Configuration des chemins d'accès au corpus brut et au répertoire de destination
input_file = "Oscar/Raw/corpus_4go.txt"
output_dir = "Oscar/shards"
os.makedirs(output_dir, exist_ok=True)

# Définition de la taille des segments (shards) pour optimiser le parallélisme du DataLoader 
lines_per_shard = 100_000
shard_id = 0
buffer = []

# Lecture par flux (streaming) pour éviter l'épuisement de la mémoire RAM avec des fichiers de plusieurs Go 
with open(input_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        buffer.append(line)
        
        # Seuil de déclenchement de l'écriture physique sur disque
        if len(buffer) == lines_per_shard:
            # Formatage du nom de fichier avec padding (001, 002...) pour un tri naturel
            with open(f"{output_dir}/shard_{shard_id:03d}.txt", "w", encoding="utf-8") as out:
                out.writelines(buffer)
            
            # Réinitialisation du tampon pour la prochaine partition
            buffer = []
            shard_id += 1

# Gestion du reliquat de lignes (dernier shard pouvant être plus petit)
if buffer:
    with open(f"{output_dir}/shard_{shard_id:03d}.txt", "w", encoding="utf-8") as out:
        out.writelines(buffer)

print(f"Prétraitement terminé : {shard_id + 1} segments générés dans {output_dir}.")

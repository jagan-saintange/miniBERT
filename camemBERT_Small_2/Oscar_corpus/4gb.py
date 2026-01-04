import random
import os
from pathlib import Path

os.chdir(r"/home/silver/Oscar")
input_file = "corpus_utf8.txt"
output_file = "corpus_4gb.txt"

target_size_gb = 4.1
current_size_gb = os.path.getsize(input_file) / (1024 ** 3)
keep_ratio = target_size_gb / current_size_gb

random.seed(42)  # reproductibilité

kept_bytes = 0
target_bytes = target_size_gb * (1024 ** 3)

with open(input_file, "rb") as fin, open(output_file, "wb") as fout:
    for line in fin:
        if random.random() < keep_ratio:
            fout.write(line)
            kept_bytes += len(line)

        if kept_bytes >= target_bytes:
            break

print("Corpus sous-échantillonné créé :", Path(output_file).resolve())
print(f"Taille finale : {kept_bytes / (1024 ** 3):.2f} Go")

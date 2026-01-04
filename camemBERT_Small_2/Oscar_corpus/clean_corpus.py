from pathlib import Path
import unicodedata
import os
import chardet

os.chdir(r"/home/silver/Oscar")


input_file = "corpus_sp.txt"
output_file = "corpus_utf8.txt"

#boucle pour essayer de convertir chaque ligne en encodage utf-8
with open(input_file, "rb") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for raw_line in f_in:
        # Détection
        det = chardet.detect(raw_line)
        enc = det["encoding"]

        # Tentative de décodage
        if enc:
            try:
                line = raw_line.decode(enc)
            except:
                # fallback si l'encodage détecté échoue
                line = raw_line.decode("latin-1", errors="replace")
        else:
            # fallback direct car on ne connait pas
            line = raw_line.decode("latin-1", errors="replace")

        f_out.write(line)

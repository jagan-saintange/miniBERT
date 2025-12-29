# test_spm.py
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="models/spm.model")

texts = [
    "Ceci est une phrase d'exemple.",
    "L'Université de Versailles est située en Île de France."
]

for t in texts:
    pieces = sp.encode_as_pieces(t)
    ids = sp.encode_as_ids(t)
    print("TEXT:", t)
    print("PIECES:", pieces)
    print("IDS:", ids)
    print("DECODE:", sp.decode_pieces(pieces))
    print("---")

# vérifier préfixe début de mot
print("Exemples de tokens débutant un mot (préfixe ▁ présent) :")
for p in pieces:
    if p.startswith("▁"):
        print(p)

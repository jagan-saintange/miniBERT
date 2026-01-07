import sentencepiece as spm
from pathlib import Path

##ENTRAINEMENT
CORPUS = "/home/silver/Oscar/corpus_4go.txt"  # ou un fichier concaténé corpus_all.txt
OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

model_prefix = str(OUT_DIR / "spm")
vocab_size = 32000
model_type = "unigram"   # ou "bpe"
character_coverage = 1.0

spm.SentencePieceTrainer.Train(
    input=CORPUS,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    model_type=model_type,
    character_coverage=character_coverage,
    train_extremely_large_corpus = True,
    input_sentence_size=5_000_000,
    shuffle_input_sentence=True,
    hard_vocab_limit=False
)

print("Trained:", OUT_DIR / "spm.model", OUT_DIR / "spm.vocab")


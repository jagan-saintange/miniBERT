import sentencepiece as spm
from pathlib import Path

##ENTRAINEMENT
CORPUS = "data/raw_shards/train_shard_*.txt"  # ou un fichier concaténé corpus_all.txt
OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

model_prefix = str(OUT_DIR / "spm")
vocab_size = 32000
model_type = "unigram"   # ou "bpe"
character_coverage = 1.0
input_sentence_size = 2000000  # échantillon si corpus très grand

spm.SentencePieceTrainer.Train(
    input=CORPUS,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    model_type=model_type,
    character_coverage=character_coverage,
    input_sentence_size=input_sentence_size,
    shuffle_input_sentence=True,
    hard_vocab_limit=False
)

print("Trained:", OUT_DIR / "spm.model", OUT_DIR / "spm.vocab")


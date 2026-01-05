from transformers import CamembertTokenizer
import os

os.chdir(r"/home/silver/models/")


tokenizer = CamembertTokenizer(
    vocab_file="spm.model"
)

tokenizer.save_pretrained("tokenizer")

texts = [
    "Le groupe 11 est très efficace.",
    "CamemBERT small va adorer ces tokens.",
    "Miam Miam Scrountch scrountch."
]

for t in texts:
    print(t)
    print(tokenizer.tokenize(t))
    print()

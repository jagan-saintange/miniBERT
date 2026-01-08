import re, json, random
from pathlib import Path

INPUT = "ptb.txt"
OUT_DIR = Path("out_data")
OUT_DIR.mkdir(exist_ok=True)
SEED = 42
TRAIN_FRAC = 0.7

leaf_re = re.compile(r'\(\s*[^()\s]+\s+([^()]+?)\s*\)')

def extract_tokens(line):
    parts = line.strip().split(None, 1)
    tree = parts[1] if len(parts) > 1 else parts[0]
    tokens = [t.strip() for t in leaf_re.findall(tree) if t.strip()]
    return tokens

def read_lines(path):
    # try utf-8, fallback to latin-1
    try:
        return Path(path).read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return Path(path).read_text(encoding="latin-1").splitlines()

lines = [l for l in read_lines(INPUT) if l.strip()]
examples = [{"tokens": extract_tokens(l)} for l in lines if extract_tokens(l)]

random.seed(SEED)
random.shuffle(examples)
n = int(len(examples) * TRAIN_FRAC)
train, val = examples[:n], examples[n:]

def save_jsonl(lst, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in lst:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

save_jsonl(train, OUT_DIR / "train.jsonl")
save_jsonl(val, OUT_DIR / "val.jsonl")
print(f"Saved {len(train)} train and {len(val)} val examples to {OUT_DIR}")



# --- Exemple : tokenisation + alignement (si vous avez labels au niveau mot) ---
# Ici on montre la fonction d'alignement; sans labels, on produit juste input_ids/word_ids.
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("camembert-base", use_fast=True)

def tokenize_example_tokens(example):
    return tokenizer(example["tokens"], is_split_into_words=True, truncation=True)

# Tokeniser un batch d'exemples (sans labels)
batch = train[:2]
tok_out = tokenizer([ex["tokens"] for ex in batch], is_split_into_words=True, padding=False, truncation=True)
print("Sample tokenized:", list(tok_out.keys()))
# Si vous avez labels (liste de labels par mot), utilisez la fonction d'alignement vue précédemment
def align_labels(tokenized_inputs, word_level_labels, label_to_id):
    aligned = []
    for i, labels in enumerate(word_level_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        prev = None
        lab_ids = []
        for wid in word_ids:
            if wid is None:
                lab_ids.append(-100)
            elif wid != prev:
                lab_ids.append(label_to_id[labels[wid]])
            else:
                lab_ids.append(-100)
            prev = wid
        aligned.append(lab_ids)
    return aligned

# Fin

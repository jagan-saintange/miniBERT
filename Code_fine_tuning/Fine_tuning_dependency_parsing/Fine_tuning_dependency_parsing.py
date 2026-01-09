import math
import argparse
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import CamembertTokenizerFast, CamembertModel, get_linear_schedule_with_warmup
import numpy as np
from collections import Counter

# -------------------------
# Utilities: pooling & mapping
# -------------------------
def get_word_mappings(batch_encoding):
    """
    Given a BatchEncoding returned by a tokenizer called with is_split_into_words=True,
    return a list (len=batch) of lists mapping token_index -> word_index, and
    max_word_count across batch.
    We use encoding.word_ids(i) to get per-example word id list.
    """
    batch_word_ids = []
    max_words = 0
    for i in range(len(batch_encoding['input_ids'])):
        wids = batch_encoding.word_ids(batch_index=i)
        # wids is a list of length = seq_len with None for special tokens
        token_to_word = wids  # positions -> word_id or None
        # compute number of words
        word_count = 0
        if token_to_word:
            word_count = max([w for w in token_to_word if w is not None]) + 1 if any(w is not None for w in token_to_word) else 0
        batch_word_ids.append(token_to_word)
        max_words = max(max_words, word_count)
    return batch_word_ids, max_words

def pool_subword_representations(encoded_outputs: torch.Tensor,
                                 token_to_word: List[List[int]],
                                 pooling: str = "first") -> torch.Tensor:
    """
    Args:
      encoded_outputs: (batch, seq_len, hidden)
      token_to_word: list(len=batch) of lists length seq_len mapping token pos -> word index or None
    Returns:
      word_reprs: (batch, max_words, hidden) with padding zeros for missing words
    """
    device = encoded_outputs.device
    batch_size, seq_len, hidden = encoded_outputs.size()
    batch_word_ids, max_words = token_to_word, 0
    # compute max words
    for wids in batch_word_ids:
        cnt = 0
        if wids:
            cnt = max([w for w in wids if w is not None]) + 1 if any(w is not None for w in wids) else 0
        max_words = max(max_words, cnt)

    word_reprs = encoded_outputs.new_zeros(batch_size, max_words, hidden)
    word_masks = encoded_outputs.new_zeros(batch_size, max_words, dtype=torch.bool)

    for b in range(batch_size):
        wids = batch_word_ids[b]
        if not wids:
            continue
        # collect indices per word
        words_positions = {}
        for pos, wid in enumerate(wids):
            if wid is None:
                continue
            words_positions.setdefault(wid, []).append(pos)
        for wid, poses in words_positions.items():
            sub_vecs = encoded_outputs[b, poses, :]  # (n_subwords, hidden)
            if pooling == "first":
                word_reprs[b, wid, :] = sub_vecs[0]
            elif pooling == "mean":
                word_reprs[b, wid, :] = sub_vecs.mean(dim=0)
            elif pooling == "max":
                word_reprs[b, wid, :] = sub_vecs.max(dim=0).values
            else:
                raise ValueError("Unknown pooling")
            word_masks[b, wid] = 1
    return word_reprs, word_masks  # masks: True where a word exists

# -------------------------
# Data Collation: UD -> tokenized batch
# -------------------------
class UDDatasetWrapper:
    """
    Wrap a 'universal_dependencies' dataset split to provide items:
      dict with: tokens(list[str]), head(list[int]), deprel(list[str]), upos(list[str])
    """
    def __init__(self, dataset_split):
        self.dataset = dataset_split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        return {
            "tokens": ex["tokens"],
            "heads": ex["head"],       # integer head indices (0-based or 0=root? UD gives ints with 0 meaning root)
            "deprel": ex["deprel"],
            "upos": ex.get("upos", None)
        }

def ud_collate_fn(batch, tokenizer: CamembertTokenizerFast, label_to_id: Dict[str,int], device="cpu"):
    """
    Corrected version that handles TRUNCATION safely.
    """
    sentences = [ex["tokens"] for ex in batch]
    encoding = tokenizer(sentences, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
    batch_token_to_word, max_words = get_word_mappings(encoding)

    batch_size = len(batch)
    heads_padded = torch.full((batch_size, max_words + 1), -100, dtype=torch.long)
    labels_padded = torch.full((batch_size, max_words + 1), -100, dtype=torch.long)
    word_masks = torch.zeros((batch_size, max_words + 1), dtype=torch.bool)

    for i, ex in enumerate(batch):
        gold_heads = ex["heads"]
        gold_labels = ex["deprel"]
        
        wids = [w for w in batch_token_to_word[i] if w is not None]
        if wids:
            n_encoded_words = max(wids) + 1
        else:
            n_encoded_words = 0
            
        effective_len = min(len(gold_heads), n_encoded_words)
        
        gold_heads = gold_heads[:effective_len]
        gold_labels = gold_labels[:effective_len]
        
        # **BETTER FIX**: Mark truncated dependencies as invalid
        clamped_heads = []
        clamped_labels = []
        truncated_count = 0  # DEBUG COUNTER
        
        for j, (h, l) in enumerate(zip(gold_heads, gold_labels)):
            if h > effective_len:
                # This dependency points to a truncated word - mark as invalid
                clamped_heads.append(-100)  
                clamped_labels.append(-100)
                truncated_count += 1  # DEBUG
            else:
                clamped_heads.append(h)
                clamped_labels.append(l)
        
        # DEBUG OUTPUT
        if truncated_count > 0:
            print(f"⚠️ Batch {i}: Truncated {truncated_count} dependencies (effective_len={effective_len}, original={len(ex['heads'])})")

        heads_padded[i, 1:effective_len+1] = torch.tensor(clamped_heads, dtype=torch.long)
        labels_padded[i, 1:effective_len+1] = torch.tensor([label_to_id.get(l, -100) if l != -100 else -100 for l in clamped_labels], dtype=torch.long)
        word_masks[i, 1:effective_len+1] = 1

    encoding = {k: v.to(device) for k, v in encoding.items()}
    heads_padded = heads_padded.to(device)
    labels_padded = labels_padded.to(device)
    word_masks = word_masks.to(device)

    return encoding, batch_token_to_word, heads_padded, labels_padded, word_masks

# -------------------------
# Model: Feature-based parser (CamemBERT frozen + BiLSTM + biaffine head)
# -------------------------
class FeatureBasedParser(nn.Module):
    def __init__(self, camembert_name="camembert-base", lstm_hidden=400, lstm_layers=3, mlp_dim=500, rel_labels=50, dropout=0.33):
        super().__init__()
        self.camembert = CamembertModel.from_pretrained(camembert_name)
        # Freeze CamemBERT
        for p in self.camembert.parameters():
            p.requires_grad = False

        hidden = self.camembert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        
        # ADD: Learned ROOT embedding
        self.root_embedding = nn.Parameter(torch.randn(hidden))

        # BiLSTM on top of subword outputs (word-level pooling will be done before LSTM)
        self.lstm = nn.LSTM(hidden, lstm_hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True)

        lstm_out_dim = lstm_hidden * 2
        self.arc_head = nn.Linear(lstm_out_dim, mlp_dim)
        self.arc_dep  = nn.Linear(lstm_out_dim, mlp_dim)
        self.rel_head = nn.Linear(lstm_out_dim, mlp_dim)
        self.rel_dep  = nn.Linear(lstm_out_dim, mlp_dim)

        self.label_classifier = nn.Linear(mlp_dim * 2, rel_labels)

    def forward(self, encoding, token_to_word: List[List[int]], pooling="first"):
        # get subword contextual outputs from frozen transformer
        with torch.no_grad():
            outputs = self.camembert(**encoding).last_hidden_state  # (batch, seq_len, hidden)
        # pool to word-level
        word_reprs, word_masks = pool_subword_representations(outputs, token_to_word, pooling=pooling)
        
        # ADD: Prepend ROOT embedding to each sentence
        batch_size = word_reprs.size(0)
        root_embed = self.root_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        word_reprs = torch.cat([root_embed, word_reprs], dim=1)  # (b, n+1, hidden)
        
        # Update mask to include ROOT (always valid)
        root_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=word_reprs.device)
        word_masks = torch.cat([root_mask, word_masks], dim=1)  # (b, n+1)
        
        # feed through BiLSTM
        lstm_out, _ = self.lstm(word_reprs)  # (batch, max_words+1, 2*lstm_hidden)
        lstm_out = self.dropout(lstm_out)

        arc_h = torch.relu(self.arc_head(lstm_out))   # (b, n+1, mlp)
        arc_d = torch.relu(self.arc_dep(lstm_out))    # (b, n+1, mlp)
        rel_h = torch.relu(self.rel_head(lstm_out))
        rel_d = torch.relu(self.rel_dep(lstm_out))

        arc_logits = torch.einsum('bih,bjh->bij', arc_d, arc_h)  # (b, dep_positions, head_positions)

        return arc_logits, rel_d, rel_h, word_masks

# -------------------------
# Model: Fine-tuned parser (CamemBERT trainable + small biaffine head)
# -------------------------
class FineTunedParser(nn.Module):
    def __init__(self, camembert_name="camembert-base", mlp_dim=500, rel_labels=50, dropout=0.33):
        super().__init__()
        self.camembert = CamembertModel.from_pretrained(camembert_name)
        hidden = self.camembert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        
        # ADD: Learned ROOT embedding
        self.root_embedding = nn.Parameter(torch.randn(hidden))

        self.arc_head = nn.Linear(hidden, mlp_dim)
        self.arc_dep  = nn.Linear(hidden, mlp_dim)
        self.rel_head = nn.Linear(hidden, mlp_dim)
        self.rel_dep  = nn.Linear(hidden, mlp_dim)

        self.label_classifier = nn.Linear(mlp_dim * 2, rel_labels)

    def forward(self, encoding, token_to_word: List[List[int]], pooling="first"):
        outputs = self.camembert(**encoding).last_hidden_state  # (batch, seq_len, hidden)
        word_reprs, word_masks = pool_subword_representations(outputs, token_to_word, pooling=pooling)
        word_reprs = self.dropout(word_reprs)
        
        # ADD: Prepend ROOT embedding to each sentence
        batch_size = word_reprs.size(0)
        root_embed = self.root_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        word_reprs = torch.cat([root_embed, word_reprs], dim=1)  # (b, n+1, hidden)
        
        # Update mask to include ROOT (always valid)
        root_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=word_reprs.device)
        word_masks = torch.cat([root_mask, word_masks], dim=1)  # (b, n+1)

        arc_h = torch.relu(self.arc_head(word_reprs))  # (b, n+1, mlp)
        arc_d = torch.relu(self.arc_dep(word_reprs))
        rel_h = torch.relu(self.rel_head(word_reprs))
        rel_d = torch.relu(self.rel_dep(word_reprs))

        arc_logits = torch.einsum('bih,bjh->bij', arc_d, arc_h)  # (b, dep, head)
        return arc_logits, rel_d, rel_h, word_masks

# -------------------------
# Loss & training helpers
# -------------------------
def compute_loss(arc_logits, rel_d, rel_h, gold_heads, gold_labels, word_mask, label_classifier):
    """
    arc_logits: (b, n, n)  where arc_logits[b, dep_idx, head_idx]
    rel_d, rel_h: (b, n, mlp)
    gold_heads: (b, n) integers (head indices). BEWARE: UD uses 0 as root index (we keep as-is)
    gold_labels: (b, n) integers
    word_mask: (b, n) bool
    """
    b, n, _ = arc_logits.shape
    device = arc_logits.device
    # Arc loss: for each dependent (token), predict gold head index
    # Flatten to (b*n, n)
    # For positions that are padding, set loss ignore index
    arc_logits_flat = arc_logits.view(b * n, n)
    gold_heads_flat = gold_heads.view(b * n)

    # Create mask to ignore padded positions in loss
    valid_mask = word_mask.view(b * n)  # bool
    # For invalid positions, replace target with -100 so CrossEntropyLoss ignores them
    target = gold_heads_flat.clone()
    target[~valid_mask] = -100

    arc_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    arc_loss = arc_loss_fct(arc_logits_flat, target)

    # Label loss: compute label logits only for gold head (training)
    # Gather rel_h at gold head positions for every dependent
    # rel_h: (b, n, mlp) ; we want rel_h_selected: (b, n, mlp) where each dependent gets its gold head vector
    gold_heads_expanded = gold_heads.unsqueeze(-1).expand(-1, -1, rel_h.size(-1))  # (b, n, mlp)
    # clamp gold_heads to valid range to avoid gather errors for padded positions; we'll mask later
# clamp gold_heads to valid range to avoid gather errors
    gold_heads_clamped = gold_heads_expanded.clone()
    
    # FIX: Replace -100 (and any other negative numbers) with 0 so gather doesn't crash
    gold_heads_clamped[gold_heads_clamped < 0] = 0
    
    # Also mask out padding positions (standard safety)
    gold_heads_clamped[~word_mask.unsqueeze(-1).expand_as(gold_heads_clamped)] = 0
    
    rel_h_selected = torch.gather(rel_h, dim=1, index=gold_heads_clamped)

    # Combine rel_d and rel_h_selected and classify
    rel_pair = torch.cat([rel_d, rel_h_selected], dim=-1)  # (b, n, 2*mlp)
    label_logits = label_classifier(rel_pair)  # (b, n, num_labels)
    label_logits_flat = label_logits.view(b * n, -1)
    gold_labels_flat = gold_labels.view(b * n)
    gold_labels_flat[~valid_mask] = -100
    
    # print(f"DEBUG CHECK:")
    # print(f"  - Model Output Size (Classes): {arc_logits_flat.shape[-1]}")
    # print(f"  - Max Target Value found: {target.max().item()}")
    # if target.max().item() >= arc_logits_flat.shape[-1]:
    #     print("  - 🚨 CRASH IMMINENT: Target index is larger than model size!")

    label_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    label_loss = label_loss_fct(label_logits_flat, gold_labels_flat)

    total_loss = arc_loss + label_loss
    return total_loss, arc_loss.item(), label_loss.item()

def compute_metrics(arc_logits, rel_d, rel_h, gold_heads, gold_labels, word_mask, label_classifier):
    """
    Simple greedy decode: predict head = argmax over head dim; label choose label based on predicted head.
    Returns UAS, LAS as ratios over valid tokens.
    """
    b, n, _ = arc_logits.shape
    device = arc_logits.device
    pred_heads = arc_logits.argmax(dim=-1)  # (b, n)
    # get label logits for predicted heads
    pred_heads_expanded = pred_heads.unsqueeze(-1).expand(-1, -1, rel_h.size(-1))
    rel_h_selected = torch.gather(rel_h, dim=1, index=pred_heads_expanded)
    rel_pair = torch.cat([rel_d, rel_h_selected], dim=-1)
    label_logits = label_classifier(rel_pair)  # (b, n, num_labels)
    pred_labels = label_logits.argmax(dim=-1)  # (b, n)

    valid = word_mask
    total = valid.sum().item()

    correct_head = ((pred_heads == gold_heads) & valid).sum().item()
    correct_label = ((pred_heads == gold_heads) & (pred_labels == gold_labels) & valid).sum().item()

    uas = correct_head / total if total > 0 else 0.0
    las = correct_label / total if total > 0 else 0.0
    return uas, las

# -------------------------
# Training loop
# -------------------------
def train_epoch(model, dataloader, optimizer, scheduler, device, pooling="first", clip=1.0):
    model.train()
    total_loss = 0.0
    total_arc = 0.0
    total_label = 0.0
    for batch_idx, batch in enumerate(dataloader):
        encoding, token_to_word, gold_heads, gold_labels, word_masks = batch
        # forward
        arc_logits, rel_d, rel_h, word_mask = model(encoding, token_to_word, pooling=pooling)
        loss, arc_loss_val, label_loss_val = compute_loss(arc_logits, rel_d, rel_h, gold_heads, gold_labels, word_mask, model.label_classifier)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
        total_arc += arc_loss_val
        total_label += label_loss_val
    n_batches = len(dataloader)
    return total_loss / max(1, n_batches), total_arc / max(1, n_batches), total_label / max(1, n_batches)

def eval_model(model, dataloader, device, pooling="first"):
    model.eval()
    total_uas = 0.0
    total_las = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            encoding, token_to_word, gold_heads, gold_labels, word_masks = batch
            arc_logits, rel_d, rel_h, word_mask = model(encoding, token_to_word, pooling=pooling)
            uas, las = compute_metrics(arc_logits, rel_d, rel_h, gold_heads, gold_labels, word_mask, model.label_classifier)
            total_uas += uas
            total_las += las
            count += 1
    return total_uas / max(1, count), total_las / max(1, count)

# -------------------------
# Main entry: prepare data, build vocab of labels, create dataloaders & train
# -------------------------
def build_label_map(dataset_splits):
    # collect all labels from splits (train/valid/test)
    counter = Counter()
    for split in dataset_splits:
        for ex in split:
            counter.update(ex["deprel"])
    labels = sorted(counter.keys())
    label_to_id = {l: i for i, l in enumerate(labels)}
    id_to_label = {i: l for l, i in label_to_id.items()}
    return label_to_id, id_to_label

# def prepare_dataloaders(dataset_name="fr_gsd", batch_size=8, device="cuda"):
#     # Load UD French GSD as example; replace or add other treebanks for experiments
#     ds = load_dataset("universal_dependencies", dataset_name, trust_remote_code=True)
#     train_raw = ds["train"]
#     dev_raw = ds["validation"] if "validation" in ds.column_names else ds["test"]
#     test_raw = ds["test"]

#     label_to_id, id_to_label = build_label_map([train_raw, dev_raw, test_raw])

#     train_dataset = UDDatasetWrapper(train_raw)
#     dev_dataset = UDDatasetWrapper(dev_raw)
#     test_dataset = UDDatasetWrapper(test_raw)

#     tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")

#     def collate_wrapper(batch):
#         return ud_collate_fn(batch, tokenizer, label_to_id, device=device)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_wrapper)
#     dev_loader   = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_wrapper)
#     test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_wrapper)

#     return train_loader, dev_loader, test_loader, label_to_id, id_to_label, tokenizer
from datasets import Dataset
from conllu import parse_incr

def load_conllu_to_hf(filepath):
    """
    Load CoNLL-U file and keep UD head format as-is (0=root, 1+=word indices).
    After we add ROOT embedding at position 0, these indices will align perfectly.
    """
    with open(filepath, "r", encoding="utf-8") as data_file:
        data_list = []
        for tokenlist in parse_incr(data_file):
            tokens = [token["form"] for token in tokenlist]
            
            # Keep UD heads as-is: 0=root, 1=first word, 2=second word, etc.
            heads = []
            for token in tokenlist:
                if token["head"] is None:
                    heads.append(-100)  # ignore index for invalid heads
                else:
                    heads.append(token["head"])  # Keep original UD indexing
            
            deprels = [token["deprel"] for token in tokenlist]
            upos = [token["upos"] for token in tokenlist]
            
            data_list.append({
                "tokens": tokens, 
                "head": heads, 
                "deprel": deprels, 
                "upos": upos
            })
    return Dataset.from_list(data_list)

def prepare_dataloaders(dataset_name="fr_gsd", batch_size=8, device="cuda", debug_mode = False):
    # MANUAL LOAD: Bypass the blocked Hugging Face script
    from datasets import DatasetDict
    
    print(f"Loading local CONLLU files for {dataset_name}...")
    # These filenames must match the files in your Documents folder
    ds = DatasetDict({
        "train": load_conllu_to_hf(f"{dataset_name}-ud-train.conllu"),
        "validation": load_conllu_to_hf(f"{dataset_name}-ud-dev.conllu"),
        "test": load_conllu_to_hf(f"{dataset_name}-ud-test.conllu")
    })
    
    if debug_mode:
        print("🔧 DEBUG MODE: Using only 50 training examples")
        ds["train"] = ds["train"].select(range(min(50, len(ds["train"]))))
        ds["validation"] = ds["validation"].select(range(min(20, len(ds["validation"]))))
        ds["test"] = ds["test"].select(range(min(20, len(ds["test"]))))

    train_raw = ds["train"]
    dev_raw = ds["validation"]
    test_raw = ds["test"]

    label_to_id, id_to_label = build_label_map([train_raw, dev_raw, test_raw])

    train_dataset = UDDatasetWrapper(train_raw)
    dev_dataset = UDDatasetWrapper(dev_raw)
    test_dataset = UDDatasetWrapper(test_raw)

    tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")

    def collate_wrapper(batch):
        return ud_collate_fn(batch, tokenizer, label_to_id, device=device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_wrapper)
    dev_loader   = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_wrapper)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_wrapper)

    return train_loader, dev_loader, test_loader, label_to_id, id_to_label, tokenizer
def main(args):
    # Clear CUDA cache at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, dev_loader, test_loader, label_to_id, id_to_label, tokenizer = prepare_dataloaders(args.treebank, args.batch_size, device=device, debug_mode=args.debug)

    if args.arch == "feature":
        model = FeatureBasedParser(camembert_name="camembert-base",
                                   lstm_hidden=args.lstm_hidden,
                                   lstm_layers=args.lstm_layers,
                                   mlp_dim=args.mlp_dim,
                                   rel_labels=len(label_to_id),
                                   dropout=args.dropout)
        # Only parser params are optimized
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        model = FineTunedParser(camembert_name="camembert-base",
                                mlp_dim=args.mlp_dim,
                                rel_labels=len(label_to_id),
                                dropout=args.dropout)
        params = model.parameters()  # fine-tune whole model

    model.to(device)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    best_dev_las = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_arc, train_label = train_epoch(model, train_loader, optimizer, scheduler, device, pooling=args.pooling, clip=args.clip)
        dev_uas, dev_las = eval_model(model, dev_loader, device, pooling=args.pooling)
        print(f"Epoch {epoch} | train_loss={train_loss:.4f} (arc={train_arc:.4f}, lab={train_label:.4f}) | dev UAS={dev_uas:.4f} LAS={dev_las:.4f}")
        if dev_las > best_dev_las:
            best_dev_las = dev_las
            # Save checkpoint
            torch.save(model.state_dict(), f"best_{args.arch}_{args.treebank}.pt")
            print("Saved best model")

    # final test
    model.load_state_dict(torch.load(f"best_{args.arch}_{args.treebank}.pt"))
    test_uas, test_las = eval_model(model, test_loader, device, pooling=args.pooling)
    print(f"Test UAS={test_uas:.4f} LAS={test_las:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["feature", "fine"], default="fine", help="Which architecture to use")
    parser.add_argument("--treebank", type=str, default="fr_gsd", help="UD treebank name (dataset config) e.g., fr_gsd, fr_sequoia")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lstm_hidden", type=int, default=400)
    parser.add_argument("--lstm_layers", type=int, default=3)
    parser.add_argument("--mlp_dim", type=int, default=500)
    parser.add_argument("--dropout", type=float, default=0.33)
    parser.add_argument("--pooling", choices=["first","mean","max"], default="first")
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)



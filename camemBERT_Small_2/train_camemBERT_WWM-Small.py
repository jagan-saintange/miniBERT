from datasets import load_dataset
import torch
import torch
from transformers import (
    CamembertTokenizerFast,
    CamembertForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

class CamembertWWMCollator(DataCollatorForLanguageModeling):
    def torch_mask_tokens(self, inputs, special_tokens_mask=None, **kwargs): # On ajoute **kwargs ici
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Logique WWM pour CamemBERT
        for i in range(len(inputs)):
            # On convertit les IDs en tokens pour repérer les mots
            tokens = self.tokenizer.convert_ids_to_tokens(inputs[i].tolist())
            for j in range(1, len(tokens)):
                # Si le token ne commence PAS par ' ' (ou l'underscore de SentencePiece),
                # il appartient au mot précédent.
                if not tokens[j].startswith(" ") and not special_tokens_mask[i][j]:
                    # On propage la décision de masquage du premier token du mot
                    probability_matrix[i][j] = probability_matrix[i][j-1]

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        
        # 80% [MASK], 10% random, 10% original
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        return inputs, labels


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))

tokenizer = CamembertTokenizerFast(
    vocab_file="models/spm.model",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>"
)

dataset = load_dataset(
    "text",
    data_files={"train": "Oscar/shards/*.txt"}
)

# 1. Tokenisation initiale sans troncature
def tokenize_function(examples):
    return tokenizer(examples["text"], add_special_tokens=True)

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    num_proc=8 # Accélère le traitement CPU
)

# 2. Groupage par blocs de 512 (Packing)
def group_texts(examples):
    block_size = 512
    # Concatène tous les textes
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # On arrondit au multiple de 512 inférieur
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
        
    # Découpage en blocs de 512
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # Pour le MLM, on a besoin des labels (qui sont une copie des input_ids au départ)
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=8
)

model = CamembertForMaskedLM.from_pretrained("model_init")
model.resize_token_embeddings(len(tokenizer))


data_collator = CamembertWWMCollator(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)


training_args = TrainingArguments(
    output_dir="camembert_repro",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=2500,
    weight_decay=0.01,
    logging_steps=100,
    save_strategy="steps",
    save_steps=5000,
    save_total_limit=3,
    fp16=False,
    bf16=True,
    dataloader_num_workers=4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    data_collator=data_collator,
    tokenizer=tokenizer
)



trainer.train()
trainer.save_model("./camembert-repro-best")
tokenizer.save_pretrained("./camembert-repro-best")

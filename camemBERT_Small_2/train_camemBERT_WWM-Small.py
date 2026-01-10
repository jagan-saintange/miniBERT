"""
Projet de Machine Learning Avancé (MLA) - Sorbonne Université 
Script de reproduction : Pré-entraînement CamemBERT avec Whole Word Masking (WWM) 
Ce script implémente la logique de masquage par mot complet pour optimiser la cohérence sémantique des prédictions.
"""
from datasets import load_dataset
import torch
from transformers import (
    CamembertTokenizerFast,
    CamembertForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

class CamembertWWMCollator(DataCollatorForLanguageModeling):
    """
    Assure que tous les sous-tokens d'un même mot sont masqués simultanément pour augmenter la difficulté de la tâche MLM.
    """
    def torch_mask_tokens(self, inputs, special_tokens_mask=None, **kwargs):
        labels = inputs.clone()
        # Initialisation de la matrice de probabilité selon mlm_probability (0.15 par défaut)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # Identification et protection des tokens de structure (<s>, </s>, <pad>)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Logique de propagation du masque pour SentencePiece :
        # Un nouveau mot commence par le caractère ' ' (U+2581). Les tokens suivants sont des continuations.
        for i in range(len(inputs)):
            tokens = self.tokenizer.convert_ids_to_tokens(inputs[i].tolist())
            for j in range(1, len(tokens)):
                # Si le token j ne commence pas par l'underscore de SentencePiece, il est lié au token j-1
                if not tokens[j].startswith(" ") and not special_tokens_mask[i][j]:
                    probability_matrix[i][j] = probability_matrix[i][j-1]

        # Échantillonnage de Bernoulli pour déterminer les positions à masquer
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # Seuls les tokens masqués sont conservés pour le calcul de la Cross-Entropy Loss
        labels[~masked_indices] = -100 
        
        # Application de la règle 80/10/10 : [MASK] / Token aléatoire / Token original
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        return inputs, labels

# Configuration des optimisations Tensor Core pour architectures NVIDIA récentes
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Vérification de l'environnement d'exécution (GPU requis pour l'entraînement intensif)
print(f"CUDA status: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# Initialisation du tokenizer SentencePiece spécifique à CamemBERT
tokenizer = CamembertTokenizerFast(
    vocab_file="models/spm.model",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>"
)

# Chargement du corpus d'entraînement (shards au format texte brut)
dataset = load_dataset(
    "text",
    data_files={"train": "Oscar/shards/*.txt"}
)

def tokenize_function(examples):
    return tokenizer(examples["text"], add_special_tokens=True)

# Tokenisation distribuée sur les cœurs CPU disponibles
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    num_proc=4
)

def group_texts(examples):
    """
    Concaténation et segmentation des séquences en blocs de longueur fixe (Packing).
    Indispensable pour minimiser le padding et optimiser le débit de tokens (throughput).
    """
    block_size = 512
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
        
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# Structuration finale du dataset pour le Masked Language Modeling (MLM)
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=8
)

# Chargement des poids du modèle et ajustement des embeddings (si modification du vocabulaire)
model = CamembertForMaskedLM.from_pretrained("model_init")
model.resize_token_embeddings(len(tokenizer))

# Initialisation du collator avec la probabilité de masquage standard (15%)
data_collator = CamembertWWMCollator(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Hyperparamètres d'entraînement calibrés pour la reproduction 
training_args = TrainingArguments(
    output_dir="camembert_repro",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,  # Batch effectif de 64
    learning_rate=1e-4,
    warmup_steps=2500,              # Stabilisation du gradient en début d'entraînement
    weight_decay=0.01,
    logging_steps=100,
    save_strategy="steps",
    save_steps=5000,
    save_total_limit=3,
    fp16=False,                     # BF16 préféré sur architectures récentes
    bf16=True,                      # Utilisation du format Bfloat16 pour la stabilité numérique
    dataloader_num_workers=4
)

# Initialisation du moteur d'entraînement Trainer d'Hugging Face
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Cycle d'entraînement et sauvegarde finale pour le rendu du projet 
trainer.train()
trainer.save_model("./camembert-repro-best")
tokenizer.save_pretrained("./camembert-repro-best")

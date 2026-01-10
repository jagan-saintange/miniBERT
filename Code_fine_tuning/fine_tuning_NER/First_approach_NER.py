import torch
from transformers import (
    CamembertForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from datasets import load_dataset
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score


import os
import shutil
from transformers import TrainerCallback

################################
# BLOC POUR NE PAS SATURER LA MEMOIRE GPU
###################################### 
class KeepBestCheckpointsCallback(TrainerCallback):
    def __init__(self, keep_best=2, metric_name="eval_f1"): #garder le top 2 de F1 score
        self.keep_best = keep_best
        self.metric_name = metric_name
        self.best = []  # list of (score, path)

    def on_save(self, args, state, control, **kwargs):
        # Current checkpoint path
        ckpt_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

        # Get latest metric
        metrics = state.log_history[-1] if state.log_history else {}
        score = metrics.get(self.metric_name)

        if score is None:
            return control

        # Add and sort
        self.best.append((score, ckpt_path))
        self.best = sorted(self.best, key=lambda x: x[0], reverse=True)

        # Keep only top N
        allowed = set([p for _, p in self.best[:self.keep_best]])

        # Delete all others
        for dirname in os.listdir(args.output_dir):
            full = os.path.join(args.output_dir, dirname)
            if dirname.startswith("checkpoint-") and full not in allowed:
                shutil.rmtree(full, ignore_errors=True)

        return control
###########################"


class CamemBERTNERModel:
    def __init__(self, num_labels=7):
        print(f"[DEBUG] Initializing CamemBERTNERModel with {num_labels} labels")
        # Initialize CamemBERT base model
        self.model = CamembertForTokenClassification.from_pretrained(
            "./camembert_v2_MLM_40000step", #CHANGER LE PATH EN FONCTION
            num_labels=num_labels
        )
        print("[DEBUG] Model loaded successfully")
        
        # Configure tokenizer with fast tokenization enabled
        #print("[DEBUG] Loading tokenizer with fast tokenization...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "./camembert_v2_MLM_40000step", #CHANGER LE PATH EN FONCTION
            use_fast=True active le tokenizer "Fast" 
        )
        #print("[DEBUG] Tokenizer loaded successfully (fast tokenization enabled)")

        self.num_labels = num_labels # Stocker le nombre de labels (utile pour compute_metrics)

    def prepare_dataset(self, dataset, language='fr'):
        """Prepare dataset by tokenizing and aligning labels."""
        print(f"[DEBUG] Preparing dataset with {len(dataset)} samples")
        print(f"[DEBUG] Dataset columns: {dataset.column_names}")
        
        def tokenize_and_align_labels(examples):
            #print(f"[DEBUG] Tokenizing batch of {len(examples['tokens'])} samples")
            # Official HuggingFace implementation - canonical version
            tokenized_inputs = self.tokenizer(
                examples["tokens"],
                truncation=True, # coupe si tropg lon
                is_split_into_words=True # on indique que "examples['tokens']" est déjà tokenisé en mots
            )

            labels = []
            for i, label in enumerate(examples["ner_tags"]): 
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # mapping des sous mots
                previous_word_idx = None # pour détecter
                label_ids = []

                for word_idx in word_ids:
                    if word_idx is None: # spécial
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx: # premier sous-mot, label
                        label_ids.append(label[word_idx])
                    else: # suite mot
                        label_ids.append(-100)

                    previous_word_idx = word_idx

                labels.append(label_ids)

            tokenized_inputs["labels"] = labels # on injecte les labels alignés dans les features du batch
            
            # Debug length check
            #print(f"[DEBUG] Sample {i}: input_ids length = {len(tokenized_inputs['input_ids'][i])}, labels length = {len(label_ids)}")
            
            return tokenized_inputs
        
        # Apply tokenization
        print("[DEBUG] Mapping tokenization function to dataset...")
        tokenized_dataset = dataset.map(
            tokenize_and_align_labels, 
            batched=True,
            batch_size=32,
            remove_columns=dataset.column_names
        )
        print(f"[DEBUG] Dataset preparation complete: {len(tokenized_dataset)} samples")
        
        return tokenized_dataset
    
    
    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2) # on récupère la classe prédite pour chaque token

        # Filtrage des labels -100 (tokens spéciaux + sous-mots ignorés)
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # Calcul des métriques seqeval (NER)
        f1 = f1_score(true_labels, true_predictions)
        precision = precision_score(true_labels, true_predictions)
        recall = recall_score(true_labels, true_predictions)

        return {
            "f1": f1, # métrique principale
            "precision": precision,
            "recall": recall
        }


    def train(self, train_dataset, val_dataset):
        """Train the NER model using HuggingFace Trainer."""
        print("[DEBUG] Configuring training arguments...")
        training_args = TrainingArguments(
            output_dir='./results', # dossier où checkpoints + logs seront stockés
            num_train_epochs=30,
            per_device_train_batch_size=16, # batch size entraînement
            per_device_eval_batch_size=16, # batch size validation
            eval_strategy="epoch", # évaluation à chaque epoch
            save_strategy="epoch", # sauvegarde à chaque epoch
            learning_rate=5e-5,
            weight_decay=0.0, # depend si regularisation L2
            warmup_steps=500, # warmup (pas comme dans le papier camemBERT)
            logging_steps=100, # logs
            load_best_model_at_end=True, # recharge le meilleur checkpoint (selon F1)
            metric_for_best_model="f1",
            push_to_hub=False,
        )
        print("[DEBUG] Training arguments configured")
        
        # Initialize DataCollator for token classification
        print("[DEBUG] Initializing DataCollatorForTokenClassification...")
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        print("[DEBUG] DataCollator initialized")
        
        print("[DEBUG] Initializing Trainer...")
        trainer = Trainer(
            model=self.model,
            args=training_args, # hyperparamètres d'entraînement
            train_dataset=train_dataset, # dataset d'entraînement
            eval_dataset=val_dataset, # dataset de validation
            tokenizer=self.tokenizer,
            data_collator=data_collator, # padding + labels
            compute_metrics=self.compute_metrics,
            callbacks=[KeepBestCheckpointsCallback(keep_best=2)],
        )
        print("[DEBUG] Trainer initialized")
        
        print("[DEBUG] Starting training...")
        trainer.train()
        print("[DEBUG] Training completed")
        
        # Save the best model
        print("[DEBUG] Saving best model...")
        trainer.save_model('./best_camembert_ner_model')
        print("[DEBUG] Model saved to ./best_camembert_ner_model")
        
        return trainer

# Example usage
def main():
    print("[DEBUG] ========== STARTING NER TRAINING PIPELINE ==========")
    
    # Load French WikiANN dataset
    print("[DEBUG] Loading WikiANN French dataset...")
    dataset = load_dataset("wikiann", "fr")
    print(f"[DEBUG] WikiANN dataset loaded successfully")
    print(f"[DEBUG] Dataset splits: {list(dataset.keys())}")
    print(f"[DEBUG] Train set size: {len(dataset['train'])}")
    print(f"[DEBUG] Validation set size: {len(dataset['validation'])}")
    print(f"[DEBUG] Test set size: {len(dataset['test'])}")
    
    # Get label information from dataset
    print("[DEBUG] Extracting label information...")
    label_list = dataset["train"].features["ner_tags"].feature.names
    print(f"[DEBUG] Labels found: {label_list}")
    print(f"[DEBUG] Number of labels: {len(label_list)}")
    # mapping label / id pour compute_metrics
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for i, label in enumerate(label_list)}
    print("[DEBUG] Label mappings created")
    
    # Initialize model with correct number of labels
    print(f"[DEBUG] Initializing CamemBERT model with {len(label_list)} labels...")
    ner_model = CamemBERTNERModel(num_labels=len(label_list))
    ner_model.id2label = id_to_label
    ner_model.label2id = label_to_id

    # Mise à jour de la config interne du modèle (HuggingFace)
    ner_model.model.config.id2label = id_to_label 
    ner_model.model.config.label2id = label_to_id

    print("[DEBUG] Model initialization complete")
    
    # Prepare datasets
    print("\n[DEBUG] ========== PREPARING TRAINING DATASET ==========")
    train_dataset = ner_model.prepare_dataset(dataset["train"])
    
    print("\n[DEBUG] ========== PREPARING VALIDATION DATASET ==========")
    val_dataset = ner_model.prepare_dataset(dataset["validation"])
    
    # Train the model
    print("\n[DEBUG] ========== STARTING TRAINING PROCESS ==========")
    trainer = ner_model.train(train_dataset, val_dataset)
    
    print("\n[DEBUG] ========== TRAINING PIPELINE COMPLETE ==========")
    print("Training completed successfully!")

if __name__ == "__main__":
    main()


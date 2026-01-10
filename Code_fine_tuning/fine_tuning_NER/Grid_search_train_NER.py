import torch
from transformers import (
    CamembertForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    TrainerCallback
)
from datasets import load_dataset
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score

import os
import shutil

###########################



PATH = ""



############################
bloc pour ne pas saturer stockage
###########################################
class CheckpointPruningCallback(TrainerCallback):
    def __init__(self, keep_best=2, keep_last=0, metric_name="eval_f1"):
        self.keep_best = keep_best
        self.keep_last = keep_last
        self.metric_name = metric_name
        self.best_checkpoints = []   # list of (score, path)
        self.last_checkpoints = []   # list of paths

    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

        # Track last checkpoints
        self.last_checkpoints.append(checkpoint_path)
        if len(self.last_checkpoints) > self.keep_last:
            self.last_checkpoints.pop(0)

        # Track best checkpoints
        metrics = state.log_history[-1] if state.log_history else {}
        score = metrics.get(self.metric_name, None)

        if score is not None:
            self.best_checkpoints.append((score, checkpoint_path))
            self.best_checkpoints = sorted(self.best_checkpoints, key=lambda x: x[0], reverse=True)
            self.best_checkpoints = self.best_checkpoints[:self.keep_best]

        # Compute allowed checkpoints
        allowed = set([p for _, p in self.best_checkpoints] + self.last_checkpoints)

        # Delete others
        for dirname in os.listdir(args.output_dir):
            full_path = os.path.join(args.output_dir, dirname)
            if dirname.startswith("checkpoint-") and full_path not in allowed:
                print(f"[PRUNE] Removing old checkpoint: {full_path}")
                shutil.rmtree(full_path, ignore_errors=True)

        return control
###########################################

class CamemBERTNERModel:
    def __init__(self, num_labels=7):
        print(f"[DEBUG] Initializing CamemBERTNERModel with {num_labels} labels")
        # Initialiser le modèle Camembert pour la classifiacation de tokens
        self.model = CamembertForTokenClassification.from_pretrained(
            PATH,
            #'camembert-base
            num_labels=num_labels # Nombre de labels de classe NER 
            #(va reconnaitre si c'est une personne, organisation, lieu etc, ici 7 classes)
        )
        print("[DEBUG] Model loaded successfully")
        
        # Configurer tokenizer
        '''print("[DEBUG] Loading tokenizer with fast tokenization...")'''
        self.tokenizer = AutoTokenizer.from_pretrained(
            PATH,
            #'camembert-base', # On use le modèle de camembert-base
            use_fast=True # On active fast tokenizer (méthode post camemBERT)
        )
        '''print("[DEBUG] Tokenizer loaded successfully (fast tokenization enabled)")'''

        self.num_labels = num_labels

    def prepare_dataset(self, dataset, language='fr'):
        # Tokenisation du dataset et alignement des labels avec les tokens subwords
        # SEULE LE PREMIER SOUS-MOT RECOIT LE LABEL LES AUTRES REC-100
        print(f"[DEBUG] Preparing dataset with {len(dataset)} samples")
        print(f"[DEBUG] Dataset columns: {dataset.column_names}")
        
        def tokenize_and_align_labels(examples):
            '''print(f"[DEBUG] Tokenizing batch of {len(examples['tokens'])} samples")'''
            # partie tokenization d'exemples, 
            tokenized_inputs = self.tokenizer(
                examples["tokens"], # echantillon
                truncation=True, # Si trop long on coupe
                is_split_into_words=True # Car on a déjà des tokens en entrée ici
            )

            labels = [] # labels alignés
            for i, label in enumerate(examples["ner_tags"]):  # Pour chaque echantillion...
                word_ids = tokenized_inputs.word_ids(batch_index=i) # FAst tokenizer donne cette méthode pour 
                #indexage rapide des sous-mots selon les mots complets ou -100 pour les tokens spécaiux
                previous_word_idx = None
                label_ids = []

                for word_idx in word_ids: # pour chaque subword
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx: # Si premier subword du mot ...
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)

                    previous_word_idx = word_idx

                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            
            # Debug longueur
            '''print(f"[DEBUG] Sample {i}: input_ids length = {len(tokenized_inputs['input_ids'][i])}, labels length = {len(label_ids)}")'''
            
            return tokenized_inputs
        
        # Appliquer les résultats de la tokenization
        print("[DEBUG] Mapping tokenization function to dataset...")
        tokenized_dataset = dataset.map(
            tokenize_and_align_labels, 
            batched=True, # en batch
            batch_size=32, # taille du batch pour la tokenisation (indépendant du batch d'entraînement)
            remove_columns=dataset.column_names 
        )
        print(f"[DEBUG] Dataset preparation complete: {len(tokenized_dataset)} samples")
        
        return tokenized_dataset # on renvoie le dataset tokenisé + labels alignés
    
    
    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2) # Prediction pour chaque token, classe la plus probable

        # prédiction 
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100] # on filtre les -100 dehors
            # on garde uniquement les tokens réellement annotés (pas les sous-mots ignorés)
            for prediction, label in zip(predictions, labels)
        ] # iterer pour chaque prediction et label
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # scores
        f1 = f1_score(true_labels, true_predictions)
        precision = precision_score(true_labels, true_predictions) 
        recall = recall_score(true_labels, true_predictions)

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall
        }


    def train(self, train_dataset, val_dataset):
        """Train the NER model using HuggingFace Trainer."""
        print("[DEBUG] Configuring training arguments...")
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=30,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            eval_strategy="epoch", # évaluation à chaque epoch
            save_strategy="epoch", # save à chaque epoch
            learning_rate=5e-5,
            weight_decay=0,
            warmup_steps=0, # /!\ Pas de warmup comme dans l'article (maj par rapport à la version précendente)
            lr_scheduler_type="constant", # Le papier propose un taux d'apprentissage fixe (maj par rapport à la version précendente)
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            push_to_hub=False,
        )
        print("[DEBUG] Training arguments configured")
        
        # Initialize DataCollator for token classification
        # DataCollator padding dynamique + alignement des labels
        print("[DEBUG] Initializing DataCollatorForTokenClassification...")
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        print("[DEBUG] DataCollator initialized")
        
        print("[DEBUG] Initializing Trainer...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator, # collator pour gérer les labels + padding
            compute_metrics=self.compute_metrics, # fonction de calcul des métriques seqeval
            callbacks=[CheckpointPruningCallback(keep_best=2, keep_last=2)], #top 2, last 2
        )
        print("[DEBUG] Trainer initialized")
        
        print("[DEBUG] Starting training...")
        trainer.train(resume_from_checkpoint=False,)
        print("[DEBUG] Training completed")
        
        # Save the best model
        print("[DEBUG] Saving best model...")
        trainer.save_model('./best_camembert_ner_model')
        print("[DEBUG] Model saved to ./best_camembert_ner_model")
        
        return trainer

def run_grid_search(dataset, label_list):
    """
    Runs grid search over learning rates and batch sizes,
    as described in the CamemBERT paper for NER.
    """

    learning_rates = [1e-5]#, 2e-5, 3e-5, 5e-5] # remplir à souhait
    batch_sizes = [8]#, 16, 32]  #ça aussi
    results = []

    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"\n========== GRID SEARCH RUN: LR={lr}, BS={bs} ==========\n")

            # Initialize model
            ner_model = CamemBERTNERModel(num_labels=len(label_list))
            ner_model.id2label = {i: label for i, label in enumerate(label_list)}
            ner_model.label2id = {label: i for i, label in enumerate(label_list)}
            ner_model.model.config.id2label = ner_model.id2label
            ner_model.model.config.label2id = ner_model.label2id

            # Prepare datasets
            train_dataset = ner_model.prepare_dataset(dataset["train"])
            val_dataset = ner_model.prepare_dataset(dataset["validation"])

            # Training arguments for this run
            training_args = TrainingArguments(
                output_dir=f'./results_lr{lr}_bs{bs}',
                num_train_epochs=30,
                per_device_train_batch_size=bs, # batch size variable
                per_device_eval_batch_size=bs,
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=lr, # LR variable
                weight_decay=0,
                warmup_steps=0, # pas de warm
                lr_scheduler_type="constant",
                logging_steps=100,
                load_best_model_at_end=True, # recharge auto du meilleur checkpoint
                metric_for_best_model="f1",
                push_to_hub=False,
            )

            trainer = Trainer(
                model=ner_model.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=ner_model.tokenizer,
                callbacks=[CheckpointPruningCallback(keep_best=2, keep_last=2)], # 2 previous checkpoints
                data_collator=DataCollatorForTokenClassification(ner_model.tokenizer), # padding + alignement des labels
                compute_metrics=ner_model.compute_metrics,
            )

            trainer.train(resume_from_checkpoint=False)

            # Evaluate on validation set
            metrics = trainer.evaluate()
            f1 = metrics["eval_f1"]

            print(f"--> Finished run: LR={lr}, BS={bs}, F1={f1:.4f}")

            results.append({
                "learning_rate": lr,
                "batch_size": bs,
                "f1": f1
            })

    # Sort by best F1
    results = sorted(results, key=lambda x: x["f1"], reverse=True)

    print("\n========== GRID SEARCH RESULTS ==========")
    for r in results:
        print(f"LR={r['learning_rate']}, BS={r['batch_size']}, F1={r['f1']:.4f}")

    print("\nBest configuration:")
    print(results[0])

    return results


def main():
    print("[DEBUG] ========== STARTING NER TRAINING PIPELINE ==========")
    
    # Load French WikiANN dataset
    print("[DEBUG] Loading WikiANN French dataset...")
    dataset = load_dataset("wikiann", "fr")
    print(f"[DEBUG] WikiANN dataset loaded successfully")
    
    # Get label information depuis dataset
    print("[DEBUG] Extracting label information...")
    label_list = dataset["train"].features["ner_tags"].feature.names
    print(f"[DEBUG] Labels found: {label_list}")
    
    print("[DEBUG] ========== STARTING GRID SEARCH ==========")
    run_grid_search(dataset, label_list) # lancement du grid search
    
    print("\n[DEBUG] ========== GRID SEARCH COMPLETE ==========")


if __name__ == "__main__":
    main()



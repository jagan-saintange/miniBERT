import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import f1_score
from datasets import load_from_disk

# Load the fine-tuned model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

# Load your test dataset (with tokenized inputs and labels)
test_dataset = load_from_disk("./test_dataset")

# Create a minimal trainer to use its predict method
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./temp_results",
        per_device_eval_batch_size=16,
    ),
    tokenizer=tokenizer,
)

# Make predictions on the test dataset
predictions = trainer.predict(test_dataset)
predicted_labels = predictions.predictions.argmax(-1)  # Get predicted class IDs
true_labels = test_dataset["labels"]  # Get true labels

# Flatten the lists to compute F1 score (exclude -100 padding labels)
true_labels_flat = [label for labels in true_labels for label in labels if label != -100]
predicted_labels_flat = [pred for preds in predicted_labels for pred in preds if pred != -100]

# Calculate the F1 Score
f1 = f1_score(true_labels_flat, predicted_labels_flat, average='weighted')  # or 'micro', 'macro'
print("F1 Score:", f1)

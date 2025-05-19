import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch

# Load and inspect the dataset
df = pd.read_csv('complaints_preprocessed.csv').sample(500, random_state=42)

# Train/test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Choose a suitable model (you can replace this with a domain-specific one later)
model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=df["Target"].nunique()
)

# Preprocessing function
def preprocess(batch):
    texts = [str(x) for x in batch["narrative"]]
    tokenized = tokenizer(texts, truncation=True, padding='max_length')
    tokenized["labels"] = batch["Target"]
    return tokenized


# Tokenize
train_dataset = train_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)

# Padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Compute metrics for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "report": classification_report(labels, predictions, output_dict=False),
    }

# Training setup
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Evaluate
results = trainer.evaluate()
print("\nEvaluation Results:")
print("Accuracy:", results["eval_accuracy"])
print("Classification Report:")
print(results["eval_report"])

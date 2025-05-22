import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import transformers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch

# Load data
df = pd.read_csv('complaints_preprocessed.csv')

# Rename label column to 'labels' for HF Trainer
df = df.rename(columns={"Target": "labels"})

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Reset index
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load tokenizer and model
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=df['labels'].nunique())

# Tokenization function
def preprocess(batch):
    texts = batch["narrative"]
    # Ensure all elements are strings (optional)
    texts = [str(t) if t is not None else "" for t in texts]
    return tokenizer(texts, truncation=True, padding="max_length")




# Apply tokenization
train_dataset = train_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)

# Set dataset format for Trainer
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

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

# Evaluate on test set
eval_results = trainer.evaluate()
print(f"Test set accuracy: {eval_results['eval_accuracy']:.4f}")

# Optional: print classification report on test set predictions
test_predictions = trainer.predict(test_dataset)
preds = torch.argmax(torch.tensor(test_predictions.predictions), dim=-1)
print(classification_report(test_predictions.label_ids, preds))

# Save best model
trainer.save_model("./best_model")

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch

# Load dataset (use a small sample for quick training)
df = pd.read_csv("complaints_preprocessed.csv").sample(frac=0.2, random_state=42)
train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

def run_model(model_checkpoint):
    print(f"\n--- Running with model: {model_checkpoint} ---")
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=df["Target"].nunique()
    )

    # Define LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,                         # Rank
        lora_alpha=16,               # Scaling
        lora_dropout=0.1,            # Dropout for LoRA layers
    )

    # Wrap the model with PEFT
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()  # Optional: to verify only a few params are trainable


    def preprocess(batch):
        # Prompting style input
        texts = [
            f"Classify the type of complaint from the following narrative: '{n}'"
            if n is not None else "Classify the type of complaint from the following narrative: ''"
            for n in batch["narrative"]
        ]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=128)



    # Apply preprocessing and add labels
    tokenized_train = train_dataset.map(preprocess, batched=True)
    tokenized_train = tokenized_train.rename_column("Target", "labels")
    tokenized_test = test_dataset.map(preprocess, batched=True)
    tokenized_test = tokenized_test.rename_column("Target", "labels")

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "report": classification_report(labels, predictions, output_dict=False),
        }

    training_args = TrainingArguments(
        output_dir=f"./results_{model_checkpoint.replace('/', '_')}",
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
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.evaluate()

    print("\nEvaluation Results:")
    print("Accuracy:", results["eval_accuracy"])
    print("Classification Report:\n", results["eval_report"])

model_names = [
    #"distilbert-base-uncased",
    "prajjwal1/bert-tiny"
    #"google/bert_uncased_L-2_H-128_A-2"
]

# Run all 3 models
for model_name in model_names:
    run_model(model_name)


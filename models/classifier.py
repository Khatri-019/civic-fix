"""
Train and serve a DistilBERT classifier for complaint category + severity.
Training: python -m models.classifier --train
Inference: imported as a module
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import evaluate

MODEL_NAME = "distilbert-base-uncased"
SAVE_DIR = Path("models/saved/classifier")
LABEL_DIR = Path("models/saved/labels")


class ComplaintDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_data(csv_path: Path, label_col: str):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["raw_complaint", label_col])
    le = LabelEncoder()
    labels = le.fit_transform(df[label_col])
    return df["raw_complaint"].tolist(), labels, le


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train(label_col: str, csv_path: Path, epochs: int = 4):
    texts, labels, le = load_data(csv_path, label_col)
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=128)
    val_enc = tokenizer(X_val, truncation=True, padding=True, max_length=128)

    train_dataset = ComplaintDataset(train_enc, y_train)
    val_dataset = ComplaintDataset(val_enc, y_val)

    save_path = SAVE_DIR / label_col
    save_path.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(le.classes_)
    )

    args = TrainingArguments(
        output_dir=str(save_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,  # each checkpoint is ~766 MB; keep only the best
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        # logging_dir removed: no longer a TrainingArguments parameter in transformers 5
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    # Save label mapping
    label_map = {int(i): cls for i, cls in enumerate(le.classes_)}
    with open(LABEL_DIR / f"{label_col}_labels.json", "w") as f:
        json.dump(label_map, f)

    # Eval report
    preds = np.argmax(trainer.predict(val_dataset).predictions, axis=-1)
    print(f"\n=== {label_col} Classification Report ===")
    print(classification_report(y_val, preds, target_names=le.classes_))

    return str(save_path)


_loaded_models = {}


def predict(text: str, label_col: str) -> str:
    global _loaded_models
    if label_col not in _loaded_models:
        save_path = SAVE_DIR / label_col
        if not save_path.exists():
            return None
        from transformers import pipeline
        _loaded_models[label_col] = {
            "pipe": pipeline("text-classification", model=str(save_path), device=-1),
            "labels": json.load(open(LABEL_DIR / f"{label_col}_labels.json")),
        }
    entry = _loaded_models[label_col]
    result = entry["pipe"](text, truncation=True, max_length=128)[0]
    label_id = int(result["label"].replace("LABEL_", ""))
    return entry["labels"][str(label_id)]  # JSON object keys are strings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--data", default="data/synthetic/complaints.csv")
    parser.add_argument("--epochs", type=int, default=15)  # plan used 4; too few for ~220 rows (val acc still rising)
    args = parser.parse_args()

    if args.train:
        csv = Path(args.data)
        print("Training category classifier...")
        train("category", csv, args.epochs)
        print("\nTraining severity classifier...")
        train("severity", csv, args.epochs)
        print("\nDone. Models saved to models/saved/classifier/")

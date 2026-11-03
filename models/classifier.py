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
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
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


def load_data(csv_path: Path, label_col: str, skip_contradicting: bool = False):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["raw_complaint", label_col])
    if skip_contradicting:
        # Real data: a contradicting row keeps its photo's severity, so the label doesn't describe the text
        df = df[~df["has_contradiction"].astype(str).str.lower().eq("true")]
    le = LabelEncoder()
    labels = le.fit_transform(df[label_col])
    # Real data has several texts per photo: group them so a photo never sits in both train and validation
    groups = df["image_id"].fillna(df["record_id"]).tolist() if "image_id" in df.columns else None
    return df["raw_complaint"].tolist(), labels, le, groups


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    scores = metric.compute(predictions=predictions, references=labels)
    scores["macro_f1"] = f1_score(labels, predictions, average="macro", zero_division=0)
    return scores


class WeightedTrainer(Trainer):
    """Class-weighted loss, so a model can't score well by always predicting the majority class."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels,
                                                 weight=self.class_weights.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss


def train(label_col: str, csv_path: Path, epochs: int = 4, skip_contradicting: bool = False, balanced: bool = False):
    texts, labels, le, groups = load_data(csv_path, label_col, skip_contradicting)
    if groups is None:
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
    else:  # ~20% validation, split by photo
        train_idx, val_idx = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                                  .split(texts, labels, groups))
        X_train, y_train = [texts[i] for i in train_idx], labels[train_idx]
        X_val, y_val = [texts[i] for i in val_idx], labels[val_idx]
    print(f"{label_col}: {len(X_train)} train / {len(X_val)} validation rows")

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
        metric_for_best_model="macro_f1" if balanced else "accuracy",
        # logging_dir removed: no longer a TrainingArguments parameter in transformers 5
        logging_steps=10,
        report_to="none",
    )

    trainer_kwargs = dict(model=model, args=args, train_dataset=train_dataset, eval_dataset=val_dataset,
                          compute_metrics=compute_metrics)
    if balanced:
        weights = compute_class_weight("balanced", classes=np.arange(len(le.classes_)), y=y_train)
        trainer = WeightedTrainer(**trainer_kwargs, class_weights=torch.tensor(weights, dtype=torch.float))
    else:
        trainer = Trainer(**trainer_kwargs)
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
    print(classification_report(y_val, preds, labels=range(len(le.classes_)), target_names=le.classes_,
                                zero_division=0))

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
    parser.add_argument("--severity-skip-contradicting", action="store_true",
                        help="drop has_contradiction rows from severity training (use for data/real/train.csv)")
    parser.add_argument("--balanced-severity", action="store_true",
                        help="class-weighted loss + best epoch by macro-F1 for severity (stops always-Medium)")
    parser.add_argument("--only", choices=["category", "severity"], help="train just one of the two models")
    args = parser.parse_args()

    if args.train:
        csv = Path(args.data)
        if args.only != "severity":
            print("Training category classifier...")
            train("category", csv, args.epochs)
        if args.only != "category":
            print("\nTraining severity classifier...")
            train("severity", csv, args.epochs, args.severity_skip_contradicting, args.balanced_severity)
        print("\nDone. Models saved to models/saved/classifier/")

"""
Trains a TF-IDF + Logistic Regression fake-news classifier and saves:
  - models/model.joblib       (trained LogisticRegression)
  - models/vectorizer.joblib  (fitted TfidfVectorizer)
  - models/metrics.json       (accuracy/precision/recall/f1 + confusion matrix)

Usage:
    python ml/train_model.py --data data/sample_data.csv
"""

import argparse
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Allow running this script directly (python ml/train_model.py) or as a module.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Be forgiving about column names — Kaggle's dataset sometimes uses
    # "title"/"text" separately; combine them if "text" alone is missing.
    if "text" not in df.columns and "title" in df.columns:
        df["text"] = df["title"].fillna("") + " " + df.get("text", "").fillna("")

    if "label" not in df.columns:
        raise ValueError("CSV must have a 'label' column (e.g. REAL/FAKE or 0/1).")

    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_data.csv", help="Path to training CSV")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=5000)
    args = parser.parse_args()

    print(f"Loading data from {args.data} ...")
    df = load_data(args.data)
    print(f"Loaded {len(df)} rows. Label distribution:\n{df['label'].value_counts()}\n")

    print("Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=args.test_size,
        random_state=42, stratify=df["label"],
    )

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test_vec)
    labels_sorted = sorted(df["label"].unique())

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "labels": labels_sorted,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels_sorted).tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    print("\n=== Results ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print(f"Confusion matrix (labels={labels_sorted}):")
    for row in metrics["confusion_matrix"]:
        print(row)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "model.joblib"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.joblib"))
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model + vectorizer + metrics to {MODELS_DIR}/")


if __name__ == "__main__":
    main()

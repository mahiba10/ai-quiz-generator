"""Train and evaluate the topic classifier (the ONLY machine-learning part of the project).

Run from the project root:   python ml/train_model.py

What it does
1. Loads data/topic_training_data.csv (text, label) and cleans it.
2. Splits it into a training set (80%) and a test set (20%), stratified, seed 42.
3. Compares two candidate models using 5-fold cross-validation on the TRAINING set only.
4. Trains the better candidate on the training set and evaluates it ONCE on the test set.
5. Saves the model (models/topic_classifier.joblib) and the real measured metrics
   (models/metrics.json). Nothing is hard-coded.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "topic_training_data.csv"
MODEL_PATH = BASE_DIR / "models" / "topic_classifier.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

RANDOM_STATE = 42   # fixed seed -> same split and same results every run
TEST_SIZE = 0.2     # 20% of data is kept aside for the final test
MIN_PER_CLASS = 10  # refuse to train if a topic has too few examples


def load_and_clean_data(path=DATA_PATH):
    """Read the CSV, validate columns, strip whitespace and drop empty/duplicate sentences."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    df = pd.read_csv(path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain the columns 'text' and 'label'.")
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["text"] != ""]
    # Duplicates would let the same sentence appear in both train and test (data leakage).
    df = df.assign(_key=df["text"].str.lower()).drop_duplicates("_key").drop(columns="_key")
    counts = df["label"].value_counts()
    if counts.min() < MIN_PER_CLASS or len(counts) < 2:
        raise ValueError(f"Each topic needs at least {MIN_PER_CLASS} sentences. Found: {counts.to_dict()}")
    return df.reset_index(drop=True)


def build_pipeline(classifier):
    """TF-IDF turns text into numbers; the classifier learns from those numbers.
    Keeping both in one Pipeline means TF-IDF is fitted ONLY on training text (no leakage)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
        ("clf", classifier),
    ])


def run(data_path=DATA_PATH, model_path=MODEL_PATH, metrics_path=METRICS_PATH):
    """Train, evaluate and save. Returns the metrics dictionary."""
    df = load_and_clean_data(data_path)
    labels = sorted(df["label"].unique())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"])

    candidates = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        # Larger C = weaker regularisation = more confident (sharper) probabilities.
        "Logistic Regression (C=10)": LogisticRegression(C=10, max_iter=1000, random_state=RANDOM_STATE),
    }
    # Model choice uses cross-validation on the TRAINING data only; the test set is not touched.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    for name, clf in candidates.items():
        scores = cross_val_score(build_pipeline(clf), X_train, y_train, cv=cv, scoring="f1_macro")
        cv_results[name] = {"mean_macro_f1": float(scores.mean()), "std": float(scores.std())}
    best_name = max(cv_results, key=lambda n: cv_results[n]["mean_macro_f1"])

    model = build_pipeline(candidates[best_name])
    model.fit(X_train, y_train)

    # Final evaluation on unseen test data (done once).
    y_pred = model.predict(X_test)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    metrics = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "sklearn_version": sklearn.__version__,
        "chosen_model": best_name,
        "cross_validation_on_training_set": cv_results,
        "n_samples_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "samples_per_label": {k: int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "labels": labels,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_macro_precision": float(prec),
        "test_macro_recall": float(rec),
        "test_macro_f1": float(f1),
        "classification_report": classification_report(y_test, y_pred, labels=labels,
                                                       output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "note": "Measured on a very small test set; treat these numbers as indicative only.",
    }

    model_path, metrics_path = Path(model_path), Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    try:
        m = run()
    except (FileNotFoundError, ValueError) as err:
        print(f"Training failed: {err}")
        sys.exit(1)
    print(f"Samples: {m['n_samples_total']} (train {m['n_train']}, test {m['n_test']})")
    for name, r in m["cross_validation_on_training_set"].items():
        print(f"CV macro-F1 | {name}: {r['mean_macro_f1']:.3f} +/- {r['std']:.3f}")
    print(f"Chosen model: {m['chosen_model']}")
    print(f"Test accuracy: {m['test_accuracy']:.3f}  macro-P: {m['test_macro_precision']:.3f}  "
          f"macro-R: {m['test_macro_recall']:.3f}  macro-F1: {m['test_macro_f1']:.3f}")
    print(f"Saved model to {MODEL_PATH} and metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()

"""Persist reproducible held-out evaluation metrics for gesture classifiers."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, features, labels, output_dir: str | Path, model_label: str):
    """Evaluate a fitted model on a held-out test set and save machine-readable reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = model.predict(features)
    classes = [str(value) for value in model.classes_]
    matrix = confusion_matrix(labels, predictions, labels=model.classes_)
    report = classification_report(labels, predictions, labels=model.classes_, output_dict=True, zero_division=0)
    metrics = {
        "model": model_label,
        "evaluation": "stratified held-out test split",
        "split": {"test_size": 0.20, "random_state": 42},
        "test_samples": int(len(labels)),
        "features": int(features.shape[1]),
        "classes": classes,
        "accuracy": float(accuracy_score(labels, predictions)),
        "classification_report": report,
        "confusion_matrix_labels": classes,
        "confusion_matrix": matrix.astype(int).tolist(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame(matrix, index=classes, columns=classes).rename_axis("actual").to_csv(output_dir / "confusion_matrix.csv")
    return metrics

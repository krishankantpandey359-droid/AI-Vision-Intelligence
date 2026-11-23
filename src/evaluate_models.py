"""Reproducible held-out evaluation for the gesture classifiers.

Run from the repository root with: python src/evaluate_models.py
The report is written to outputs/model_evaluation.json. It records the
dataset fingerprint, split configuration, per-class metrics and confusion
matrices so the checked-in model's performance is auditable.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset_utils import PROJECT_ROOT, load_feature_dataset


RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


def main():
    X, y = load_feature_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf")),
        ]),
    }

    dataset_path = PROJECT_ROOT / "data" / "gesture_features.csv"
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    labels = sorted(y.unique().tolist())
    minimum_class_count = int(y.value_counts().min())
    effective_cv_folds = min(CV_FOLDS, minimum_class_count)
    if effective_cv_folds < 2:
        raise ValueError(
            "At least two samples per gesture class are required for stratified "
            "cross-validation. Collect more samples before comparing models."
        )
    report = {
        "dataset": dataset_path.relative_to(PROJECT_ROOT).as_posix(),
        "dataset_sha256": dataset_hash,
        "samples": int(len(y)),
        "features": int(X.shape[1]),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "split": {
            "strategy": "stratified_train_test_split",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
        },
        "cross_validation": {
            "strategy": "stratified_k_fold_on_training_partition_only",
            "folds": effective_cv_folds,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "metrics": ["accuracy", "balanced_accuracy", "macro_f1"],
        },
        "models": {},
        "interpretation": (
            "The held-out metrics use one stratified split. Cross-validation is "
            "performed only within the training partition. Samples have no "
            "recorded participant/session identifier, so random splitting may "
            "place related captures in both partitions. Results do not establish "
            "performance on unseen people, cameras, lighting, or real-world use."
        ),
    }

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
    }

    for name, estimator in models.items():
        cv_results = cross_validate(
            estimator,
            X_train,
            y_train,
            cv=StratifiedKFold(
                n_splits=effective_cv_folds, shuffle=True, random_state=RANDOM_STATE
            ),
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro")),
            "classification_report": classification_report(
                y_test, predictions, labels=labels, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=labels
            ).tolist(),
            "cross_validation": {
                metric: {
                    "fold_scores": cv_results[f"test_{metric}"].tolist(),
                    "mean": float(cv_results[f"test_{metric}"].mean()),
                    "std": float(cv_results[f"test_{metric}"].std(ddof=1)),
                }
                for metric in scoring
            },
        }
        report["models"][name] = metrics
        print(f"{name}: accuracy={metrics['accuracy']:.4f}, "
              f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
              f"macro_f1={metrics['macro_f1']:.4f}")
        for metric, cv_summary in metrics["cross_validation"].items():
            print(
                f"  CV {metric}: {cv_summary['mean']:.4f} "
                f"+/- {cv_summary['std']:.4f}"
            )

    output = PROJECT_ROOT / "outputs" / "model_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Evaluation report saved to: {output}")


if __name__ == "__main__":
    main()

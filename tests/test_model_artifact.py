from pathlib import Path
import json
import sys
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from dataset_utils import load_feature_dataset
from model_evaluation import evaluate_model

def test_committed_random_forest_matches_dataset_schema_and_classes():
    model = joblib.load(PROJECT_ROOT / "models" / "gesture_random_forest.pkl")
    features, labels = load_feature_dataset()
    assert getattr(model, "n_features_in_", None) == features.shape[1] == 63
    assert list(model.classes_) == sorted(labels.unique())

def test_held_out_metrics_and_confusion_matrix_are_saved(tmp_path):
    features, labels = load_feature_dataset()
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=42, stratify=labels)
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test, tmp_path, "test model")
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    matrix = np.asarray(saved["confusion_matrix"])
    assert metrics["test_samples"] == 200
    assert metrics["features"] == 63
    assert 0 <= metrics["accuracy"] <= 1
    assert matrix.shape == (5, 5)
    assert matrix.sum() == 200
    assert (tmp_path / "confusion_matrix.csv").is_file()

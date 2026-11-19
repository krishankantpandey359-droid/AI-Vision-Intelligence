"""Evaluate the committed model artifact before CI retrains a fresh model."""
from pathlib import Path
import sys
import joblib
from sklearn.model_selection import train_test_split
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from dataset_utils import load_feature_dataset
from model_evaluation import evaluate_model

features, labels = load_feature_dataset()
_, x_test, _, y_test = train_test_split(features, labels, test_size=0.20, random_state=42, stratify=labels)
model_path = PROJECT_ROOT / "models" / "gesture_random_forest.pkl"
model = joblib.load(model_path)
if getattr(model, "n_features_in_", None) != features.shape[1]:
    raise ValueError("Committed model feature count does not match the dataset.")
if list(model.classes_) != sorted(labels.unique()):
    raise ValueError("Committed model classes do not match the dataset.")
metrics = evaluate_model(model, x_test, y_test, PROJECT_ROOT / "outputs" / "committed_model", "Committed models/gesture_random_forest.pkl")
print("Committed model held-out test accuracy:", f"{metrics['accuracy']:.4f}")
print("Committed model report:", PROJECT_ROOT / "outputs" / "committed_model")

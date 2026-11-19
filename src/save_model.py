"""Train the deployable Random Forest and save its held-out evaluation."""
from pathlib import Path
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from dataset_utils import load_feature_dataset
from model_evaluation import evaluate_model

X, y = load_feature_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
model_path = PROJECT_ROOT / "models" / "gesture_random_forest.pkl"
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_path)
metrics = evaluate_model(model, X_test, y_test, PROJECT_ROOT / "outputs" / "retrained_model", "RandomForestClassifier trained in this workflow")
print("Model training completed successfully.")
print("Model saved at:", model_path)
print("Features:", X.shape[1])
print("Classes:", sorted(y.unique()))
print("Held-out test accuracy:", f"{metrics['accuracy']:.4f}")
print("Evaluation report:", PROJECT_ROOT / "outputs" / "retrained_model")

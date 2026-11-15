from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from dataset_utils import PROJECT_ROOT, load_feature_dataset


X, y = load_feature_dataset()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

model_path = PROJECT_ROOT / "models" / "gesture_random_forest.pkl"
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_path)

print("\\nModel training completed successfully.")
print("Model saved at:", model_path)
print("Number of features:", X.shape[1])
print("Classes:", sorted(y.unique()))

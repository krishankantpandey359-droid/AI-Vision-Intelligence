import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Load dataset
df = pd.read_csv("data/gesture_features.csv")

# Separate features and target
X = df.drop("label", axis=1)
y = df["label"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# Train model
model.fit(X_train, y_train)


# Save trained model
MODEL_PATH = "models/gesture_random_forest.pkl"

joblib.dump(model, MODEL_PATH)


print("\nModel training completed successfully.")
print("Model saved at:", MODEL_PATH)
print("Number of features:", X.shape[1])
print("Classes:", sorted(y.unique()))
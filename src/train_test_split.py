import pandas as pd
from sklearn.model_selection import train_test_split

# Load engineered dataset
DATA_PATH = "data/gesture_features.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# Features and target
X = df.drop("label", axis=1)
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())
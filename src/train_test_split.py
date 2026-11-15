from sklearn.model_selection import train_test_split

from dataset_utils import load_feature_dataset


X, y = load_feature_dataset()
print("Dataset shape:", (len(X), X.shape[1] + 1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print("\\nTraining label distribution:")
print(y_train.value_counts())
print("\\nTesting label distribution:")
print(y_test.value_counts())

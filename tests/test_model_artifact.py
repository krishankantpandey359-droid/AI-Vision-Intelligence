from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset_utils import load_feature_dataset


def test_checked_in_random_forest_matches_dataset_schema_and_classes():
    model_path = PROJECT_ROOT / "models" / "gesture_random_forest.pkl"
    assert model_path.is_file(), "The committed Random Forest model is missing."

    model = joblib.load(model_path)
    features, labels = load_feature_dataset()

    assert getattr(model, "n_features_in_", None) == features.shape[1] == 63
    assert list(model.classes_) == sorted(labels.unique())


def test_checked_in_random_forest_has_finite_held_out_metrics():
    model_path = PROJECT_ROOT / "models" / "gesture_random_forest.pkl"
    model = joblib.load(model_path)
    features, labels = load_feature_dataset()

    _, x_test, _, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=model.classes_)

    assert np.isfinite(accuracy)
    assert 0.0 <= accuracy <= 1.0
    assert matrix.shape == (5, 5)
    assert int(matrix.sum()) == len(y_test) == 200
    assert probabilities.shape == (len(y_test), 5)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(len(y_test)),
        rtol=0,
        atol=1e-6,
    )

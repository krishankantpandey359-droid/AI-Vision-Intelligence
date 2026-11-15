from pathlib import Path
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset_utils import load_feature_dataset


def test_random_forest_training_and_evaluation_pipeline():
    features, labels = load_feature_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )

    assert len(x_train) == 800
    assert len(x_test) == 200
    assert model.n_features_in_ == 63
    assert set(model.classes_) == set(labels.unique())
    assert probabilities.shape == (200, 5)
    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()
    assert abs(probabilities.sum(axis=1) - 1).max() < 1e-9
    assert 0.0 <= accuracy <= 1.0
    assert set(report).issuperset(set(labels.unique()))

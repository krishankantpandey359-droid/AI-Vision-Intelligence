from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset_utils import EXPECTED_FEATURE_COLUMNS, load_feature_dataset
from feature_utils import normalize_landmarks


def test_existing_feature_dataset_loads_and_is_balanced():
    features, labels = load_feature_dataset()
    assert features.shape == (1000, 63)
    assert labels.value_counts().to_dict() == {
        "fist": 200,
        "open_palm": 200,
        "one": 200,
        "two": 200,
        "thumbs_up": 200,
    }
    assert list(features.columns) == EXPECTED_FEATURE_COLUMNS


def test_dataset_loader_rejects_missing_required_columns(tmp_path, monkeypatch):
    path = tmp_path / "features.csv"
    pd.DataFrame({"label": ["fist"]}).to_csv(path, index=False)
    monkeypatch.setattr("dataset_utils.FEATURE_DATA_PATH", path)
    with pytest.raises(ValueError, match="missing required columns"):
        load_feature_dataset()


def test_dataset_loader_rejects_non_numeric_feature_values(tmp_path, monkeypatch):
    path = tmp_path / "features.csv"
    row = {column: 0.0 for column in EXPECTED_FEATURE_COLUMNS}
    row["x_1"] = "not-a-number"
    row["label"] = "fist"
    pd.DataFrame([row]).to_csv(path, index=False)
    monkeypatch.setattr("dataset_utils.FEATURE_DATA_PATH", path)
    with pytest.raises(ValueError, match="missing or non-numeric"):
        load_feature_dataset()


def test_dataset_loader_rejects_missing_labels(tmp_path, monkeypatch):
    path = tmp_path / "features.csv"
    row = {column: 0.0 for column in EXPECTED_FEATURE_COLUMNS}
    row["label"] = ""
    pd.DataFrame([row]).to_csv(path, index=False)
    monkeypatch.setattr("dataset_utils.FEATURE_DATA_PATH", path)
    with pytest.raises(ValueError, match="missing gesture labels"):
        load_feature_dataset()



class Landmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def test_live_feature_normalization_matches_training_schema():
    landmarks = [Landmark(float(i), float(i * 2), float(i * 3)) for i in range(21)]
    features = normalize_landmarks(landmarks)
    assert list(features.columns) == EXPECTED_FEATURE_COLUMNS
    assert features.shape == (1, 63)
    assert features.loc[0, ["x_0", "y_0", "z_0"]].tolist() == [0.0, 0.0, 0.0]
    assert features.to_numpy().min() >= -1.0
    assert features.to_numpy().max() <= 1.0


def test_live_feature_normalization_rejects_wrong_landmark_count():
    with pytest.raises(ValueError, match="exactly 21"):
        normalize_landmarks([Landmark(0, 0, 0)])


def test_live_feature_normalization_rejects_non_finite_coordinates():
    landmarks = [Landmark(0.0, 0.0, 0.0) for _ in range(21)]
    landmarks[4].x = float("nan")
    with pytest.raises(ValueError, match="finite"):
        normalize_landmarks(landmarks)

def test_raw_landmarks_reproduce_committed_engineered_features():
    import numpy as np

    raw_path = PROJECT_ROOT / "data" / "gesture_landmarks.csv"
    engineered_path = PROJECT_ROOT / "data" / "gesture_features.csv"
    raw = pd.read_csv(raw_path)
    engineered = pd.read_csv(engineered_path)

    assert raw.shape == (1000, 64)
    assert engineered.shape == (1000, 64)
    assert raw["label"].astype(str).tolist() == engineered["label"].astype(str).tolist()
    assert raw["label"].value_counts().to_dict() == {
        "fist": 200,
        "open_palm": 200,
        "one": 200,
        "two": 200,
        "thumbs_up": 200,
    }

    raw_values = raw[EXPECTED_FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=float)
    expected_rows = []
    for sample in raw_values:
        points = sample.reshape(21, 3)
        normalized = points - points[0]
        scale = np.linalg.norm(normalized, axis=1).max()
        if scale > 0:
            normalized = normalized / scale
        expected_rows.append(normalized.reshape(-1))

    engineered_features = engineered[EXPECTED_FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    # The committed engineered CSV has one blank wrist-anchor value. The
    # loader normalizes this known anchor to zero, so mirror that behavior.
    engineered_features[["x_0", "y_0", "z_0"]] = engineered_features[
        ["x_0", "y_0", "z_0"]
    ].fillna(0)
    assert not engineered_features.isna().any().any()
    actual = engineered_features.to_numpy(dtype=float)
    np.testing.assert_allclose(
        actual, np.asarray(expected_rows), rtol=0, atol=1e-7
    )

def test_dataset_loader_rejects_missing_non_anchor_values(tmp_path, monkeypatch):
    path = tmp_path / "features.csv"
    row = {column: 0.0 for column in EXPECTED_FEATURE_COLUMNS}
    row["y_0"] = float("nan")
    row["label"] = "fist"
    pd.DataFrame([row]).to_csv(path, index=False)
    monkeypatch.setattr("dataset_utils.FEATURE_DATA_PATH", path)
    with pytest.raises(ValueError, match="missing or non-numeric"):
        load_feature_dataset()


def test_dataset_loader_rejects_infinite_feature_values(tmp_path, monkeypatch):
    path = tmp_path / "features.csv"
    row = {column: 0.0 for column in EXPECTED_FEATURE_COLUMNS}
    row["x_1"] = float("inf")
    row["label"] = "fist"
    pd.DataFrame([row]).to_csv(path, index=False)
    monkeypatch.setattr("dataset_utils.FEATURE_DATA_PATH", path)
    with pytest.raises(ValueError, match="non-finite"):
        load_feature_dataset()


def test_dataset_loader_rejects_missing_x_anchor_when_other_anchor_values_are_nonzero(tmp_path, monkeypatch):
    path = tmp_path / "features.csv"
    row = {column: 0.0 for column in EXPECTED_FEATURE_COLUMNS}
    row["x_0"] = float("nan")
    row["y_0"] = 0.25
    row["label"] = "fist"
    pd.DataFrame([row]).to_csv(path, index=False)
    monkeypatch.setattr("dataset_utils.FEATURE_DATA_PATH", path)
    with pytest.raises(ValueError, match="zero-valued wrist anchor"):
        load_feature_dataset()


def test_dataset_loader_requires_all_wrist_anchor_coordinates_to_be_zero(tmp_path, monkeypatch):
    path = tmp_path / "features.csv"
    row = {column: 0.0 for column in EXPECTED_FEATURE_COLUMNS}
    row["z_0"] = 0.1
    row["label"] = "fist"
    pd.DataFrame([row]).to_csv(path, index=False)
    monkeypatch.setattr("dataset_utils.FEATURE_DATA_PATH", path)
    with pytest.raises(ValueError, match="anchor coordinates must all equal zero"):
        load_feature_dataset()

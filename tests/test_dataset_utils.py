from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset_utils import EXPECTED_FEATURE_COLUMNS, load_feature_dataset


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

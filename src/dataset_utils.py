from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "gesture_features.csv"
EXPECTED_FEATURE_COLUMNS = [
    f"{axis}_{index}"
    for index in range(21)
    for axis in ("x", "y", "z")
]


def load_feature_dataset():
    """Load and validate the project's engineered gesture dataset."""
    if not FEATURE_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_DATA_PATH}"
        )

    frame = pd.read_csv(FEATURE_DATA_PATH)
    required = EXPECTED_FEATURE_COLUMNS + ["label"]
    missing = [column for column in required if column not in frame.columns]

    if missing:
        raise ValueError(
            "Feature dataset is missing required columns: "
            + ", ".join(missing)
        )

    if frame.empty:
        raise ValueError("Feature dataset contains no samples.")

    features = frame[EXPECTED_FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    # Wrist-relative normalization makes the anchor landmark exactly (0, 0, 0).
    # One existing feature row has a blank x_0; normalize that known anchor only.
    anchor_columns = ["x_0", "y_0", "z_0"]
    features[anchor_columns] = features[anchor_columns].fillna(0)
    if features.isna().any().any():
        raise ValueError(
            "Feature dataset contains missing or non-numeric landmark values."
        )

    if not features.apply(lambda column: column.map(lambda value: pd.notna(value) and abs(value) != float("inf"))).all().all():
        raise ValueError("Feature dataset contains non-finite landmark values.")

    if frame["label"].isna().any() or frame["label"].astype(str).str.strip().eq("").any():
        raise ValueError("Feature dataset contains missing gesture labels.")

    return features, frame["label"].astype(str)

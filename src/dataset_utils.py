from pathlib import Path

import numpy as np
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
    # Permit the single documented blank x_0 only when the other anchor
    # coordinates confirm the same zero-valued wrist anchor.
    missing_x_anchor = features["x_0"].isna()
    if (missing_x_anchor & (features["y_0"].ne(0) | features["z_0"].ne(0))).any():
        raise ValueError("A missing x_0 is only valid for a zero-valued wrist anchor.")
    features.loc[missing_x_anchor, "x_0"] = 0.0
    if features.isna().any().any():
        raise ValueError(
            "Feature dataset contains missing or non-numeric landmark values."
        )

    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Feature dataset contains non-finite landmark values.")
    anchor = features[["x_0", "y_0", "z_0"]].to_numpy(dtype=float)
    if not np.allclose(anchor, 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("Wrist-relative anchor coordinates must all equal zero.")

    if frame["label"].isna().any() or frame["label"].astype(str).str.strip().eq("").any():
        raise ValueError("Feature dataset contains missing gesture labels.")

    return features, frame["label"].astype(str)

from pathlib import Path

import numpy as np
import pandas as pd


# Resolve paths from the project root so the script works from any directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "gesture_landmarks.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "gesture_features.csv"

# Load raw landmark data without touching the existing feature dataset.
if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Raw landmark dataset not found: {INPUT_PATH}. "
        "Run src/collect_data.py first."
    )

if INPUT_PATH.stat().st_size == 0:
    raise ValueError(
        f"Raw landmark dataset is empty: {INPUT_PATH}. "
        "Collect gesture samples before running feature engineering. "
        "The existing gesture_features.csv file has not been changed."
    )

df = pd.read_csv(INPUT_PATH)
print("Raw dataset shape:", df.shape)

if df.empty:
    raise ValueError(
        "Raw landmark dataset contains no samples. "
        "Collect gesture samples before running feature engineering. "
        "The existing gesture_features.csv file has not been changed."
    )

# Validate the complete set of expected landmark columns and target label.
feature_columns = [
    f"{axis}_{i}"
    for i in range(21)
    for axis in ("x", "y", "z")
]
required_columns = feature_columns + ["label"]
missing_columns = sorted(set(required_columns).difference(df.columns))

if missing_columns:
    raise ValueError(
        "Raw landmark dataset is missing required columns: "
        + ", ".join(missing_columns)
    )

if df[required_columns].isnull().any().any():
    raise ValueError(
        "Raw landmark dataset contains missing landmark values or labels. "
        "Correct the source data before feature engineering."
    )

try:
    landmark_values = df[feature_columns].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=float)
except (TypeError, ValueError) as exc:
    raise ValueError(
        "Landmark columns must contain numeric values."
    ) from exc

if not np.isfinite(landmark_values).all():
    raise ValueError(
        "Landmark columns contain non-finite values. "
        "Correct the source data before feature engineering."
    )

labels = df["label"].reset_index(drop=True)

# Convert each sample to wrist-relative, scale-normalized coordinates.
feature_data = []

for sample in landmark_values:
    points = sample.reshape(21, 3)
    normalized = points - points[0]

    distances = np.linalg.norm(normalized, axis=1)
    scale = np.max(distances)

    if scale > 0:
        normalized = normalized / scale

    feature_data.append(normalized.reshape(-1))

features_df = pd.DataFrame(feature_data, columns=feature_columns)
features_df["label"] = labels

# Write only after all input validation and transformations succeed.
features_df.to_csv(OUTPUT_PATH, index=False)

print("\\nFeature engineering completed successfully.")
print("New dataset shape:", features_df.shape)
print("Saved at:", OUTPUT_PATH)

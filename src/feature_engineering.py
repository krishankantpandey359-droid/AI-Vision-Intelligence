import pandas as pd
import numpy as np

# File paths
INPUT_PATH = "data/gesture_landmarks.csv"
OUTPUT_PATH = "data/gesture_features.csv"

# Load raw landmark dataset
df = pd.read_csv(INPUT_PATH)

print("Raw dataset shape:", df.shape)

# Separate features and label
labels = df["label"]

feature_data = []

for _, row in df.iterrows():

    # Wrist coordinates = landmark 0
    wrist_x = row["x_0"]
    wrist_y = row["y_0"]
    wrist_z = row["z_0"]

    normalized = []

    # Make every landmark relative to wrist
    for i in range(21):

        x = row[f"x_{i}"] - wrist_x
        y = row[f"y_{i}"] - wrist_y
        z = row[f"z_{i}"] - wrist_z

        normalized.extend([x, y, z])

    # Convert to numpy array
    normalized = np.array(normalized)

    # Scale normalization
    points = normalized.reshape(21, 3)

    distances = np.sqrt(
        points[:, 0] ** 2 +
        points[:, 1] ** 2 +
        points[:, 2] ** 2
    )

    scale = np.max(distances)

    if scale > 0:
        normalized = normalized / scale

    feature_data.append(normalized)

# Create feature column names
feature_columns = []

for i in range(21):
    feature_columns.extend([
        f"x_{i}",
        f"y_{i}",
        f"z_{i}"
    ])

# Create final dataframe
features_df = pd.DataFrame(
    feature_data,
    columns=feature_columns
)

# Add target label
features_df["label"] = labels.values

# Save engineered dataset
features_df.to_csv(OUTPUT_PATH, index=False)

print("\nFeature engineering completed successfully.")
print("New dataset shape:", features_df.shape)
print("Saved at:", OUTPUT_PATH)
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    f"{axis}_{index}"
    for index in range(21)
    for axis in ("x", "y", "z")
]


def normalize_landmarks(landmarks):
    """Convert 21 MediaPipe landmarks to the trained 63-feature format."""
    points = np.asarray(
        [[point.x, point.y, point.z] for point in landmarks],
        dtype=float,
    )

    if points.shape != (21, 3):
        raise ValueError("Expected exactly 21 landmarks with x, y and z values.")

    if not np.isfinite(points).all():
        raise ValueError("Landmark coordinates must be finite numbers.")

    normalized = points - points[0]
    scale = np.linalg.norm(normalized, axis=1).max()

    if scale > 0:
        normalized = normalized / scale

    return pd.DataFrame(
        normalized.reshape(1, -1),
        columns=FEATURE_COLUMNS,
    )

import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
import csv
import time

from pathlib import Path
from collections import deque


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
ML_MODEL_PATH = PROJECT_ROOT / "models" / "gesture_random_forest.pkl"

LOG_PATH = PROJECT_ROOT / "logs" / "predictions.csv"


# ============================================================
# LOAD TRAINED ML MODEL
# ============================================================

model = joblib.load(ML_MODEL_PATH)

print("ML model loaded successfully.")
print("Number of features:", model.n_features_in_)
print("Classes:", model.classes_)


# ============================================================
# PREDICTION SMOOTHING
# ============================================================

prediction_history = deque(maxlen=7)

last_prediction = "Show your hand"


# ============================================================
# LOGGING SETTINGS
# ============================================================

last_logged_time = 0

LOG_INTERVAL = 1.0


# ============================================================
# CREATE LOG FILE IF IT DOES NOT EXIST
# ============================================================

if not LOG_PATH.exists():

    with open(
        LOG_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "gesture",
            "confidence"
        ])


# ============================================================
# MEDIAPIPE HAND LANDMARKER SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)


# ============================================================
# HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [

    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17)
]


# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

print("Starting AI Vision Intelligence System...")
print("Press Q to quit.")


# ============================================================
# START MEDIAPIPE
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:

            print(
                "Camera frame could not be read."
            )

            break


        # ----------------------------------------------------
        # CONVERT BGR TO RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # CREATE MEDIAPIPE IMAGE
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ----------------------------------------------------
        # DETECT HAND
        # ----------------------------------------------------

        result = landmarker.detect(mp_image)


        # ====================================================
        # IF HAND IS DETECTED
        # ====================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]


            # ------------------------------------------------
            # FEATURE ENGINEERING
            # ------------------------------------------------

            wrist_x = hand[0].x
            wrist_y = hand[0].y
            wrist_z = hand[0].z

            normalized = []


            for landmark in hand:

                normalized.extend([
                    landmark.x - wrist_x,
                    landmark.y - wrist_y,
                    landmark.z - wrist_z
                ])


            # ------------------------------------------------
            # CONVERT TO 21 x 3
            # ------------------------------------------------

            points = np.array(
                normalized
            ).reshape(21, 3)


            # ------------------------------------------------
            # SCALE NORMALIZATION
            # ------------------------------------------------

            distances = np.sqrt(
                points[:, 0] ** 2 +
                points[:, 1] ** 2 +
                points[:, 2] ** 2
            )


            scale = np.max(distances)


            if scale > 0:

                normalized = (
                    np.array(normalized) / scale
                )


            # ------------------------------------------------
            # FINAL 63 FEATURES
            # ------------------------------------------------

            features = np.array(
                normalized
            ).reshape(1, -1)


            # =================================================
            # ML PREDICTION
            # =================================================

            features_df = pd.DataFrame(
                features,
                columns=model.feature_names_in_
            )


            prediction = model.predict(
                features_df
            )[0]


            probabilities = model.predict_proba(
                features_df
            )[0]


            confidence = float(
                np.max(probabilities)
            )


            # =================================================
            # PREDICTION SMOOTHING
            # =================================================

            prediction_history.append(
                prediction
            )


            counts = {}


            for item in prediction_history:

                counts[item] = (
                    counts.get(item, 0) + 1
                )


            smoothed_prediction = max(
                counts,
                key=counts.get
            )


            last_prediction = (
                smoothed_prediction
            )


            # =================================================
            # SAVE PREDICTION TO CSV
            # =================================================

            current_time = time.time()


            if (
                current_time - last_logged_time
                >= LOG_INTERVAL
            ):

                timestamp = time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )


                with open(
                    LOG_PATH,
                    "a",
                    newline="",
                    encoding="utf-8"
                ) as file:

                    writer = csv.writer(file)

                    writer.writerow([
                        timestamp,
                        last_prediction,
                        round(
                            confidence * 100,
                            2
                        )
                    ])


                last_logged_time = (
                    current_time
                )


            # =================================================
            # DRAW LANDMARKS
            # =================================================

            h, w, _ = frame.shape

            points_2d = []


            for landmark in hand:

                x = int(
                    landmark.x * w
                )

                y = int(
                    landmark.y * h
                )


                points_2d.append(
                    (x, y)
                )


                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


            # ------------------------------------------------
            # DRAW CONNECTIONS
            # ------------------------------------------------

            for start, end in HAND_CONNECTIONS:

                cv2.line(
                    frame,
                    points_2d[start],
                    points_2d[end],
                    (255, 0, 0),
                    2
                )


            # =================================================
            # DISPLAY PREDICTION
            # =================================================

            gesture_text = (
                f"Gesture: {last_prediction}"
            )


            confidence_text = (
                f"Confidence: "
                f"{confidence * 100:.1f}%"
            )


            cv2.putText(
                frame,
                gesture_text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )


            cv2.putText(
                frame,
                confidence_text,
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )


        # ====================================================
        # NO HAND DETECTED
        # ====================================================

        else:

            prediction_history.clear()

            last_prediction = (
                "Show your hand"
            )


            cv2.putText(
                frame,
                "Show your hand",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )


        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(
            "AI Vision Intelligence - Live Prediction",
            frame
        )


        # ====================================================
        # PRESS Q TO QUIT
        # ====================================================

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):

            break


# ============================================================
# RELEASE CAMERA
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("Live prediction stopped.")
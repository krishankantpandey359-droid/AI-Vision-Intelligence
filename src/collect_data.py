import cv2
import mediapipe as mp
import pandas as pd
import os
import time

MODEL_PATH = "models/hand_landmarker.task"
DATA_PATH = "data/gesture_landmarks.csv"

# Gesture classes
GESTURES = {
    "1": "fist",
    "2": "open_palm",
    "3": "one",
    "4": "two",
    "5": "thumbs_up"
}

# Create MediaPipe Hand Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

# Create data folder if needed
os.makedirs("data", exist_ok=True)

# Create column names
columns = []

for i in range(21):
    columns.extend([
        f"x_{i}",
        f"y_{i}",
        f"z_{i}"
    ])

columns.append("label")

# Load existing data if available
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    df = pd.DataFrame(columns=columns)

print("\n====================================")
print(" AI Vision Intelligence")
print(" Gesture Data Collector")
print("====================================")
print("1 = fist")
print("2 = open_palm")
print("3 = one")
print("4 = two")
print("5 = thumbs_up")
print("Q = quit")
print("====================================\n")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    landmarker.close()
    exit()

current_gesture = None
samples_collected = 0
target_samples = 200
last_capture_time = 0

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read frame.")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect(mp_image)

    # Select gesture using number keys
    key = cv2.waitKey(1) & 0xFF

    if key in [ord("1"), ord("2"), ord("3"), ord("4"), ord("5")]:
        current_gesture = GESTURES[chr(key)]
        samples_collected = 0

        print(f"\nCollecting gesture: {current_gesture}")
        print(f"Target samples: {target_samples}")

    # Quit
    if key == ord("q"):
        break

    # Draw and collect landmarks
    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        height, width, _ = frame.shape

        # Draw landmark points
        for landmark in hand:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

        # Automatically collect one sample every 0.1 second
        current_time = time.time()

        if (
            current_gesture is not None
            and samples_collected < target_samples
            and current_time - last_capture_time >= 0.1
        ):

            row = []

            for landmark in hand:
                row.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            row.append(current_gesture)

            df.loc[len(df)] = row

            samples_collected += 1
            last_capture_time = current_time

            print(
                f"{current_gesture}: "
                f"{samples_collected}/{target_samples}"
            )

            if samples_collected == target_samples:
                print(
                    f"\nCompleted: {current_gesture}"
                )

                df.to_csv(DATA_PATH, index=False)

    # Display information
    if current_gesture:
        text = (
            f"Gesture: {current_gesture} | "
            f"Samples: {samples_collected}/{target_samples}"
        )
    else:
        text = "Press 1-5 to select a gesture"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "AI Vision - Data Collection",
        frame
    )

# Save final dataset
df.to_csv(DATA_PATH, index=False)

camera.release()
landmarker.close()
cv2.destroyAllWindows()

print("\n====================================")
print("Data collection completed.")
print(f"Total samples: {len(df)}")
print(f"Dataset saved at: {DATA_PATH}")
print("====================================")
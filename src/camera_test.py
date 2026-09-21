import cv2
import mediapipe as mp
from pathlib import Path
 
 
# Project ke models folder ka path
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
 
 
# Hand Landmarker setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
 
 
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    num_hands=2
)
 
 
# Hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]
 
 
# Camera start
camera = cv2.VideoCapture(0)
 
if not camera.isOpened():
    print("Camera could not be opened.")
    exit()
 
 
# Hand Landmarker start
with HandLandmarker.create_from_options(options) as landmarker:
 
    while True:
 
        success, frame = camera.read()
 
        if not success:
            print("Could not read frame.")
            break
 
        # Mirror effect
        frame = cv2.flip(frame, 1)
 
        # BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
        # MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )
 
        # Detect hands
        result = landmarker.detect(mp_image)
 
        height, width, _ = frame.shape
 
        # Draw landmarks
        for hand_landmarks in result.hand_landmarks:
 
            points = []
 
            for landmark in hand_landmarks:
 
                x = int(landmark.x * width)
                y = int(landmark.y * height)
 
                points.append((x, y))
 
                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )
 
            # Draw connections
            for start, end in HAND_CONNECTIONS:
 
                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 0, 0),
                    2
                )
 
 
        # Display
        cv2.imshow(
            "AI Vision - Hand Landmarks",
            frame
        )
 
 
        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
 
 
# Cleanup
camera.release()
cv2.destroyAllWindows()
 
print("Camera test completed successfully.")
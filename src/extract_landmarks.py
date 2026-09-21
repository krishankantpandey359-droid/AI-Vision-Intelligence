import cv2
import mediapipe as mp

MODEL_PATH = "models/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    landmarker.close()
    exit()

print("Camera started. Show your hand.")
print("Press Q to quit.")

printed_landmarks = False

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

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        height, width, _ = frame.shape

        for index, landmark in enumerate(hand):

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            if not printed_landmarks:
                print(
                    f"Landmark {index}: "
                    f"x={landmark.x:.4f}, "
                    f"y={landmark.y:.4f}, "
                    f"z={landmark.z:.4f}"
                )

        printed_landmarks = True

        cv2.putText(
            frame,
            "Hand Detected - 21 Landmarks",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("AI Vision - Landmark Extraction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
landmarker.close()
cv2.destroyAllWindows()

print("Landmark extraction test completed successfully.")